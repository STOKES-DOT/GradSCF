// Analytic full/selected ERI coefficient and exponent products. Four independent
// shadow contractions preserve repeated-shell legs and individual zero
// coefficients. Exponents use -r^2 Gaussian angular raising, never differences.
#include "eri_basis.h"
#include "parallel.h"
extern "C" {
double *CINTc2s_ket_sph(double*,int,double*,int);
double CINTcommon_fac_sp(int);
}

struct ERICoefficientTables:ERIBasisTables {
  int cart;
  size_t count=0,capacity=0;
  std::vector<size_t> offsets;
  std::vector<int> primitives,contractions,angular;
  std::array<int,4> shadows,pointers;
  Intor intor;

  ERICoefficientTables(ffi::BufferR2<ffi::S32> atoms,ffi::BufferR2<ffi::S32> basis,
      ffi::BufferR1<ffi::F64> env,ffi::BufferR1<ffi::F64> coefficients,int representation,
      const int32_t* slice=nullptr,size_t slice_count=0)
      :ERIBasisTables(atoms,basis,env,representation,slice,slice_count),cart(representation) {
    for (int s=0;s<active;++s) {
      const int* row=t.bas.data()+s*BAS_SLOTS;
      int np=row[NPRIM_OF],nc=row[NCTR_OF],l=row[ANG_OF];
      offsets.push_back(count);primitives.push_back(np);contractions.push_back(nc);
      angular.push_back(cart ? eri_ncart(l):2*l+1);
      count+=size_t(np)*nc;
      if (selected(s)) capacity=std::max(capacity,size_t(np)*std::max(np,nc));
    }
    if (coefficients.element_count()!=count)
      throw std::invalid_argument("ERI coefficient vector has an incorrect shape");
    finite(coefficients.typed_data(),count);
    if (t.env.size()+count+4*capacity>size_t(std::numeric_limits<int>::max()) ||
        active>std::numeric_limits<int>::max()-4)
      throw std::invalid_argument("ERI coefficient shadow storage exceeds int32 indexing");
    for (int s=0;s<active;++s) {
      int* row=t.bas.data()+s*BAS_SLOTS;
      row[PTR_COEFF]=int(t.env.size());
      const double* source=coefficients.typed_data()+offsets[s];
      t.env.insert(t.env.end(),source,source+size_t(primitives[s])*contractions[s]);
    }
    t.bas.resize(size_t(active+4)*BAS_SLOTS);
    for (int leg=0;leg<4;++leg) {
      shadows[leg]=active+leg;pointers[leg]=int(t.env.size());
      t.env.resize(t.env.size()+capacity,0.);
    }
    t.nbas+=4;intor=cart ? int2e_cart:int2e_sph;
  }
  std::array<int,4> evaluate(const std::array<int,4>& original,
      const std::array<const double*,4>& directions,int identity=-1) {
    std::array<int,4> dims;
    for (int leg=0;leg<4;++leg) {
      int s=original[leg],np=primitives[s],nc=contractions[s];
      const int* source=t.bas.data()+s*BAS_SLOTS;
      int* row=t.bas.data()+shadows[leg]*BAS_SLOTS;
      std::copy(source,source+BAS_SLOTS,row);row[PTR_COEFF]=pointers[leg];
      double* values=t.env.data()+pointers[leg];
      if (leg==identity) {
        row[NCTR_OF]=np;std::fill(values,values+size_t(np)*np,0.);
        for (int p=0;p<np;++p) values[size_t(p)*np+p]=1.;
      } else {
        const double* coefficients=directions[leg] ? directions[leg]+offsets[s]
                                                   :t.env.data()+source[PTR_COEFF];
        std::copy(coefficients,coefficients+size_t(np)*nc,values);
      }
      dims[leg]=angular[s]*(leg==identity ? np:nc);
    }
    // No optimizer: nonzero masks from the original contraction can erase
    // genuine derivatives when a shadow direction replaces a zero coefficient.
    shell(intor,shadows,dims);return dims;
  }
  void directional(const double* left,const double* right,double* output) {
    quartets([&](const auto& shells) {
      const auto dims=contracted_dims(shells);reset_contracted(dims);
      for (int a=0;a<4;++a) for (int b=0;b<(right ? 4:1);++b) {
        if (right && a==b) continue;
        std::array<const double*,4> directions{};directions[a]=left;
        if (right) directions[b]=right;
        evaluate(shells,directions);
        for (size_t index=0;index<contracted.size();++index) contracted[index]+=block[index];
      }
      // Only the complete sum is covariant under the shell stabilizer. In
      // particular, diagonal shells still contribute four/twelve leg terms.
      orbit<false>(shells,dims,nullptr,output);
    });
  }
  void transpose(const double* direction,const double* cotangent,double* output) {
    quartets([&](const auto& shells) {
      const auto dims=contracted_dims(shells);reset_contracted(dims);
      orbit<true>(shells,dims,cotangent,nullptr);
      for (int free=0;free<4;++free)
        for (int leg=0;leg<(direction ? 4:1);++leg) {
          if (direction && free==leg) continue;
          std::array<const double*,4> directions{};
          if (direction) directions[leg]=direction;
          const auto free_dims=evaluate(shells,directions,free);
          int s=shells[free],np=primitives[s],nc=contractions[s],na=angular[s];
          eri_local_entries(free_dims,[&](const auto& local,size_t index) {
            int primitive=local[free]/na,angular_index=local[free]%na;
            auto target=local;
            for (int contraction=0;contraction<nc;++contraction) {
              target[free]=contraction*na+angular_index;
              // Identity-shadow dimensions describe primitives; the pooled
              // cotangent retains the physical contracted AO dimensions.
              output[offsets[s]+size_t(contraction)*np+primitive]+=
                block[index]*contracted[local_index(dims,target)];
            }
          });
        }
    });
  }
};

static std::vector<std::array<int,3>> eri_powers(int l) {
  std::vector<std::array<int,3>> out;
  for (int x=l;x>=0;--x) for (int y=l-x;y>=0;--y) out.push_back({x,y,l-x-y});
  return out;
}

struct ERIExponentTables:ERIBasisTables {
  int cart,shadow,pointer;
  size_t count=0;
  std::vector<size_t> offsets;
  std::array<std::vector<double>,13> transformations;
  std::array<std::vector<std::array<int,3>>,11> raise_indices;
  std::vector<double> reduced,scratch;

  ERIExponentTables(ffi::BufferR2<ffi::S32> atoms,ffi::BufferR2<ffi::S32> basis,
      ffi::BufferR1<ffi::F64> env,int representation,
      const int32_t* slice=nullptr,size_t slice_count=0)
      :ERIBasisTables(atoms,basis,env,representation,slice,slice_count),cart(representation) {
    int capacity=0;
    for (int s=0;s<active;++s) {
      const int* row=t.bas.data()+s*BAS_SLOTS;
      if (selected(s) && row[ANG_OF]>10)
        throw std::invalid_argument("ERI exponent angular raising requires selected l <= 10");
      offsets.push_back(count);count+=row[NPRIM_OF];
      if (selected(s)) capacity=std::max(capacity,row[NCTR_OF]);
    }
    if (active==std::numeric_limits<int>::max() ||
        t.env.size()+capacity>size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("ERI exponent shadow storage exceeds int32 indexing");
    shadow=t.nbas++;pointer=int(t.env.size());
    t.bas.resize(size_t(t.nbas)*BAS_SLOTS);t.env.resize(t.env.size()+capacity,0.);
  }
  const std::vector<double>& transform(int l) {
    auto& matrix=transformations[l];
    if (matrix.empty()) {
      int nc=eri_ncart(l),ns=2*l+1;
      std::vector<double> identity(size_t(nc)*nc,0.),out(size_t(nc)*ns);
      for (int i=0;i<nc;++i) identity[size_t(i)*nc+i]=1.;
      double* values=CINTc2s_ket_sph(out.data(),nc,identity.data(),l);
      matrix.assign(values,values+size_t(nc)*ns);
    }
    return matrix;
  }
  void spherical(std::array<int,4>& dims,const std::array<int,4>& shells) {
    size_t stride=1;
    for (int axis=0;axis<4;++axis) {
      const int* row=t.bas.data()+shells[axis]*BAS_SLOTS;
      int l=row[ANG_OF],nc=eri_ncart(l),ns=2*l+1,contractions=row[NCTR_OF];
      int old=dims[axis],next=contractions*ns;
      size_t outer=reduced.size()/(stride*old);
      scratch.assign(outer*stride*next,0.);
      const auto& matrix=transform(l);
      for (size_t o=0;o<outer;++o) for (int c=0;c<contractions;++c)
        for (int s=0;s<ns;++s) for (int p=0;p<nc;++p) {
          double weight=matrix[size_t(s)*nc+p];
          if (weight==0.) continue;
          for (size_t q=0;q<stride;++q)
            scratch[(o*next+c*ns+s)*stride+q]+=
              weight*reduced[(o*old+c*nc+p)*stride+q];
        }
      reduced.swap(scratch);dims[axis]=next;stride*=next;
    }
  }
  std::array<int,4> evaluate(const std::array<int,4>& original,int leg,int primitive) {
    int s=original[leg];const int* source=t.bas.data()+s*BAS_SLOTS;
    int l=source[ANG_OF],np=source[NPRIM_OF],nc=source[NCTR_OF];
    int* row=t.bas.data()+shadow*BAS_SLOTS;
    std::copy(source,source+BAS_SLOTS,row);
    row[ANG_OF]=l+2;row[NPRIM_OF]=1;row[PTR_EXP]+=primitive;row[PTR_COEFF]=pointer;
    double ratio=CINTcommon_fac_sp(l)/CINTcommon_fac_sp(l+2);
    for (int c=0;c<nc;++c) t.env[pointer+c]=ratio*t.env[source[PTR_COEFF]+c*np+primitive];
    auto shells=original;shells[leg]=shadow;
    std::array<int,4> dims,raised;
    for (int a=0;a<4;++a) {
      const int* base=t.bas.data()+original[a]*BAS_SLOTS;
      const int* current=t.bas.data()+shells[a]*BAS_SLOTS;
      dims[a]=eri_ncart(base[ANG_OF])*base[NCTR_OF];
      raised[a]=eri_ncart(current[ANG_OF])*current[NCTR_OF];
    }
    shell(int2e_cart,shells,raised);
    auto& indices=raise_indices[l];
    if (indices.empty()) {
      auto lower=eri_powers(l),higher=eri_powers(l+2);indices.resize(lower.size());
      for (size_t p=0;p<lower.size();++p) for (int axis=0;axis<3;++axis) {
        auto power=lower[p];power[axis]+=2;
        indices[p][axis]=int(std::find(higher.begin(),higher.end(),power)-higher.begin());
      }
    }
    size_t size=1;for (int d:dims) size*=d;
    reduced.assign(size,0.);
    eri_local_entries(dims,[&](const auto& local,size_t index) {
      int contraction=local[leg]/eri_ncart(l),angular=local[leg]%eri_ncart(l);
      for (int axis=0;axis<3;++axis) {
        auto shifted=local;shifted[leg]=contraction*eri_ncart(l+2)+indices[angular][axis];
        size_t source_index=shifted[3];
        for (int a=2;a>=0;--a) source_index=source_index*raised[a]+shifted[a];
        reduced[index]-=block[source_index];
      }
    });
    if (!cart) spherical(dims,original);
    return dims;
  }
  template<bool transpose>
  void apply(const double* input,double* output) {
    quartets([&](const auto& shells) {
      const auto dims=contracted_dims(shells);reset_contracted(dims);
      if constexpr (transpose) orbit<true>(shells,dims,input,nullptr);
      for (int leg=0;leg<4;++leg) {
        int s=shells[leg],np=t.bas[s*BAS_SLOTS+NPRIM_OF];
        for (int p=0;p<np;++p) {
          size_t parameter=offsets[s]+p;
          if constexpr (!transpose) if (input[parameter]==0.) continue;
          const auto evaluated=evaluate(shells,leg,p);double value=0.;
          eri_local_entries(evaluated,[&](const auto&,size_t index) {
            if constexpr (transpose) value+=contracted[index]*reduced[index];
            else contracted[index]+=input[parameter]*reduced[index];
          });
          if constexpr (transpose) output[parameter]+=value;
        }
      }
      if constexpr (!transpose) orbit<false>(shells,dims,nullptr,output);
    });
  }
};

// Canonical outer pairs own disjoint full/window tensor blocks. Forward
// workers share the output; adjoints replicate only the small parameter vector.
template<bool transpose,class Driver,class Apply>
static void parallel_eri_basis(Driver& driver,double* output,Apply apply) {
  const int workers=native_threads(driver.symmetry.tasks());
  if (workers==1) {apply(driver,output);return;}
  std::vector<std::vector<double>> partial;
  if constexpr (transpose) partial.resize(workers);
  std::vector<std::exception_ptr> errors(workers);
  std::vector<std::thread> threads;
  auto run=[&](int worker) {
    try {
      Driver local=driver;
      local.first_shell=worker;local.shell_stride=workers;
      if constexpr (transpose) {
        partial[worker].assign(driver.count,0.);
        apply(local,partial[worker].data());
      } else apply(local,output);
    } catch (...) {errors[worker]=std::current_exception();}
  };
  try {
    for (int w=1;w<workers;++w) threads.emplace_back(run,w);
  } catch (...) {
    for (auto& thread:threads) thread.join();
    throw;
  }
  run(0);
  for (auto& thread:threads) thread.join();
  for (const auto& error:errors) if (error) std::rethrow_exception(error);
  if constexpr (transpose)
    for (const auto& values:partial)
      for (size_t p=0;p<driver.count;++p) output[p]+=values[p];
}

template<bool transpose,bool second>
static ffi::Error ERICoefficientProduct(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> env,
    ffi::BufferR1<ffi::F64> coefficients,ffi::Buffer<ffi::F64> direction,
    ffi::Buffer<ffi::F64> vector,ffi::ResultBuffer<ffi::F64> result,int32_t cart,
    const int32_t* slice=nullptr,size_t slice_count=0) {
  try {
    ERICoefficientTables driver(atoms,basis,env,coefficients,cart,slice,slice_count);
    if constexpr (second) {
      driver.check_vector(direction,driver.count);
      driver.finite(direction.typed_data(),direction.element_count());
    }
    if constexpr (transpose) {driver.check_integrals(vector);driver.check_vector(*result,driver.count);}
    else {driver.check_vector(vector,driver.count);driver.check_integrals(*result);}
    driver.finite(vector.typed_data(),vector.element_count());
    double* output=result->typed_data();std::fill(output,output+result->element_count(),0.);
    const double* left=second ? direction.typed_data():nullptr;
    parallel_eri_basis<transpose>(driver,output,[&](auto& local,double* out) {
      if constexpr (transpose) local.transpose(left,vector.typed_data(),out);
      else local.directional(second ? left:vector.typed_data(),second ? vector.typed_data():nullptr,out);
    });
    return ffi::Error::Success();
  } catch (const std::invalid_argument& e) {return ffi::Error::InvalidArgument(e.what());}
    catch (const std::exception& e) {return ffi::Error::Internal(e.what());}
}

template<bool transpose>
static ffi::Error ERICoefficientFirst(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> env,
    ffi::BufferR1<ffi::F64> coefficients,ffi::Buffer<ffi::F64> vector,
    ffi::ResultBuffer<ffi::F64> result,int32_t cart) {
  return ERICoefficientProduct<transpose,false>(atoms,basis,env,coefficients,vector,vector,result,cart);
}

template<bool transpose>
static ffi::Error ERICoefficientSecond(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> env,
    ffi::BufferR1<ffi::F64> coefficients,ffi::Buffer<ffi::F64> direction,
    ffi::Buffer<ffi::F64> vector,ffi::ResultBuffer<ffi::F64> result,int32_t cart) {
  return ERICoefficientProduct<transpose,true>(atoms,basis,env,coefficients,direction,vector,result,cart);
}

template<bool transpose>
static ffi::Error ERIExponentProduct(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> env,
    ffi::Buffer<ffi::F64> vector,ffi::ResultBuffer<ffi::F64> result,int32_t cart,
    const int32_t* slice=nullptr,size_t slice_count=0) {
  try {
    ERIExponentTables driver(atoms,basis,env,cart,slice,slice_count);
    if constexpr (transpose) {driver.check_integrals(vector);driver.check_vector(*result,driver.count);}
    else {driver.check_vector(vector,driver.count);driver.check_integrals(*result);}
    driver.finite(vector.typed_data(),vector.element_count());
    double* output=result->typed_data();std::fill(output,output+result->element_count(),0.);
    parallel_eri_basis<transpose>(driver,output,[&](auto& local,double* out) {
      local.template apply<transpose>(vector.typed_data(),out);
    });
    return ffi::Error::Success();
  } catch (const std::invalid_argument& e) {return ffi::Error::InvalidArgument(e.what());}
    catch (const std::exception& e) {return ffi::Error::Internal(e.what());}
}

template<bool transpose>
static ffi::Error ERIExponentFull(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> env,
    ffi::Buffer<ffi::F64> vector,ffi::ResultBuffer<ffi::F64> result,int32_t cart) {
  return ERIExponentProduct<transpose>(atoms,basis,env,vector,result,cart);
}

#define ERI_BASIS_SYMBOL(name) \
 extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* name(XLA_FFI_CallFrame*)
ERI_BASIS_SYMBOL(GradSCFERICoefficientJVP);
ERI_BASIS_SYMBOL(GradSCFERICoefficientVJP);
ERI_BASIS_SYMBOL(GradSCFERICoefficientHessianJVP);
ERI_BASIS_SYMBOL(GradSCFERICoefficientHessianVJP);
ERI_BASIS_SYMBOL(GradSCFERIExponentJVP);
ERI_BASIS_SYMBOL(GradSCFERIExponentVJP);
#define ERI_BASIS_BIND ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>() \
 .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>()
#define ERI_COEFFICIENT_BIND ERI_BASIS_BIND.Arg<ffi::BufferR1<ffi::F64>>() \
 .Arg<ffi::Buffer<ffi::F64>>()
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERICoefficientJVP,ERICoefficientFirst<false>,
 ERI_COEFFICIENT_BIND.Ret<ffi::Buffer<ffi::F64>>().Attr<int32_t>("cart"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERICoefficientVJP,ERICoefficientFirst<true>,
 ERI_COEFFICIENT_BIND.Ret<ffi::Buffer<ffi::F64>>().Attr<int32_t>("cart"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERICoefficientHessianJVP,ERICoefficientSecond<false>,
 ERI_COEFFICIENT_BIND.Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>().Attr<int32_t>("cart"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERICoefficientHessianVJP,ERICoefficientSecond<true>,
 ERI_COEFFICIENT_BIND.Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>().Attr<int32_t>("cart"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERIExponentJVP,ERIExponentFull<false>,
 ERI_BASIS_BIND.Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>().Attr<int32_t>("cart"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERIExponentVJP,ERIExponentFull<true>,
 ERI_BASIS_BIND.Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>().Attr<int32_t>("cart"));

template<bool transpose>
static ffi::Error ERISliceCoefficientFirst(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> env,
    ffi::BufferR1<ffi::F64> coefficients,ffi::Buffer<ffi::F64> vector,
    ffi::ResultBuffer<ffi::F64> result,int32_t cart,ffi::Span<const int32_t> slice) {
  if (slice.size()!=8) return ffi::Error::InvalidArgument("ERI basis shell_slice must have eight endpoints");
  return ERICoefficientProduct<transpose,false>(atoms,basis,env,coefficients,vector,vector,
                                               result,cart,slice.begin(),slice.size());
}
template<bool transpose>
static ffi::Error ERISliceCoefficientSecond(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> env,
    ffi::BufferR1<ffi::F64> coefficients,ffi::Buffer<ffi::F64> direction,
    ffi::Buffer<ffi::F64> vector,ffi::ResultBuffer<ffi::F64> result,
    int32_t cart,ffi::Span<const int32_t> slice) {
  if (slice.size()!=8) return ffi::Error::InvalidArgument("ERI basis shell_slice must have eight endpoints");
  return ERICoefficientProduct<transpose,true>(atoms,basis,env,coefficients,direction,vector,
                                              result,cart,slice.begin(),slice.size());
}
template<bool transpose>
static ffi::Error ERISliceExponent(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> env,
    ffi::Buffer<ffi::F64> vector,ffi::ResultBuffer<ffi::F64> result,
    int32_t cart,ffi::Span<const int32_t> slice) {
  if (slice.size()!=8) return ffi::Error::InvalidArgument("ERI basis shell_slice must have eight endpoints");
  return ERIExponentProduct<transpose>(atoms,basis,env,vector,result,cart,slice.begin(),slice.size());
}
ERI_BASIS_SYMBOL(GradSCFERISliceCoefficientJVP);
ERI_BASIS_SYMBOL(GradSCFERISliceCoefficientVJP);
ERI_BASIS_SYMBOL(GradSCFERISliceCoefficientHessianJVP);
ERI_BASIS_SYMBOL(GradSCFERISliceCoefficientHessianVJP);
ERI_BASIS_SYMBOL(GradSCFERISliceExponentJVP);
ERI_BASIS_SYMBOL(GradSCFERISliceExponentVJP);
#define ERI_SLICE_ATTR .Attr<int32_t>("cart").Attr<ffi::Span<const int32_t>>("shell_slice")
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERISliceCoefficientJVP,ERISliceCoefficientFirst<false>,
 ERI_COEFFICIENT_BIND.Ret<ffi::Buffer<ffi::F64>>() ERI_SLICE_ATTR);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERISliceCoefficientVJP,ERISliceCoefficientFirst<true>,
 ERI_COEFFICIENT_BIND.Ret<ffi::Buffer<ffi::F64>>() ERI_SLICE_ATTR);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERISliceCoefficientHessianJVP,ERISliceCoefficientSecond<false>,
 ERI_COEFFICIENT_BIND.Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>() ERI_SLICE_ATTR);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERISliceCoefficientHessianVJP,ERISliceCoefficientSecond<true>,
 ERI_COEFFICIENT_BIND.Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>() ERI_SLICE_ATTR);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERISliceExponentJVP,ERISliceExponent<false>,
 ERI_BASIS_BIND.Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>() ERI_SLICE_ATTR);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERISliceExponentVJP,ERISliceExponent<true>,
 ERI_BASIS_BIND.Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>() ERI_SLICE_ATTR);
