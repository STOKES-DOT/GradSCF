// Analytic full/selected ERI coefficient and exponent products. Four independent
// shadow contractions preserve repeated-shell legs and individual zero
// coefficients. Exponents use -r^2 Gaussian angular raising, never differences.
#include "eri_basis.h"
#include "parallel.h"
#include "coulomb_shells.h"
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
      const std::array<const double*,4>& directions,int identity=-1,int geometry_leg=-1) {
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
    if (geometry_leg<0) shell(intor,shadows,dims);
    else geometry_shell(cart ? int2e_ip1_cart:int2e_ip1_sph,shadows,dims,geometry_leg);
    return dims;
  }
  template<class Keep,class Emit>
  void directional_blocks(const double* left,const double* right,Keep keep,Emit emit) {
    quartets([&](const auto& shells) {
      if (!keep(shells)) return;
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
      emit(shells,dims);
    });
  }
  void directional(const double* left,const double* right,double* output) {
    directional_blocks(left,right,[](const auto&) {return true;},
      [&](const auto& shells,const auto& dims) {orbit<false>(shells,dims,nullptr,output);});
  }
  void accumulate_free(const std::array<int,4>& shells,int free,
                       const std::array<int,4>& dims,const std::array<int,4>& free_dims,
                       double* output,const double* geometry_weights=nullptr) {
    int s=shells[free],np=primitives[s],nc=contractions[s],na=angular[s];
    size_t size=1;for (int d:free_dims) size*=d;
    eri_local_entries(free_dims,[&](const auto& local,size_t index) {
      int primitive=local[free]/na,angular_index=local[free]%na;
      double value=block[index];
      if (geometry_weights) {
        value=0.;
        for (int axis=0;axis<3;++axis) value+=geometry_weights[axis]*block[size_t(axis)*size+index];
      }
      auto target=local;
      for (int contraction=0;contraction<nc;++contraction) {
        target[free]=contraction*na+angular_index;
        // Identity axes describe primitives; pooled weights describe AOs.
        output[offsets[s]+size_t(contraction)*np+primitive]+=
          value*contracted[local_index(dims,target)];
      }
    });
  }
  template<class Keep,class Pool>
  void transpose_blocks(const double* direction,double* output,Keep keep,Pool pool) {
    quartets([&](const auto& shells) {
      if (!keep(shells)) return;
      const auto dims=contracted_dims(shells);reset_contracted(dims);
      pool(shells,dims);
      for (int free=0;free<4;++free)
        for (int leg=0;leg<(direction ? 4:1);++leg) {
          if (direction && free==leg) continue;
          std::array<const double*,4> directions{};
          if (direction) directions[leg]=direction;
          const auto free_dims=evaluate(shells,directions,free);
          accumulate_free(shells,free,dims,free_dims,output);
        }
    });
  }
  void transpose(const double* direction,const double* cotangent,double* output) {
    transpose_blocks(direction,output,[](const auto&) {return true;},
      [&](const auto& shells,const auto& dims) {orbit<true>(shells,dims,cotangent,nullptr);});
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
  std::array<int,4> evaluate(const std::array<int,4>& original,int leg,int primitive,int geometry_leg=-1) {
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
    if (geometry_leg<0) shell(int2e_cart,shells,raised);
    else geometry_shell(int2e_ip1_cart,shells,raised,geometry_leg);
    auto& indices=raise_indices[l];
    if (indices.empty()) {
      auto lower=eri_powers(l),higher=eri_powers(l+2);indices.resize(lower.size());
      for (size_t p=0;p<lower.size();++p) for (int axis=0;axis<3;++axis) {
        auto power=lower[p];power[axis]+=2;
        indices[p][axis]=int(std::find(higher.begin(),higher.end(),power)-higher.begin());
      }
    }
    size_t size=1,raised_size=1;
    for (int a=0;a<4;++a) {size*=dims[a];raised_size*=raised[a];}
    const int components=geometry_leg<0 ? 1:3;
    reduced.assign(size_t(components)*size,0.);
    eri_local_entries(dims,[&](const auto& local,size_t index) {
      int contraction=local[leg]/eri_ncart(l),angular=local[leg]%eri_ncart(l);
      for (int axis=0;axis<3;++axis) {
        auto shifted=local;shifted[leg]=contraction*eri_ncart(l+2)+indices[angular][axis];
        size_t source_index=shifted[3];
        for (int a=2;a>=0;--a) source_index=source_index*raised[a]+shifted[a];
        for (int component=0;component<components;++component)
          reduced[size_t(component)*size+index]-=block[size_t(component)*raised_size+source_index];
      }
    });
    if (!cart) spherical(dims,original);
    return dims;
  }
  template<bool transpose,class Keep,class Orbit>
  void apply_blocks(const double* input,double* output,Keep keep,Orbit orbit) {
    quartets([&](const auto& shells) {
      if (!keep(shells)) return;
      const auto dims=contracted_dims(shells);reset_contracted(dims);
      if constexpr (transpose) orbit(shells,dims);
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
      if constexpr (!transpose) orbit(shells,dims);
    });
  }
  template<bool transpose>
  void apply(const double* input,double* output) {
    apply_blocks<transpose>(input,output,[](const auto&) {return true;},
      [&](const auto& shells,const auto& dims) {
        orbit<transpose>(shells,dims,input,output);
      });
  }
};

// Canonical outer pairs own disjoint full/window tensor blocks. Forward
// workers share the output; adjoints replicate only the small parameter vector.
template<bool transpose,class Driver,class Apply>
static void parallel_eri_basis(Driver& driver,double* output,Apply apply,int requested_workers=0,
                               size_t result_count=0) {
  const int workers=requested_workers>0 ? std::min(requested_workers,driver.active)
                                       : native_threads(driver.symmetry->tasks());
  const size_t count=result_count ? result_count:driver.count;
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
        partial[worker].assign(count,0.);
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
      for (size_t p=0;p<count;++p) output[p]+=values[p];
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

// Direct products share the full/window parameter drivers, but pool/scatter
// only one contracted shell. No AO ERI or AO-parameter Jacobian is formed.
struct DirectBasisJK {
  int n,nb,with_j,with_k;
  size_t nn;
  const double* density;

  DirectBasisJK(const IntegralTables& physical,ffi::BufferR3<ffi::F64> dm,
                ffi::BufferR3<ffi::F64> reference,int j,int k,double cutoff,int workers)
      :n(physical.n),nb(dm.dimensions()[0]),with_j(j),with_k(k),nn(size_t(n)*n),
       density(dm.typed_data()) {
    if ((j!=0 && j!=1) || (k!=0 && k!=1) || !(j || k) || workers<1 ||
        !std::isfinite(cutoff) || cutoff<0.)
      throw std::invalid_argument("Invalid direct basis product attributes");
    if (dm.dimensions()[1]!=n || dm.dimensions()[2]!=n ||
        reference.dimensions()[1]!=n || reference.dimensions()[2]!=n)
      throw std::invalid_argument("Direct basis density/reference shape mismatch");
    ERIBasisTables::finite(dm.typed_data(),dm.element_count());
  }
  void check_jk(const ffi::Buffer<ffi::F64>& values) const {
    const auto dims=values.dimensions();
    if (dims.size()!=4 || dims[0]!=with_j+with_k || dims[1]!=nb || dims[2]!=n || dims[3]!=n)
      throw std::invalid_argument("Direct basis J/K product shape mismatch");
  }
  template<bool transpose,class Driver>
  void orbit(Driver& driver,const std::array<int,4>& shells,const std::array<int,4>& dims,
             const double* cotangent,double* output) const {
    eri_shell_orbit(driver.window,shells,[&](const auto& permutation,const auto& target) {
      libcint_diagnostics::note(libcint_diagnostics::eri4c,libcint_diagnostics::emissions);
      eri_local_entries(dims,[&](const auto& local,size_t index) {
        std::array<int,4> ao;
        for (int leg=0;leg<4;++leg) ao[leg]=driver.t.ao[target[leg]]+local[permutation[leg]];
        const int a=ao[0],b=ao[1],c=ao[2],d=ao[3];
        double pooled=0.;
        for (int batch=0;batch<nb;++batch) {
          const double* dm=density+size_t(batch)*nn;
          const size_t j_index=size_t(batch)*nn+size_t(a)*n+b;
          const size_t k_index=size_t(with_j*nb+batch)*nn+size_t(a)*n+c;
          if constexpr (transpose) {
            if (with_j) pooled+=cotangent[j_index]*dm[size_t(c)*n+d];
            if (with_k) pooled+=cotangent[k_index]*dm[size_t(b)*n+d];
          } else {
            if (with_j) output[j_index]+=driver.contracted[index]*dm[size_t(c)*n+d];
            if (with_k) output[k_index]+=driver.contracted[index]*dm[size_t(b)*n+d];
          }
        }
        if constexpr (transpose) driver.contracted[index]+=pooled;
      });
    });
  }
};

template<bool transpose>
static ffi::Error DirectCoefficientProduct(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> env,
    ffi::BufferR1<ffi::F64> coefficients,ffi::BufferR3<ffi::F64> density,
    ffi::BufferR3<ffi::F64> reference,ffi::Buffer<ffi::F64> vector,
    ffi::ResultBuffer<ffi::F64> result,int32_t cart,double cutoff,
    int32_t with_j,int32_t with_k,int32_t workers) {
  try {
    IntegralTables physical(atoms,basis,env,4,cart,true);
    DirectBasisJK jk(physical,density,reference,with_j,with_k,cutoff,workers);
    // Freeze screening from ORIGINAL primal shells before any shadow append.
    ERIQuartetScreen screen(physical,cart,cutoff,with_j,with_k,
                            reference.dimensions()[0] ? &reference:nullptr);
    ERICoefficientTables driver(atoms,basis,env,coefficients,cart);
    if constexpr (transpose) {jk.check_jk(vector);driver.check_vector(*result,driver.count);}
    else {driver.check_vector(vector,driver.count);jk.check_jk(*result);}
    driver.finite(vector.typed_data(),vector.element_count());
    double* output=result->typed_data();std::fill(output,output+result->element_count(),0.);
    auto keep=[&](const auto& shells) {return screen.keep(shells);};
    if constexpr (transpose) {
      parallel_eri_basis<true>(driver,output,[&](auto& local,double* out) {
        local.transpose_blocks(nullptr,out,keep,[&](const auto& shells,const auto& dims) {
          jk.orbit<true>(local,shells,dims,vector.typed_data(),nullptr);
        });
      },workers);
    } else {
      // Serial accumulation retains one output and one shell workspace. No
      // unbudgeted per-worker J/K matrices are introduced by basis JVPs.
      driver.directional_blocks(vector.typed_data(),nullptr,keep,
        [&](const auto& shells,const auto& dims) {jk.orbit<false>(driver,shells,dims,nullptr,output);});
    }
    return ffi::Error::Success();
  } catch (const std::invalid_argument& e) {return ffi::Error::InvalidArgument(e.what());}
    catch (const std::exception& e) {return ffi::Error::Internal(e.what());}
}

template<bool transpose>
static ffi::Error DirectExponentProduct(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> env,
    ffi::BufferR3<ffi::F64> density,ffi::BufferR3<ffi::F64> reference,
    ffi::Buffer<ffi::F64> vector,ffi::ResultBuffer<ffi::F64> result,
    int32_t cart,double cutoff,int32_t with_j,int32_t with_k,int32_t workers) {
  try {
    IntegralTables physical(atoms,basis,env,4,cart,true);
    DirectBasisJK jk(physical,density,reference,with_j,with_k,cutoff,workers);
    ERIQuartetScreen screen(physical,cart,cutoff,with_j,with_k,
                            reference.dimensions()[0] ? &reference:nullptr);
    ERIExponentTables driver(atoms,basis,env,cart);
    if constexpr (transpose) {jk.check_jk(vector);driver.check_vector(*result,driver.count);}
    else {driver.check_vector(vector,driver.count);jk.check_jk(*result);}
    driver.finite(vector.typed_data(),vector.element_count());
    double* output=result->typed_data();std::fill(output,output+result->element_count(),0.);
    auto keep=[&](const auto& shells) {return screen.keep(shells);};
    auto apply=[&](auto& local,double* out) {
      local.template apply_blocks<transpose>(vector.typed_data(),out,keep,
        [&](const auto& shells,const auto& dims) {
          jk.orbit<transpose>(local,shells,dims,vector.typed_data(),out);
        });
    };
    if constexpr (transpose) parallel_eri_basis<true>(driver,output,apply,workers);
    else apply(driver,output);
    return ffi::Error::Success();
  } catch (const std::invalid_argument& e) {return ffi::Error::InvalidArgument(e.what());}
    catch (const std::exception& e) {return ffi::Error::Internal(e.what());}
}

ERI_BASIS_SYMBOL(GradSCFDirectJKCoefficientJVP);
ERI_BASIS_SYMBOL(GradSCFDirectJKCoefficientVJP);
ERI_BASIS_SYMBOL(GradSCFDirectJKExponentJVP);
ERI_BASIS_SYMBOL(GradSCFDirectJKExponentVJP);
#define DIRECT_BASIS_TAIL .Arg<ffi::BufferR3<ffi::F64>>().Arg<ffi::BufferR3<ffi::F64>>() \
 .Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>() \
 .Attr<int32_t>("cart").Attr<double>("cutoff").Attr<int32_t>("with_j") \
 .Attr<int32_t>("with_k").Attr<int32_t>("workers")
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFDirectJKCoefficientJVP,DirectCoefficientProduct<false>,
 ERI_BASIS_BIND.Arg<ffi::BufferR1<ffi::F64>>() DIRECT_BASIS_TAIL);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFDirectJKCoefficientVJP,DirectCoefficientProduct<true>,
 ERI_BASIS_BIND.Arg<ffi::BufferR1<ffi::F64>>() DIRECT_BASIS_TAIL);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFDirectJKExponentJVP,DirectExponentProduct<false>,
 ERI_BASIS_BIND DIRECT_BASIS_TAIL);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFDirectJKExponentVJP,DirectExponentProduct<true>,
 ERI_BASIS_BIND DIRECT_BASIS_TAIL);

// Mixed geometry/basis products. mode 0 pushes both directions, mode 1
// returns the basis adjoint, and mode 2 returns a coordinate ENV adjoint.
static bool CheckDirectMixed(const IntegralTables& physical,const DirectBasisJK& jk,
    size_t parameter_count,ffi::Buffer<ffi::F64> left,ffi::Buffer<ffi::F64> right,
    const ffi::Buffer<ffi::F64>& result,int mode) {
  if (mode<0 || mode>2) throw std::invalid_argument("Invalid direct mixed product mode");
  auto vector=[](const auto& value,size_t size) {
    if (value.dimensions().size()!=1 || value.element_count()!=size)
      throw std::invalid_argument("Direct mixed parameter/direction shape mismatch");
  };
  vector(left,mode==2 ? parameter_count:physical.env.size());
  if (mode==0) {vector(right,parameter_count);jk.check_jk(result);}
  else {jk.check_jk(right);vector(result,mode==1 ? parameter_count:physical.env.size());}
  ERIBasisTables::finite(left.typed_data(),left.element_count());
  ERIBasisTables::finite(right.typed_data(),right.element_count());
  bool active=false;
  std::vector<bool> coordinate(physical.env.size(),false);
  if (mode!=2)
    for (int atom=0;atom<physical.natm;++atom) for (int axis=0;axis<3;++axis)
      coordinate[physical.atm[atom*ATM_SLOTS+PTR_COORD]+axis]=true;
  for (size_t i=0;i<left.element_count();++i) if (left.typed_data()[i]!=0.) {
    if (mode!=2 && !coordinate[i])
      throw std::invalid_argument("Direct mixed geometry directions require coordinate slots only");
    active=true;
  }
  if (mode==0) {
    bool basis_active=false;
    for (size_t i=0;i<right.element_count();++i) basis_active=basis_active || right.typed_data()[i]!=0.;
    active=active && basis_active;
  }
  return active;
}

static ffi::Error DirectCoefficientGeometry(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> env,
    ffi::BufferR1<ffi::F64> coefficients,ffi::BufferR3<ffi::F64> density,
    ffi::BufferR3<ffi::F64> reference,ffi::Buffer<ffi::F64> left,
    ffi::Buffer<ffi::F64> right,ffi::ResultBuffer<ffi::F64> result,
    int32_t cart,double cutoff,int32_t with_j,int32_t with_k,int32_t workers,int32_t mode) {
  try {
    IntegralTables physical(atoms,basis,env,4,cart,true);
    DirectBasisJK jk(physical,density,reference,with_j,with_k,cutoff,workers);
    ERIQuartetScreen screen(physical,cart,cutoff,with_j,with_k,
                            reference.dimensions()[0] ? &reference:nullptr);
    ERICoefficientTables driver(atoms,basis,env,coefficients,cart);
    const bool active=CheckDirectMixed(physical,jk,driver.count,left,right,*result,mode);
    double* output=result->typed_data();std::fill(output,output+result->element_count(),0.);
    if (!active) return ffi::Error::Success();
    auto apply=[&](auto& local,double* out) {
      local.quartets([&](const auto& shells) {
        if (!screen.keep(shells)) return;
        const auto dims=local.contracted_dims(shells);local.reset_contracted(dims);
        const size_t size=local.contracted.size();
        if (mode!=0) jk.orbit<true>(local,shells,dims,right.typed_data(),nullptr);
        for (int geometry=0;geometry<4;++geometry) {
          const int coordinate=local.t.coord_ptr(shells[geometry]);
          for (int leg=0;leg<4;++leg) {
            std::array<const double*,4> directions{};
            if (mode!=1) directions[leg]=mode==0 ? right.typed_data():left.typed_data();
            const auto evaluated=local.evaluate(shells,directions,mode==1 ? leg:-1,geometry);
            if (mode==1) {
              local.accumulate_free(shells,leg,dims,evaluated,out,left.typed_data()+coordinate);
            } else {
              for (int axis=0;axis<3;++axis) {
                double value=0.;
                for (size_t index=0;index<size;++index) {
                  const double derivative=local.block[size_t(axis)*size+index];
                  if (mode==0) local.contracted[index]+=left.typed_data()[coordinate+axis]*derivative;
                  else value+=local.contracted[index]*derivative;
                }
                if (mode==2) out[coordinate+axis]+=value;
              }
            }
          }
        }
        if (mode==0) jk.orbit<false>(local,shells,dims,nullptr,out);
      });
    };
    if (mode==0) apply(driver,output);
    else parallel_eri_basis<true>(driver,output,apply,workers,result->element_count());
    return ffi::Error::Success();
  } catch (const std::invalid_argument& e) {return ffi::Error::InvalidArgument(e.what());}
    catch (const std::exception& e) {return ffi::Error::Internal(e.what());}
}

static ffi::Error DirectExponentGeometry(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> env,
    ffi::BufferR3<ffi::F64> density,ffi::BufferR3<ffi::F64> reference,
    ffi::Buffer<ffi::F64> left,ffi::Buffer<ffi::F64> right,ffi::ResultBuffer<ffi::F64> result,
    int32_t cart,double cutoff,int32_t with_j,int32_t with_k,int32_t workers,int32_t mode) {
  try {
    IntegralTables physical(atoms,basis,env,4,cart,true);
    DirectBasisJK jk(physical,density,reference,with_j,with_k,cutoff,workers);
    ERIQuartetScreen screen(physical,cart,cutoff,with_j,with_k,
                            reference.dimensions()[0] ? &reference:nullptr);
    ERIExponentTables driver(atoms,basis,env,cart);
    const bool active=CheckDirectMixed(physical,jk,driver.count,left,right,*result,mode);
    double* output=result->typed_data();std::fill(output,output+result->element_count(),0.);
    if (!active) return ffi::Error::Success();
    auto apply=[&](auto& local,double* out) {
      local.quartets([&](const auto& shells) {
        if (!screen.keep(shells)) return;
        const auto dims=local.contracted_dims(shells);local.reset_contracted(dims);
        const size_t size=local.contracted.size();
        if (mode!=0) jk.orbit<true>(local,shells,dims,right.typed_data(),nullptr);
        for (int geometry=0;geometry<4;++geometry) {
          const int coordinate=local.t.coord_ptr(shells[geometry]);
          for (int leg=0;leg<4;++leg) {
            const int shell=shells[leg],np=local.t.bas[shell*BAS_SLOTS+NPRIM_OF];
            for (int primitive=0;primitive<np;++primitive) {
              const size_t parameter=local.offsets[shell]+primitive;
              const double weight=mode==1 ? 1.:
                (mode==0 ? right.typed_data()[parameter]:left.typed_data()[parameter]);
              if (weight==0.) continue;
              local.evaluate(shells,leg,primitive,geometry);
              for (int axis=0;axis<3;++axis) {
                double value=0.;
                for (size_t index=0;index<size;++index) {
                  const double derivative=local.reduced[size_t(axis)*size+index];
                  if (mode==0)
                    local.contracted[index]+=left.typed_data()[coordinate+axis]*weight*derivative;
                  else value+=local.contracted[index]*derivative;
                }
                if (mode==1) out[parameter]+=left.typed_data()[coordinate+axis]*value;
                else if (mode==2) out[coordinate+axis]+=weight*value;
              }
            }
          }
        }
        if (mode==0) jk.orbit<false>(local,shells,dims,nullptr,out);
      });
    };
    if (mode==0) apply(driver,output);
    else parallel_eri_basis<true>(driver,output,apply,workers,result->element_count());
    return ffi::Error::Success();
  } catch (const std::invalid_argument& e) {return ffi::Error::InvalidArgument(e.what());}
    catch (const std::exception& e) {return ffi::Error::Internal(e.what());}
}

ERI_BASIS_SYMBOL(GradSCFDirectJKCoefficientGeometry);
ERI_BASIS_SYMBOL(GradSCFDirectJKExponentGeometry);
#define DIRECT_MIXED_TAIL .Arg<ffi::BufferR3<ffi::F64>>().Arg<ffi::BufferR3<ffi::F64>>() \
 .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>() \
 .Attr<int32_t>("cart").Attr<double>("cutoff").Attr<int32_t>("with_j") \
 .Attr<int32_t>("with_k").Attr<int32_t>("workers").Attr<int32_t>("mode")
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFDirectJKCoefficientGeometry,DirectCoefficientGeometry,
 ERI_BASIS_BIND.Arg<ffi::BufferR1<ffi::F64>>() DIRECT_MIXED_TAIL);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFDirectJKExponentGeometry,DirectExponentGeometry,
 ERI_BASIS_BIND DIRECT_MIXED_TAIL);
