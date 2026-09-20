// Compressed Coulomb integrals and exact direct J/K. No N^4 allocation.
#include "tables.h"
#include <array>
#include <atomic>
#include <memory>
extern "C" {
extern CINTIntegralFunction int2c2e_cart, int2c2e_sph, int3c2e_cart, int3c2e_sph;
extern CINTOptimizerFunction int2c2e_optimizer, int3c2e_optimizer;
}

static size_t pair_index(size_t a, size_t b) {
  if (a < b) std::swap(a,b);
  return a*(a+1)/2+b;
}

struct LocalOptimizer {
  CINTOpt* value=nullptr;
  LocalOptimizer(IntegralTables& t, int centers) {
    auto f=centers==4 ? int2e_optimizer : (centers==3 ? int3c2e_optimizer : int2c2e_optimizer);
    f(&value,t.atm.data(),t.natm,t.bas.data(),t.nbas,t.env.data());
  }
  ~LocalOptimizer() { CINTdel_optimizer(&value); }
};

static void block(IntegralTables& t, Intor f, CINTOpt* opt, int* shells,
                  size_t count, std::vector<double>& data, std::vector<double>& cache) {
  if (count > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Integral shell block exceeds int32 indexing");
  const int length=f(nullptr,nullptr,shells,t.atm.data(),t.natm,t.bas.data(),t.nbas,t.env.data(),opt,nullptr);
  if (length<=0) throw std::invalid_argument("libcint cache query failed");
  data.assign(count,0.); cache.resize(length);
  f(data.data(),nullptr,shells,t.atm.data(),t.natm,t.bas.data(),t.nbas,t.env.data(),opt,cache.data());
}

template<class Consume>
static void unique_eri(IntegralTables& t, int cart, double cutoff, Consume consume) {
  if (!std::isfinite(cutoff) || cutoff<0) throw std::invalid_argument("Invalid Schwarz cutoff");
  Intor f=cart ? int2e_cart : int2e_sph;
  LocalOptimizer opt(t,4);
  std::vector<double> data,cache,bounds(size_t(t.nbas)*t.nbas,0.);
  if (cutoff>0) {
    for (int i=0;i<t.nbas;++i) for (int j=0;j<=i;++j) {
      int sh[]={i,j,i,j};int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j];
      block(t,f,opt.value,sh,size_t(di)*dj*di*dj,data,cache);
      double largest=0.;
      for (int p=0;p<di;++p) for (int q=0;q<dj;++q)
        largest=std::max(largest,std::abs(data[((size_t(q)*di+p)*dj+q)*di+p]));
      bounds[size_t(i)*t.nbas+j]=std::sqrt(largest);
    }
  }
  for (int i=0;i<t.nbas;++i) for (int j=0;j<=i;++j)
    for (int k=0;k<=i;++k) for (int l=0;l<=k;++l) {
      if (cutoff>0 && bounds[size_t(i)*t.nbas+j]*bounds[size_t(k)*t.nbas+l]<cutoff) continue;
      const int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j];
      const int dk=t.ao[k+1]-t.ao[k],dl=t.ao[l+1]-t.ao[l];
      int sh[]={i,j,k,l};
      block(t,f,opt.value,sh,size_t(di)*dj*dk*dl,data,cache);
      for (int p=0;p<di;++p) for (int q=0;q<dj;++q) {
        const int a=t.ao[i]+p,b=t.ao[j]+q;
        if (a<b) continue;
        const size_t ab=pair_index(a,b);
        for (int r=0;r<dk;++r) for (int s=0;s<dl;++s) {
          const int c=t.ao[k]+r,d=t.ao[l]+s;
          if (c<d || ab<pair_index(c,d)) continue;
          consume(a,b,c,d,data[((size_t(s)*dk+r)*dj+q)*di+p]);
        }
      }
    }
}

static ffi::Error Compact(ffi::BufferR2<ffi::S32> atoms, ffi::BufferR2<ffi::S32> basis,
    ffi::BufferR1<ffi::F64> env, ffi::ResultBuffer<ffi::F64> result,
    int32_t layout, int32_t cart, int32_t split) {
  try {
    IntegralTables t(atoms,basis,env,4,cart);
    double* out=result->typed_data();std::fill(out,out+result->element_count(),0.);
    if (layout==0 || layout==1) {
      size_t np=size_t(t.n)*(t.n+1)/2;
      size_t size=layout==0 ? np*np : np*(np+1)/2;
      if (result->element_count()!=size) throw std::invalid_argument("Packed ERI shape mismatch");
      unique_eri(t,cart,0.,[&](int a,int b,int c,int d,double v) {
        size_t ab=pair_index(a,b),cd=pair_index(c,d);
        if (layout==0) {out[ab*np+cd]=v;out[cd*np+ab]=v;}
        else out[pair_index(ab,cd)]=v;
      });
    } else if (layout==2 || layout==3) {
      if (split<1 || split>=t.nbas) throw std::invalid_argument("Invalid orbital/auxiliary shell split");
      const int n=t.ao[split],m=t.n-n;
      const size_t np=size_t(n)*(n+1)/2;
      if (result->element_count()!=(layout==2 ? size_t(m)*m : size_t(m)*np))
        throw std::invalid_argument("Auxiliary integral shape mismatch");
      Intor f=layout==2 ? (cart ? int2c2e_cart:int2c2e_sph) : (cart ? int3c2e_cart:int3c2e_sph);
      LocalOptimizer opt(t,layout==2 ? 2:3);
      std::vector<double> data,cache;
      if (layout==2) {
        for (int i=split;i<t.nbas;++i) for (int j=split;j<=i;++j) {
          int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j],sh[]={i,j};
          block(t,f,opt.value,sh,size_t(di)*dj,data,cache);
          for (int p=0;p<di;++p) for (int q=0;q<dj;++q) {
            int a=t.ao[i]+p-n,b=t.ao[j]+q-n;
            out[size_t(a)*m+b]=out[size_t(b)*m+a]=data[size_t(q)*di+p];
          }
        }
      } else {
        for (int i=0;i<split;++i) for (int j=0;j<=i;++j) for (int k=split;k<t.nbas;++k) {
          int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j],dk=t.ao[k+1]-t.ao[k],sh[]={i,j,k};
          block(t,f,opt.value,sh,size_t(di)*dj*dk,data,cache);
          for (int p=0;p<di;++p) for (int q=0;q<dj;++q) {
            int a=t.ao[i]+p,b=t.ao[j]+q;if (a<b) continue;
            for (int r=0;r<dk;++r)
              out[size_t(t.ao[k]+r-n)*np+pair_index(a,b)]=data[(size_t(r)*dj+q)*di+p];
          }
        }
      }
    } else throw std::invalid_argument("Unknown compact integral layout");
    return ffi::Error::Success();
  } catch (const std::exception& e) {return ffi::Error::Internal(e.what());}
}

static ffi::Error DirectJK(ffi::BufferR2<ffi::S32> atoms,ffi::BufferR2<ffi::S32> basis,
    ffi::BufferR1<ffi::F64> env,ffi::BufferR3<ffi::F64> density,
    ffi::ResultBufferR4<ffi::F64> result,int32_t cart,double cutoff) {
  try {
    IntegralTables t(atoms,basis,env,4,cart);const int n=t.n,nb=density.dimensions()[0];
    if (density.dimensions()[1]!=n || density.dimensions()[2]!=n ||
        result->dimensions()[0]!=2 || result->dimensions()[1]!=nb || result->dimensions()[2]!=n || result->dimensions()[3]!=n)
      throw std::invalid_argument("Direct J/K shape mismatch");
    const double* d=density.typed_data();double* out=result->typed_data();size_t nn=size_t(n)*n;
    for (size_t i=0;i<density.element_count();++i) if (!std::isfinite(d[i])) throw std::invalid_argument("Nonfinite density");
    std::fill(out,out+result->element_count(),0.);
    unique_eri(t,cart,cutoff,[&](int a,int b,int c,int e,double v) {
      const std::array<std::array<int,4>,8> permutations={{{a,b,c,e},{b,a,c,e},{a,b,e,c},{b,a,e,c},
                                                         {c,e,a,b},{e,c,a,b},{c,e,b,a},{e,c,b,a}}};
      for (size_t x=0;x<permutations.size();++x) {
        bool duplicate=false;for(size_t y=0;y<x;++y) if(permutations[x]==permutations[y]) duplicate=true;
        if (duplicate) continue;
        auto u=permutations[x];
        for (int batch=0;batch<nb;++batch) {
          const double* dm=d+size_t(batch)*nn;
          out[size_t(batch)*nn+size_t(u[0])*n+u[1]]+=v*dm[size_t(u[2])*n+u[3]];
          out[size_t(nb+batch)*nn+size_t(u[0])*n+u[2]]+=v*dm[size_t(u[1])*n+u[3]];
        }
      }
    });
    return ffi::Error::Success();
  } catch(const std::exception& e) {return ffi::Error::Internal(e.what());}
}

static ffi::Future PackedJK(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> eri,
    ffi::BufferR3<ffi::F64> density, ffi::ResultBufferR4<ffi::F64> result) {
  ffi::Promise promise;
  ffi::Future future(promise);
  try {
    const int nb=density.dimensions()[0],n=density.dimensions()[1];
    const size_t nn=size_t(n)*n,np=size_t(n)*(n+1)/2;
    bool s8=eri.dimensions().size()==1;
    if (n<1 || density.dimensions()[2]!=n ||
        (s8 ? eri.element_count()!=np*(np+1)/2 :
          (eri.dimensions().size()!=2 || eri.dimensions()[0]!=np || eri.dimensions()[1]!=np)) ||
        result->dimensions()[0]!=2 || result->dimensions()[1]!=nb || result->dimensions()[2]!=n || result->dimensions()[3]!=n)
      throw std::invalid_argument("Packed J/K dimensions do not match");
    const double* integrals=eri.typed_data();const double* input=density.typed_data();
    double* output=result->typed_data();const size_t size=result->element_count();
    // Bounded worker-local accumulators avoid atomics in the integral loop.
    // A Future prevents blocking an XLA worker while waiting on its own pool.
    const int workers=np<1000 ? 1 : int(std::max<int64_t>(1,std::min<int64_t>(8,pool.num_threads())));
    auto partials=std::make_shared<std::vector<double>>(size*workers,0.);
    auto remaining=std::make_shared<std::atomic<int>>(workers);
    for (int worker=0;worker<workers;++worker) {
      // Quartet count grows quadratically in the outer pair index.
      size_t begin=size_t(np*std::sqrt(double(worker)/workers));
      size_t end=worker+1==workers ? np : size_t(np*std::sqrt(double(worker+1)/workers));
      auto task=[=]() mutable {
        double* local=partials->data()+size*worker;
        for (int a=0;a<n;++a) for (int b=0;b<=a;++b) {
          size_t ab=pair_index(a,b),offset=s8 ? ab*(ab+1)/2 : ab*np;
          if(ab<begin || ab>=end) continue;
          for (int c=0;c<=a;++c) for (int d=0;d<=(c==a ? b:c);++d) {
            size_t cd=pair_index(c,d);double v=integrals[offset+cd];
            for (int batch=0;batch<nb;++batch) {
              const double* dm=input+size_t(batch)*nn;
              double* j=local+size_t(batch)*nn;double* k=local+size_t(nb+batch)*nn;
              double j_ab=v*(dm[size_t(c)*n+d]+(c!=d ? dm[size_t(d)*n+c]:0.));
              j[size_t(a)*n+b]+=j_ab;if(a!=b)j[size_t(b)*n+a]+=j_ab;
              if(ab!=cd) {
                double j_cd=v*(dm[size_t(a)*n+b]+(a!=b ? dm[size_t(b)*n+a]:0.));
                j[size_t(c)*n+d]+=j_cd;if(c!=d)j[size_t(d)*n+c]+=j_cd;
              }
              k[size_t(a)*n+c]+=v*dm[size_t(b)*n+d];
              if(c!=d)k[size_t(a)*n+d]+=v*dm[size_t(b)*n+c];
              if(a!=b)k[size_t(b)*n+c]+=v*dm[size_t(a)*n+d];
              if(a!=b && c!=d)k[size_t(b)*n+d]+=v*dm[size_t(a)*n+c];
              if(ab!=cd) {
                k[size_t(c)*n+a]+=v*dm[size_t(d)*n+b];
                if(a!=b)k[size_t(c)*n+b]+=v*dm[size_t(d)*n+a];
                if(c!=d)k[size_t(d)*n+a]+=v*dm[size_t(c)*n+b];
                if(a!=b && c!=d)k[size_t(d)*n+b]+=v*dm[size_t(c)*n+a];
              }
            }
          }
        }
        if(remaining->fetch_sub(1,std::memory_order_acq_rel)==1) {
          // Fixed reduction order keeps a fixed worker count reproducible.
          std::copy(partials->begin(),partials->begin()+size,output);
          for(int w=1;w<workers;++w) for(size_t i=0;i<size;++i)
            output[i]+=(*partials)[size*w+i];
          promise.SetAvailable();
        }
      };
      if(workers==1) task(); else pool.Schedule(std::move(task));
    }
  } catch(const std::exception& e) {promise.SetError(ffi::Error::Internal(e.what()));}
  return future;
}

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFCompact(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFCompact,Compact,
    ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>()
      .Ret<ffi::Buffer<ffi::F64>>().Attr<int32_t>("layout").Attr<int32_t>("cart").Attr<int32_t>("split"));
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFDirectJK(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFDirectJK,DirectJK,
    ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>()
      .Arg<ffi::BufferR3<ffi::F64>>().Ret<ffi::BufferR4<ffi::F64>>().Attr<int32_t>("cart").Attr<double>("cutoff"));
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFPackedJK(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFPackedJK,PackedJK,
    ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().Arg<ffi::Buffer<ffi::F64>>()
      .Arg<ffi::BufferR3<ffi::F64>>().Ret<ffi::BufferR4<ffi::F64>>());
