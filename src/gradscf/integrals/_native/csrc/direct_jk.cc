// Shell-direct J/K and its stable FFI entry point.
#include "coulomb_shells.h"
#include <array>

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

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFDirectJK(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFDirectJK,DirectJK,
    ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>()
      .Arg<ffi::BufferR3<ffi::F64>>().Ret<ffi::BufferR4<ffi::F64>>().Attr<int32_t>("cart").Attr<double>("cutoff"));
