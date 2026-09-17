// Scalar semilocal Gaussian ECP integrals. Vendor kernels are link-local.
#include "tables.h"
extern "C" {
int ECPscalar_cart(double*,int*,int*,int*,int,int*,int,double*,CINTOpt*,double*);
int ECPscalar_sph(double*,int*,int*,int*,int,int*,int,double*,CINTOpt*,double*);
}
static ffi::Error ECP(ffi::BufferR2<ffi::S32> atoms,
                      ffi::BufferR2<ffi::S32> basis,
                      ffi::BufferR2<ffi::S32> potentials,
                      ffi::BufferR1<ffi::F64> environment,
                      ffi::ResultBuffer<ffi::F64> output,int32_t cart) {
  try {
    IntegralTables t(atoms,basis,environment,0,cart);
    t.check_shape(*output,0);
    if (potentials.dimensions()[1]!=8) throw std::invalid_argument("Invalid ECP table shape");
    const int necp=potentials.dimensions()[0];
    const int* rows=potentials.typed_data();
    for (int s=0;s<t.nbas;++s)
      if (t.bas[s*8+1]>3) throw std::invalid_argument("Scalar ECP supports orbital l<=3");
    for (int i=0;i<necp;++i) {
      const int* r=rows+8*i;
      auto valid=[&](int ptr){return ptr>=20 && int64_t(ptr)+r[2]<=int64_t(t.env.size());};
      if (r[0]<0 || r[0]>=t.natm || r[1]<-1 || r[1]>4 || r[2]<1 ||
          r[3]<0 || r[3]>6 || r[4]!=0 || !valid(r[5]) || !valid(r[6]))
        throw std::invalid_argument("Invalid scalar ECP channel, power, or pointer");
      for(int k=0;k<r[2];++k) if(t.env[r[5]+k]<=0.)
        throw std::invalid_argument("ECP exponents must be positive");
    }
    t.bas.insert(t.bas.end(),rows,rows+potentials.element_count());
    t.env[18]=t.nbas;t.env[19]=necp;
    Intor intor=cart?ECPscalar_cart:ECPscalar_sph;
    double* out=output->typed_data();std::fill(out,out+output->element_count(),0.);
    for(int i=0;i<t.nbas;++i)for(int j=0;j<t.nbas;++j){
      int shells[]={i,j};int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j];
      int cache_size=intor(nullptr,nullptr,shells,t.atm.data(),t.natm,t.bas.data(),t.nbas,t.env.data(),nullptr,nullptr);
      if(cache_size<=0)throw std::invalid_argument("Invalid ECP cache size");
      std::vector<double> block(size_t(di)*dj,0.),cache(cache_size);
      intor(block.data(),nullptr,shells,t.atm.data(),t.natm,t.bas.data(),t.nbas,t.env.data(),nullptr,cache.data());
      for(int p=0;p<di;++p)for(int q=0;q<dj;++q)
        out[size_t(t.ao[i]+p)*t.n+t.ao[j]+q]=block[q*di+p];
    }
    return ffi::Error::Success();
  }catch(const std::exception& e){return ffi::Error::InvalidArgument(e.what());}
}
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFECP(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFECP,ECP,
    ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR2<ffi::S32>>()
      .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>()
      .Ret<ffi::Buffer<ffi::F64>>().Attr<int32_t>("cart"));
