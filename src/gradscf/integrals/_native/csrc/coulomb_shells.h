// Shared shell-local Coulomb traversal; no dense AO ERI allocation.
#pragma once
#include "tables.h"

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
