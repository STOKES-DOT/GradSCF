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
  bool owns=true;
  LocalOptimizer(IntegralTables& t, int centers, CINTOpt* prepared=nullptr)
      : value(prepared),owns(prepared==nullptr) {
    if (owns) {
      auto f=centers==4 ? int2e_optimizer : (centers==3 ? int3c2e_optimizer : int2c2e_optimizer);
      f(&value,t.atm.data(),t.natm,t.bas.data(),t.nbas,t.env.data());
    }
  }
  LocalOptimizer(const LocalOptimizer&)=delete;
  LocalOptimizer& operator=(const LocalOptimizer&)=delete;
  ~LocalOptimizer() { if (owns) CINTdel_optimizer(&value); }
};

using ERICacheSizes = std::map<std::array<int,17>,int>;

static void block(IntegralTables& t, Intor f, CINTOpt* opt, int* shells,
                  size_t count, std::vector<double>& data, std::vector<double>& cache,
                  ERICacheSizes* sizes=nullptr) {
  if (count > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Integral shell block exceeds int32 indexing");
  const int family=(f==int2c2e_cart || f==int2c2e_sph) ? libcint_diagnostics::ri2c:
                   (f==int3c2e_cart || f==int3c2e_sph) ? libcint_diagnostics::ri3c:libcint_diagnostics::eri4c;
  std::array<int,17> key{};
  int length=0;
  if (sizes) {
    // Only ordinary four-center callers opt in. With fixed ENV controls,
    // scratch size depends on the intor and shell shapes, not AO centers.
    key[0]=f==int2e_cart;
    for (int a=0;a<4;++a) {
      const int* row=t.bas.data()+shells[a]*BAS_SLOTS;
      key[1+4*a]=row[ANG_OF];key[2+4*a]=row[NPRIM_OF];
      key[3+4*a]=row[NCTR_OF];key[4+4*a]=row[KAPPA_OF];
    }
    auto found=sizes->find(key);
    if (found!=sizes->end()) length=found->second;
  }
  if (!length) {
    length=libcint_diagnostics::invoke(family,f,nullptr,nullptr,shells,t.atm.data(),t.natm,t.bas.data(),t.nbas,t.env.data(),opt,nullptr);
    if (length<=0) throw std::invalid_argument("libcint cache query failed");
    if (sizes) sizes->emplace(key,length);
  }
  data.assign(count,0.); cache.resize(length);
  libcint_diagnostics::invoke(family,f,data.data(),nullptr,shells,t.atm.data(),t.natm,t.bas.data(),t.nbas,t.env.data(),opt,cache.data());
}

static std::vector<double> eri_schwarz_bounds(IntegralTables& t, int cart, double cutoff,
                                              CINTOpt* prepared_optimizer=nullptr) {
  if (!std::isfinite(cutoff) || cutoff<0) throw std::invalid_argument("Invalid Schwarz cutoff");
  if (!cutoff) return {};
  Intor f=cart ? int2e_cart : int2e_sph;
  LocalOptimizer opt(t,4,prepared_optimizer);
  std::vector<double> data,cache,bounds(size_t(t.nbas)*t.nbas,0.);
  ERICacheSizes sizes;
  for (int i=0;i<t.nbas;++i) for (int j=0;j<=i;++j) {
    int sh[]={i,j,i,j};int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j];
    block(t,f,opt.value,sh,size_t(di)*dj*di*dj,data,cache,&sizes);
    double largest=0.;
    for (int p=0;p<di;++p) for (int q=0;q<dj;++q)
      largest=std::max(largest,std::abs(data[((size_t(q)*di+p)*dj+q)*di+p]));
    bounds[size_t(i)*t.nbas+j]=std::sqrt(largest);
  }
  return bounds;
}

static std::vector<double> eri_density_bounds(const IntegralTables& t,
                                             const ffi::BufferR3<ffi::F64>& reference) {
  const int n=t.n;
  const size_t nn=size_t(n)*n;
  if (reference.dimensions()[0]<1 || reference.dimensions()[1]!=n || reference.dimensions()[2]!=n)
    throw std::invalid_argument("Direct J/K screen reference shape mismatch");
  const double* s=reference.typed_data();
  for (size_t a=0;a<reference.element_count();++a)
    if (!std::isfinite(s[a]) || s[a]<0) throw std::invalid_argument("Invalid screen reference magnitude");
  std::vector<double> bounds(size_t(t.nbas)*t.nbas,0.);
  // Reference batches remain independent of real/imaginary payload splitting
  // and the concatenated batches used by density transposition.
  for (int i=0;i<t.nbas;++i) for (int j=0;j<t.nbas;++j)
    for (int batch=0;batch<reference.dimensions()[0];++batch) {
      double sum=0.;
      for (int a=t.ao[i];a<t.ao[i+1];++a)
        for (int b=t.ao[j];b<t.ao[j+1];++b) sum+=s[size_t(batch)*nn+size_t(a)*n+b];
      double& bound=bounds[size_t(i)*t.nbas+j];
      bound=std::max(bound,sum);
    }
  for (int i=0;i<t.nbas;++i) for (int j=0;j<i;++j) {
    double bound=std::max(bounds[size_t(i)*t.nbas+j],bounds[size_t(j)*t.nbas+i]);
    bounds[size_t(i)*t.nbas+j]=bounds[size_t(j)*t.nbas+i]=bound;
  }
  return bounds;
}

// One canonical shell quartet selects its entire eightfold orbit. The bound
// controls local value contributions, not accumulated Fock/derivative errors.
static bool eri_quartet_retained(int nbas,double cutoff,const std::vector<double>& bounds,
                                 const std::vector<double>* density_bounds,
                                 bool with_j,bool with_k,int i,int j,int k,int l) {
  if (cutoff>0 && bounds[size_t(i)*nbas+j]*bounds[size_t(k)*nbas+l]<cutoff) return false;
  if (cutoff>0 && density_bounds) {
    const auto& s=*density_bounds;
    double magnitude=with_j ? std::max(s[size_t(i)*nbas+j],s[size_t(k)*nbas+l]):0.;
    if (with_k) magnitude=std::max({magnitude,s[size_t(i)*nbas+k],s[size_t(i)*nbas+l],
                                  s[size_t(j)*nbas+k],s[size_t(j)*nbas+l]});
    if (8*bounds[size_t(i)*nbas+j]*bounds[size_t(k)*nbas+l]*magnitude<cutoff) return false;
  }
  return true;
}

// Call-scoped primal selection for contracted derivative drivers. Construct
// before modifying any shadow shell; no O(nshell^4) quartet list is retained.
struct ERIQuartetScreen {
  int nbas;
  double cutoff;
  bool with_j,with_k;
  std::vector<double> bounds,density;

  ERIQuartetScreen(IntegralTables& t,int cart,double threshold,bool j,bool k,
                  const ffi::BufferR3<ffi::F64>* reference=nullptr,CINTOpt* optimizer=nullptr)
      :nbas(t.nbas),cutoff(threshold),with_j(j),with_k(k),
       bounds(eri_schwarz_bounds(t,cart,threshold,optimizer)) {
    if (reference && cutoff>0) density=eri_density_bounds(t,*reference);
  }
  bool keep(const std::array<int,4>& shells) const {
    return eri_quartet_retained(nbas,cutoff,bounds,density.empty() ? nullptr:&density,
                                with_j,with_k,shells[0],shells[1],shells[2],shells[3]);
  }
};

template<class Consume>
static void unique_eri(IntegralTables& t, int cart, double cutoff, Consume consume,
                       int shell_begin=0, int shell_end=-1,
                       const std::vector<double>* prepared_bounds=nullptr,
                       CINTOpt* prepared_optimizer=nullptr,
                       int i_start=0, int i_stride=1,
                       const std::vector<double>* density_bounds=nullptr,
                       bool with_j=true, bool with_k=true) {
  if (!std::isfinite(cutoff) || cutoff<0) throw std::invalid_argument("Invalid Schwarz cutoff");
  if (shell_end<0) shell_end=t.nbas;
  if (shell_begin<0 || shell_begin>=shell_end || shell_end>t.nbas)
    throw std::invalid_argument("Invalid ERI output-shell range");
  if (i_stride<1 || i_start<0 || i_start>=i_stride)
    throw std::invalid_argument("Invalid ERI outer-shell partition");
  Intor f=cart ? int2e_cart : int2e_sph;
  LocalOptimizer opt(t,4,prepared_optimizer);
  std::vector<double> data,cache,owned_bounds;
  if (!prepared_bounds) owned_bounds=eri_schwarz_bounds(t,cart,cutoff,opt.value);
  const auto& bounds=prepared_bounds ? *prepared_bounds : owned_bounds;
  if (cutoff>0 && bounds.size()!=size_t(t.nbas)*t.nbas)
    throw std::invalid_argument("Schwarz bounds do not match shell dimensions");
  if (density_bounds && density_bounds->size()!=size_t(t.nbas)*t.nbas)
    throw std::invalid_argument("Density bounds do not match shell dimensions");
  const double largest_bound=cutoff>0 ? *std::max_element(bounds.begin(),bounds.end()) : 0.;
  const double largest_density=density_bounds ? *std::max_element(density_bounds->begin(),density_bounds->end()) : 0.;
  ERICacheSizes sizes;
  auto owns=[&](int shell) {return shell_begin<=shell && shell<shell_end;};
  for (int i=shell_begin+i_start;i<t.nbas;i+=i_stride) for (int j=0;j<=i;++j) {
    // This upper bound rejects only pairs whose every inner quartet already
    // fails the same basis-only Schwarz test used below.
    if (cutoff>0 && bounds[size_t(i)*t.nbas+j]*largest_bound<cutoff) continue;
    // A safe upper bound for every inner density-mask decision below.
    if (cutoff>0 && density_bounds && 8*bounds[size_t(i)*t.nbas+j]*largest_bound*largest_density<cutoff) continue;
    const bool outer_owned=owns(i) || owns(j);
    // Visit a quartet only if at least one of its four output shells is
    // owned. The retained i,j,k,l order is the original lexicographic order.
    for (int k=outer_owned ? 0:shell_begin;k<=i;++k) {
      const bool owned=outer_owned || owns(k);
      const int first=owned ? 0:shell_begin;
      // Canonical shell pairs avoid computing both (i,j,i,l) and (i,l,i,j).
      const int pair_end=k==i ? j+1:k+1;
      const int last=owned ? pair_end:std::min(pair_end,shell_end);
      for (int l=first;l<last;++l) {
        if (!eri_quartet_retained(t.nbas,cutoff,bounds,density_bounds,with_j,with_k,i,j,k,l)) continue;
        const int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j];
        const int dk=t.ao[k+1]-t.ao[k],dl=t.ao[l+1]-t.ao[l];
        int sh[]={i,j,k,l};
        block(t,f,opt.value,sh,size_t(di)*dj*dk*dl,data,cache,&sizes);
        const bool same_shell_pair=i==k && j==l;
        for (int p=0;p<di;++p) for (int q=0;q<dj;++q) {
          const int a=t.ao[i]+p,b=t.ao[j]+q;
          if (a<b) continue;
          const size_t ab=pair_index(a,b);
          for (int r=0;r<dk;++r) for (int s=0;s<dl;++s) {
            const int c=t.ao[k]+r,d=t.ao[l]+s;
            if (c<d) continue;
            const bool ordered=ab>=pair_index(c,d);
            if (!ordered && same_shell_pair) continue;
            const double value=data[((size_t(s)*dk+r)*dj+q)*di+p];
            // With i==k, distinct shell pairs can interleave in AO-pair
            // order. Emit their exchanged AO tuple instead of dropping it;
            // only equal shell pairs contain both copies in this block.
            if (ordered) consume(a,b,c,d,value);
            else consume(c,d,a,b,value);
          }
        }
      }
    }
  }
}
