// First-coordinate ERI entries shared by full-integral and direct contractions.
#pragma once
#include "eri_symmetry.h"

// A completed coordinate derivative transforms like the value tensor. Kernel
// derivative positions do not: each mode below retains its differentiated
// physical leg, even when repeated shells allow one native block to be reused.
struct ERIGeometryOrbit {
  const ERIWindow& window;
  std::array<int,4> shells,dims,starts;
  std::vector<std::array<int,4>> emissions;
  size_t size=1;

  ERIGeometryOrbit(const IntegralTables& t,const ERIWindow& selection,
                   const std::array<int,4>& quartet):window(selection),shells(quartet) {
    for (int leg=0;leg<4;++leg) {
      starts[leg]=t.ao[shells[leg]];
      dims[leg]=t.ao[shells[leg]+1]-starts[leg];size*=dims[leg];
    }
    eri_shell_orbit(window,shells,[&](const auto& permutation,const auto&) {
      emissions.push_back({permutation[0],permutation[1],permutation[2],permutation[3]});
    });
    libcint_diagnostics::note(libcint_diagnostics::eri4c,libcint_diagnostics::emissions,
                              emissions.size());
  }

  // libcint uses first-leg-fastest storage. The mode maps its kernel legs
  // back to canonical physical legs, including general-contraction AO axes.
  size_t kernel_index(const std::array<int,4>& ao,const std::array<int,4>& mode) const {
    size_t result=0;
    for (int leg=3;leg>=0;--leg) result=result*dims[mode[leg]]+ao[mode[leg]];
    return result;
  }

  std::array<size_t,8> indices(const std::array<int,4>& ao) const {
    std::array<size_t,8> result;
    result.fill(ERIWindow::outside);
    for (size_t e=0;e<emissions.size();++e) {
      const auto& p=emissions[e];
      result[e]=window.index(starts[p[0]]+ao[p[0]],starts[p[1]]+ao[p[1]],
                             starts[p[2]]+ao[p[2]],starts[p[3]]+ao[p[3]]);
    }
    return result;
  }
};

// Group only exactly equal ordered source shell tuples for one fixed kernel.
// Each derivative occurrence is consumed separately. At most one native shell
// block lives at a time; no per-quartet tensor cache or N^4 connection list.
template<size_t N,class Evaluate,class Consume>
void eri_geometry_modes(const ERIGeometryOrbit& orbit,
                         const std::array<std::array<int,4>,N>& modes,
                         Evaluate evaluate,Consume consume) {
  std::array<std::array<int,4>,N> source;
  for (size_t m=0;m<N;++m) for (int leg=0;leg<4;++leg)
    source[m][leg]=orbit.shells[modes[m][leg]];
  for (size_t m=0;m<N;++m) {
    if (std::find(source.begin(),source.begin()+m,source[m])!=source.begin()+m) continue;
    evaluate(source[m].data());
    for (size_t occurrence=m;occurrence<N;++occurrence)
      if (source[occurrence]==source[m]) consume(modes[occurrence]);
  }
}

constexpr std::array<std::array<int,4>,4> eri_geometry_legs{{
  {{0,1,2,3}},{{1,0,2,3}},{{2,3,0,1}},{{3,2,0,1}}
}};

template<class Consume>
void for_each_eri_geometry_orbit(IntegralTables& t,const ERIWindow* window,
                                Consume consume,int first_shell,int last_shell) {
  const int physical_shells=int(t.ao.size())-1;
  if (last_shell<0) last_shell=physical_shells;
  if (first_shell<0 || first_shell>last_shell || last_shell>physical_shells)
    throw std::invalid_argument("Invalid ERI first-shell range");
  ERIWindow full(t);
  const ERIWindow& selected=window ? *window:full;
  ERICanonicalQuartets(selected).for_each([&](const auto& shells) {
    libcint_diagnostics::note(libcint_diagnostics::eri4c,libcint_diagnostics::scheduled_quartets);
    consume(ERIGeometryOrbit(t,selected,shells));
  },0,1,first_shell,last_shell);
}

template<class Accumulate>
void for_each_eri_geometry_entry(IntegralTables& t, int cart, Accumulate&& add,
                                 const ERIWindow* window=nullptr,
                                 int first_shell=0,int last_shell=-1) {
  Intor ip=cart ? int2e_ip1_cart : int2e_ip1_sph;
  std::vector<double> block,cache;
  for_each_eri_geometry_orbit(t,window,[&](const ERIGeometryOrbit& orbit) {
    eri_geometry_modes(orbit,eri_geometry_legs,[&](int* shells) {
      t.shell(ip,shells,3*orbit.size,block,cache);
    },[&](const auto& mode) {
      for (int s=0;s<orbit.dims[3];++s) for (int r=0;r<orbit.dims[2];++r)
        for (int q=0;q<orbit.dims[1];++q) for (int p=0;p<orbit.dims[0];++p) {
          const std::array<int,4> ao{p,q,r,s};
          const auto indices=orbit.indices(ao);
          const size_t local=orbit.kernel_index(ao,mode);
          for (int x=0;x<3;++x) {
            const double value=-block[x*orbit.size+local];
            const int coordinate=t.coord_ptr(orbit.shells[mode[0]])+x;
            for (size_t index:indices) if (index!=ERIWindow::outside) add(index,coordinate,value);
          }
        }
    });
  },first_shell,last_shell);
}
