// Shared second-coordinate contraction. Higher products differentiate only
// the shell evaluator; index permutations and nuclear-center terms stay shared.
#pragma once
#include "eri_geometry_entries.h"

template<class Evaluate, class Accumulate>
void for_each_geometry_hessian_entry(IntegralTables& t, int op, int cart,
                                     Evaluate&& evaluate, Accumulate&& add,
                                     const ERIWindow* window=nullptr,
                                     int first_shell=0,int last_shell=-1) {
  // Shadow shells may have been appended, but never enter the physical loop.
  const int nshell=int(t.ao.size())-1;
  // Each call contributes one ordered Hessian entry. Cross-slot entries
  // explicitly contribute both coordinate orders, even for a shared atom.
  auto cross=[&](size_t index,int a,int b,double value) {
    add(index,a,b,value); add(index,b,a,value);
  };
  const int n=t.n;
  auto index2=[&](int c,int i,int j) { return (size_t(c)*n+i)*n+j; };
  std::vector<double> block,cache;
  if (op==4) {
    Intor kernels_cart[]={int2e_ipip1_cart,int2e_ipvip1_cart,int2e_ip1ip2_cart};
    Intor kernels_sph[]={int2e_ipip1_sph,int2e_ipvip1_sph,int2e_ip1ip2_sph};
    constexpr std::array<std::array<int,4>,2> same_pair{{{{0,1,2,3}},{{2,3,0,1}}}};
    constexpr std::array<std::array<int,4>,4> cross_pair{{
      {{0,1,2,3}},{{0,1,3,2}},{{1,0,2,3}},{{1,0,3,2}}
    }};
    for_each_eri_geometry_orbit(t,window,[&](const ERIGeometryOrbit& orbit) {
      auto modes=[&](int kind,const auto& permutations) {
        eri_geometry_modes(orbit,permutations,[&](int* shells) {
          evaluate(cart ? kernels_cart[kind] : kernels_sph[kind],shells,
                    9*orbit.size,block,cache,-1);
        },[&](const auto& mode) {
          for (int s=0;s<orbit.dims[3];++s) for (int r=0;r<orbit.dims[2];++r)
            for (int q=0;q<orbit.dims[1];++q) for (int p=0;p<orbit.dims[0];++p) {
              const std::array<int,4> ao{p,q,r,s};
              const auto indices=orbit.indices(ao);
              const size_t local=orbit.kernel_index(ao,mode);
              const int first=t.coord_ptr(orbit.shells[mode[0]]);
              const int second=t.coord_ptr(orbit.shells[mode[kind]]);
              for (int x=0;x<3;++x) for (int y=0;y<3;++y) {
                const double value=block[(x*3+y)*orbit.size+local];
                for (size_t index:indices) if (index!=ERIWindow::outside) {
                  if (kind==0) add(index,first+x,second+y,value);
                  else cross(index,first+x,second+y,value);
                }
              }
            }
        });
      };
      modes(0,eri_geometry_legs);modes(1,same_pair);modes(2,cross_pair);
    },first_shell,last_shell);
  } else {
    Intor same_cart[]={int1e_ipipovlp_cart,int1e_ipipkin_cart,int1e_ipiprinv_cart,int1e_ipipr_cart};
    Intor same_sph[]={int1e_ipipovlp_sph,int1e_ipipkin_sph,int1e_ipiprinv_sph,int1e_ipipr_sph};
    Intor mixed_cart[]={int1e_ipovlpip_cart,int1e_ipkinip_cart,int1e_iprinvip_cart,int1e_iprip_cart};
    Intor mixed_sph[]={int1e_ipovlpip_sph,int1e_ipkinip_sph,int1e_iprinvip_sph,int1e_iprip_sph};
    const int components=op==3 ? 3 : 1;
    for (int i=0; i<nshell; ++i) for (int j=0; j<nshell; ++j) {
      int shells[]={i,j};
      int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j];
      size_t size=size_t(di)*dj;
      for (int atom=0; atom<(op==2 ? t.natm : 1); ++atom) {
        double factor=1.;
        int nucleus=op==5 ? PTR_RINV_ORIG:-1;
        if (op==2) {
          factor=-t.atm[atom*ATM_SLOTS+CHARGE_OF];
          if (factor==0.) continue;
          nucleus=t.atm[atom*ATM_SLOTS+PTR_COORD];
          for (int x=0; x<3; ++x) t.env[PTR_RINV_ORIG+x]=t.env[nucleus+x];
        }
        // A nuclear potential or explicit Rinv depends on AO-center minus
        // potential-origin vectors. Expanding both relative derivatives
        // includes all operator/basis and operator/operator terms.
        auto add_relative=[&](size_t index,int a,int b,int x,int y,double value,bool both) {
          auto entry=[&](int first,int second,double v) {
            if (both) cross(index,first,second,v); else add(index,first,second,v);
          };
          entry(a+x,b+y,value);
          if (nucleus>=0) {
            entry(a+x,nucleus+y,-value);
            entry(nucleus+x,b+y,-value);
            entry(nucleus+x,nucleus+y,value);
          }
        };
        for (int kind=0; kind<2; ++kind) {
          const int kernel_op=op==5 ? 2:op;
          Intor kernel=kind==0 ? (cart ? same_cart[kernel_op] : same_sph[kernel_op])
                               : (cart ? mixed_cart[kernel_op] : mixed_sph[kernel_op]);
          evaluate(kernel,shells,9*components*size,block,cache,nucleus);
          for (int c=0; c<components; ++c) for (int x=0; x<3; ++x) for (int y=0; y<3; ++y)
            for (int q=0; q<dj; ++q) for (int p=0; p<di; ++p) {
              // libcint ipipr: (bra,bra,r); iprip: (bra,r,ket).
              int component=op==3 ? (kind==0 ? (x*3+y)*3+c : (x*3+c)*3+y) : x*3+y;
              double value=factor*block[component*size+q*di+p];
              int a=t.ao[i]+p,b=t.ao[j]+q;
              if (kind==0) {
                add_relative(index2(c,a,b),t.coord_ptr(i),t.coord_ptr(i),x,y,value,false);
                add_relative(index2(c,b,a),t.coord_ptr(i),t.coord_ptr(i),x,y,value,false);
              } else {
                add_relative(index2(c,a,b),t.coord_ptr(i),t.coord_ptr(j),x,y,value,true);
              }
            }
        }
      }
      if (op==3) {
        // d^2 <i|(r-O)_c|j> / dO_c dR = -dS/dR = +<nabla i|j>.
        evaluate(cart ? int1e_ipovlp_cart : int1e_ipovlp_sph,shells,3*size,block,cache,-1);
        for (int c=0; c<3; ++c) for (int x=0; x<3; ++x)
          for (int q=0; q<dj; ++q) for (int p=0; p<di; ++p) {
            double value=block[x*size+q*di+p];
            int a=t.ao[i]+p,b=t.ao[j]+q;
            cross(index2(c,a,b),t.coord_ptr(i)+x,PTR_COMMON_ORIG+c,value);
            cross(index2(c,b,a),t.coord_ptr(i)+x,PTR_COMMON_ORIG+c,value);
          }
      }
    }
  }
}

template<bool transpose, class Evaluate>
void contract_geometry_hessian(IntegralTables& t, int op, int cart,
                                const double* u, const double* input,
                                double* output, Evaluate&& evaluate,
                                const ERIWindow* window=nullptr,
                                int first_shell=0,int last_shell=-1) {
  auto add=[&](size_t index,int a,int b,double value) {
    if (index==ERIWindow::outside) return;
    if constexpr (transpose) output[a]+=value*u[b]*input[index];
    else output[index]+=value*u[a]*input[b];
  };
  for_each_geometry_hessian_entry(t,op,cart,evaluate,add,window,first_shell,last_shell);
}
