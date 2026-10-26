#pragma once
#include "eri_window.h"

inline int eri_ncart(int l) { return (l+1)*(l+2)/2; }

template<class Consume>
void eri_local_entries(const std::array<int,4>& dims, Consume consume) {
  for (int s=0;s<dims[3];++s) for (int r=0;r<dims[2];++r)
    for (int q=0;q<dims[1];++q) for (int p=0;p<dims[0];++p)
      consume(std::array<int,4>{p,q,r,s},((size_t(s)*dims[2]+r)*dims[1]+q)*dims[0]+p);
}

// Scratch is bounded by one contracted (or one-leg uncontracted/raised)
// shell quartet. Neither a primitive-four-index tensor nor an AO Jacobian is
// retained. Every product uses mutable tables owned by its calling worker.
struct ERIBasisTables {
  IntegralTables t;
  ERIWindow window;
  int active,first_shell=0,shell_stride=1;
  std::vector<double> block,cache;

  ERIBasisTables(ffi::BufferR2<ffi::S32> atoms,ffi::BufferR2<ffi::S32> basis,
                 ffi::BufferR1<ffi::F64> env,int cart,
                 const int32_t* slice=nullptr,size_t count=0)
      :t(atoms,basis,env,4,cart,true),window(t,slice,count),active(t.nbas) {}

  static void finite(const double* values,size_t size) {
    for (size_t i=0;i<size;++i)
      if (!std::isfinite(values[i]))
        throw std::invalid_argument("ERI basis product inputs must be finite");
  }
  void check_integrals(const ffi::Buffer<ffi::F64>& tensor) const {window.check_shape(tensor);}
  bool selected(int shell) const {
    for (int leg=0;leg<4;++leg)
      if (window.shells[2*leg]<=shell && shell<window.shells[2*leg+1]) return true;
    return false;
  }
  void check_vector(const ffi::Buffer<ffi::F64>& vector,size_t count) const {
    if (vector.dimensions().size()!=1 || vector.element_count()!=count)
      throw std::invalid_argument("ERI basis parameter vector has an incorrect shape");
  }
  template<class Consume>
  void quartets(Consume consume) {
    for (int i=window.shells[0]+first_shell;i<window.shells[1];i+=shell_stride)
      for (int j=window.shells[2];j<window.shells[3];++j)
      for (int k=window.shells[4];k<window.shells[5];++k)
      for (int l=window.shells[6];l<window.shells[7];++l) {
        libcint_diagnostics::note(libcint_diagnostics::eri4c,libcint_diagnostics::scheduled_quartets);
        consume(std::array<int,4>{i,j,k,l});
      }
  }
  size_t integral_index(const std::array<int,4>& shells,const std::array<int,4>& local) const {
    return window.index(t.ao[shells[0]]+local[0],t.ao[shells[1]]+local[1],
                        t.ao[shells[2]]+local[2],t.ao[shells[3]]+local[3]);
  }
  void shell(Intor intor,std::array<int,4> shells,const std::array<int,4>& dims) {
    int64_t cartesian=1;size_t size=1;
    for (int leg=0;leg<4;++leg) {
      const int* row=t.bas.data()+shells[leg]*BAS_SLOTS;
      cartesian*=int64_t(eri_ncart(row[ANG_OF]))*row[NCTR_OF];
      size*=dims[leg];
    }
    // libcint contracts spherical output in Cartesian space too. A cache
    // query can wrap to a positive int, so reject this guaranteed overflow.
    if (cartesian>=std::numeric_limits<int>::max()/3)
      throw std::invalid_argument("ERI basis internal Cartesian cache exceeds int32 indexing");
    t.shell(intor,shells.data(),size,block,cache);
  }
};
