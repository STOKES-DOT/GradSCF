#pragma once
#include "eri_symmetry.h"

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
  ERICanonicalQuartets symmetry;
  int active,first_shell=0,shell_stride=1;
  std::vector<double> block,cache,contracted;

  ERIBasisTables(ffi::BufferR2<ffi::S32> atoms,ffi::BufferR2<ffi::S32> basis,
                 ffi::BufferR1<ffi::F64> env,int cart,
                 const int32_t* slice=nullptr,size_t count=0)
      :t(atoms,basis,env,4,cart,true),window(t,slice,count),symmetry(window),active(t.nbas) {}

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
    symmetry.for_each([&](const auto& shells) {
      libcint_diagnostics::note(libcint_diagnostics::eri4c,libcint_diagnostics::scheduled_quartets);
      consume(shells);
    },first_shell,shell_stride);
  }
  std::array<int,4> contracted_dims(const std::array<int,4>& shells) const {
    std::array<int,4> dims;int64_t cartesian=1;
    for (int a=0;a<4;++a) {
      dims[a]=t.ao[shells[a]+1]-t.ao[shells[a]];
      const int* row=t.bas.data()+shells[a]*BAS_SLOTS;
      cartesian*=int64_t(eri_ncart(row[ANG_OF]))*row[NCTR_OF];
    }
    // Validate before allocating the extra contracted shell buffer as well.
    if (cartesian>=std::numeric_limits<int>::max()/3)
      throw std::invalid_argument("ERI basis internal Cartesian cache exceeds int32 indexing");
    return dims;
  }
  static size_t local_index(const std::array<int,4>& dims,const std::array<int,4>& local) {
    return ((size_t(local[3])*dims[2]+local[2])*dims[1]+local[1])*dims[0]+local[0];
  }
  void reset_contracted(const std::array<int,4>& dims) {
    size_t size=1;for (int d:dims) size*=d;
    contracted.assign(size,0.);
  }
  template<bool transpose>
  void orbit(const std::array<int,4>& shells,const std::array<int,4>& dims,
             const double* input,double* output) {
    eri_shell_orbit(window,shells,[&](const auto& permutation,const auto& target) {
      libcint_diagnostics::note(libcint_diagnostics::eri4c,libcint_diagnostics::emissions);
      eri_local_entries(dims,[&](const auto& local,size_t index) {
        std::array<int,4> ao;
        for (int a=0;a<4;++a) ao[a]=t.ao[target[a]]+local[permutation[a]];
        const size_t destination=window.index(ao[0],ao[1],ao[2],ao[3]);
        if (destination==ERIWindow::outside) return;
        if constexpr (transpose) contracted[index]+=input[destination];
        else output[destination]+=contracted[index];
      });
    });
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
