// Shell-window traversal and local AO indexing for bounded ERI products.
#pragma once
#include "tables.h"

struct ERIWindow {
  static constexpr size_t outside=std::numeric_limits<size_t>::max();
  std::array<int,8> shells;
  std::array<int,4> starts,shape;
  std::array<int64_t,4> storage;
  int compact_layout;
  size_t pairs=0;

  ERIWindow(const IntegralTables& t,const int32_t* slice=nullptr,size_t count=0,
            int layout=-1):compact_layout(layout) {
    const int nshell=int(t.ao.size())-1;
    if (layout < -1 || layout > 1 || (layout>=0 && slice))
      throw std::invalid_argument("Invalid packed ERI geometry layout");
    if (layout>=0) pairs=size_t(t.n)*(t.n+1)/2;
    if (slice && count!=8) throw std::invalid_argument("shell_slice must have eight endpoints");
    for (int a=0;a<4;++a) {
      int first=slice ? slice[2*a] : 0,last=slice ? slice[2*a+1] : nshell;
      if (first<0 || first>=last || last>nshell)
        throw std::invalid_argument("shell_slice requires 0 <= start < stop <= nshell");
      shells[2*a]=first;shells[2*a+1]=last;
      starts[a]=t.ao[first];shape[a]=t.ao[last]-starts[a];
      storage[a]=shape[a];
    }
    // The vendor s1 fill computes each pair product in signed int first.
    if (slice && (int64_t(shape[0])*shape[1]>std::numeric_limits<int>::max() ||
                  int64_t(shape[2])*shape[3]>std::numeric_limits<int>::max()))
      throw std::invalid_argument("ERI shell_slice pair dimensions exceed int32 indexing");
  }

  void check_shape(const ffi::Buffer<ffi::F64>& buffer) const {
    if (compact_layout>=0) {
      const auto dims=buffer.dimensions();
      const bool valid=compact_layout==0 ?
        dims.size()==2 && dims[0]==pairs && dims[1]==pairs :
        dims.size()==1 && dims[0]==pairs*(pairs+1)/2;
      if (!valid) throw std::invalid_argument("Invalid packed ERI geometry shape");
      return;
    }
    if (buffer.dimensions().size()!=4)
      throw std::invalid_argument("Invalid ERI shell_slice tensor rank");
    for (int a=0;a<4;++a) if (buffer.dimensions()[a]!=shape[a])
      throw std::invalid_argument("ERI shell_slice AO dimensions do not match basis table");
  }

  void pad_to(const ffi::Buffer<ffi::F64>& buffer) {
    if (buffer.dimensions().size()!=4)
      throw std::invalid_argument("Invalid ERI slab tensor rank");
    for (int a=0;a<4;++a) {
      storage[a]=buffer.dimensions()[a];
      if (storage[a]<shape[a])
        throw std::invalid_argument("ERI slab storage shape is smaller than selected AO dimensions");
    }
  }

  size_t index(int i,int j,int k,int l) const {
    if (compact_layout>=0) {
      // Select exactly the full-tensor entries gathered by tril_indices.
      // Folding every symmetric emission into a packed index would count
      // distinct full entries repeatedly, particularly diagonal AO pairs.
      if (i<j || k<l) return outside;
      const size_t ij=size_t(i)*(i+1)/2+j,kl=size_t(k)*(k+1)/2+l;
      if (compact_layout==0) return ij*pairs+kl;
      return ij>=kl ? ij*(ij+1)/2+kl:outside;
    }
    const int ao[]={i,j,k,l};
    size_t result=0;
    for (int a=0;a<4;++a) {
      int local=ao[a]-starts[a];
      if (local<0 || local>=shape[a]) return outside;
      result=result*size_t(storage[a])+local;
    }
    return result;
  }

};
