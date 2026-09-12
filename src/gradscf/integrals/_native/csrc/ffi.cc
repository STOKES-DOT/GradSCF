// CPU float64 JAX FFI adapter. No Python callbacks or process-global optimizer.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

#include "xla/ffi/api/ffi.h"
extern "C" {
#include "cint.h"
#include "cint_funcs.h"
using LegacyIntor = int (*)();
void GTOnr2e_fill_s1(LegacyIntor, LegacyIntor, double*, double*, int, int, int,
                    int*, int*, CINTOpt*, int*, int, int*, int, double*);
}
#undef atm
#undef bas

namespace ffi = xla::ffi;
using Intor = int (*)(double*, int*, int*, int*, int, int*, int,
                     double*, CINTOpt*, double*);
static int NoPrescreen() { return 1; }

static ffi::Error Integrals(ffi::BufferR2<ffi::S32> atoms,
                           ffi::BufferR2<ffi::S32> basis,
                           ffi::BufferR1<ffi::F64> environment,
                           ffi::ResultBuffer<ffi::F64> result,
                           int32_t op, int32_t cart) {
  try {
    if (atoms.dimensions()[1] != ATM_SLOTS ||
        basis.dimensions()[1] != BAS_SLOTS || environment.element_count() < 20)
      return ffi::Error::InvalidArgument("Invalid libcint table dimensions");
    const int natm = atoms.dimensions()[0], nbas = basis.dimensions()[0];
    // libcint's C interface is mutable; own the buffers passed into it.
    std::vector<int> atm(atoms.typed_data(), atoms.typed_data() + atoms.element_count());
    std::vector<int> bas(basis.typed_data(), basis.typed_data() + basis.element_count());
    std::vector<double> env(environment.typed_data(),
                            environment.typed_data() + environment.element_count());
    for (double value : env)
      if (!std::isfinite(value))
        return ffi::Error::InvalidArgument("Native integral env must be finite");
    // This build intentionally exposes ordinary Coulomb only.
    if (env[PTR_RANGE_OMEGA] != 0.)
      return ffi::Error::InvalidArgument("Range-separated integrals are not supported");
    auto inside = [&](int ptr, int64_t count) {
      return ptr >= PTR_ENV_START && count > 0 &&
             int64_t(ptr) + count <= int64_t(env.size());
    };
    for (int a = 0; a < natm; ++a) {
      const int* row = atm.data() + a * ATM_SLOTS;
      if (!inside(row[PTR_COORD], 3) || row[NUC_MOD_OF] != 1)
        return ffi::Error::InvalidArgument("Native backend requires valid point nuclei");
    }
    std::vector<int> ao(nbas + 1, 0);
    for (int s = 0; s < nbas; ++s) {
      const int* row = bas.data() + s * BAS_SLOTS;
      int l = row[ANG_OF], np = row[NPRIM_OF], nc = row[NCTR_OF];
      if (row[ATOM_OF] < 0 || row[ATOM_OF] >= natm || l < 0 || l > 12 ||
          np < 1 || np > 64 || nc < 1 || nc > 64 || row[KAPPA_OF] != 0 ||
          !inside(row[PTR_EXP], np) || !inside(row[PTR_COEFF], int64_t(np)*nc))
        return ffi::Error::InvalidArgument("Invalid or unsupported libcint basis table");
      for (int p = 0; p < np; ++p)
        if (env[row[PTR_EXP] + p] <= 0.)
          return ffi::Error::InvalidArgument("Gaussian exponents must be positive");
      ao[s + 1] = ao[s] + nc * (cart ? (l + 1)*(l + 2)/2 : 2*l + 1);
    }
    const int n = ao.back(), components = op == 3 ? 3 : 1;
    const int rank = op == 4 ? 4 : (op == 3 ? 3 : 2);
    if (op < 0 || op > 4 || (cart != 0 && cart != 1) || n < 1 ||
        result->dimensions().size() != rank)
      return ffi::Error::InvalidArgument("Invalid native integral output shape or operator");
    for (int i = 0; i < rank; ++i)
      if (result->dimensions()[i] != (op == 3 && i == 0 ? 3 : n))
        return ffi::Error::InvalidArgument("AO count does not match basis table");
    Intor cart_funcs[] = {int1e_ovlp_cart, int1e_kin_cart, int1e_nuc_cart,
                          int1e_r_cart, int2e_cart};
    Intor sph_funcs[] = {int1e_ovlp_sph, int1e_kin_sph, int1e_nuc_sph,
                         int1e_r_sph, int2e_sph};
    Intor intor = cart ? cart_funcs[op] : sph_funcs[op];
    double* output = result->typed_data();
    std::fill(output, output + result->element_count(), 0.);
    if (op == 4) {
      int slices[] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
      int maxdim = 0, maxcache = 0;
      for (int s = 0; s < nbas; ++s) {
        maxdim = std::max(maxdim, ao[s+1] - ao[s]);
        int shells[] = {s, s, s, s};
        const int cache_size = intor(nullptr, nullptr, shells, atm.data(), natm,
                                     bas.data(), nbas, env.data(), nullptr, nullptr);
        // libcint's 32-bit cache ABI returns zero when its internal Cartesian
        // recurrence/contraction workspace overflows, including sph kernels.
        // It is not a valid empty cache. Never enter fill with that sentinel.
        if (cache_size <= 0)
          return ffi::Error::InvalidArgument(
              "ERI libcint cache query failed or exceeded int32 indexing");
        maxcache = std::max(maxcache, cache_size);
      }
      const int64_t blocksize = int64_t(maxdim)*maxdim*maxdim*maxdim;
      if (blocksize > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument("ERI shell workspace exceeds int32 indexing");
      // Own allocation in C++ so failure is returned to JAX, not dereferenced by C.
      std::vector<double> scratch(size_t(blocksize) + maxcache);
      for (int i = 0; i < nbas; ++i)
        for (int j = 0; j < nbas; ++j)
          GTOnr2e_fill_s1(reinterpret_cast<LegacyIntor>(intor), NoPrescreen,
                         output, scratch.data(), 1, i, j, slices, ao.data(),
                         nullptr, atm.data(), natm, bas.data(), nbas, env.data());
    } else {
      for (int i = 0; i < nbas; ++i) {
        for (int j = 0; j < nbas; ++j) {
          int shells[] = {i, j};
          const int di = ao[i+1] - ao[i], dj = ao[j+1] - ao[j];
          const size_t size = size_t(di)*dj;
          std::vector<double> block(size * components, 0.);
          intor(block.data(), nullptr, shells, atm.data(), natm,
                bas.data(), nbas, env.data(), nullptr, nullptr);
          // libcint shell blocks have the first AO index contiguous.
          for (int c = 0; c < components; ++c)
            for (int p = 0; p < di; ++p)
              for (int q = 0; q < dj; ++q)
                output[(size_t(c)*n + ao[i]+p)*n + ao[j]+q] =
                    block[size_t(c)*size + q*di + p];
        }
      }
    }
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
}

extern "C" __attribute__((visibility("default")))
XLA_FFI_Error* GradSCFIntegrals(XLA_FFI_CallFrame*);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GradSCFIntegrals, Integrals,
    ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>().Attr<int32_t>("operator")
        .Attr<int32_t>("cart"));
