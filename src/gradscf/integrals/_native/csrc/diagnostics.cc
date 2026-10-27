#include "diagnostics.h"
#ifdef GRADSCF_LIBCINT_DIAGNOSTICS
#include <atomic>
namespace libcint_diagnostics {
static std::atomic<uint64_t> counts[5][4]{};
void note(int family,int counter,uint64_t amount) noexcept {
  if (family>=0 && family<5 && counter>=0 && counter<4)
    counts[family][counter].fetch_add(amount,std::memory_order_relaxed);
}
}
#endif

extern "C" __attribute__((visibility("default")))
int GradSCFLibcintDiagnosticsEnabled() {
#ifdef GRADSCF_LIBCINT_DIAGNOSTICS
  return 1;
#else
  return 0;
#endif
}
extern "C" __attribute__((visibility("default")))
void GradSCFLibcintDiagnosticsReset() {
#ifdef GRADSCF_LIBCINT_DIAGNOSTICS
  for (auto& family:libcint_diagnostics::counts)
    for (auto& counter:family) counter.store(0,std::memory_order_relaxed);
#endif
}
extern "C" __attribute__((visibility("default")))
uint64_t GradSCFLibcintDiagnosticsRead(int family,int counter) {
#ifdef GRADSCF_LIBCINT_DIAGNOSTICS
  if (family>=0 && family<5 && counter>=0 && counter<4)
    return libcint_diagnostics::counts[family][counter].load(std::memory_order_relaxed);
#endif
  return 0;
}
