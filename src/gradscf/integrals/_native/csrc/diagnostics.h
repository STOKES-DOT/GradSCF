#pragma once
#include <cstdint>
#include <utility>

namespace libcint_diagnostics {
enum Family { one_electron=0, eri4c=1, ri3c=2, ri2c=3, ecp=4 };
enum Counter { evaluations=0, cache_queries=1, scheduled_quartets=2, emissions=3 };
#ifdef GRADSCF_LIBCINT_DIAGNOSTICS
void note(int family,int counter,uint64_t amount=1) noexcept;
#else
inline void note(int,int,uint64_t=1) noexcept {}
#endif

template<class Function,class... Args>
inline auto invoke(int family,Function function,double* output,Args&&... args)
    -> decltype(function(output,std::forward<Args>(args)...)) {
  note(family,output ? evaluations:cache_queries);
  return function(output,std::forward<Args>(args)...);
}
}
