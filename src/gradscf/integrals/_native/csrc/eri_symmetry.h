// Metadata-only eightfold real-Coulomb shell orbits. No integral evaluation,
// AO triangular restriction, optimizer, or retained quartet connection list.
#pragma once
#include "eri_window.h"

struct ERICanonicalQuartets {
  struct Pair {int first,second;unsigned membership;};
  std::vector<Pair> pairs;
  std::vector<size_t> left,right;

  explicit ERICanonicalQuartets(const ERIWindow& window) {
    std::map<std::pair<int,int>,unsigned> selected;
    for (int side=0;side<2;++side) {
      const int axis=4*side;
      for (int i=window.shells[axis];i<window.shells[axis+1];++i)
        for (int j=window.shells[axis+2];j<window.shells[axis+3];++j)
          selected[{std::max(i,j),std::min(i,j)}]|=1u<<side;
    }
    for (const auto& entry:selected) {
      const size_t index=pairs.size();
      pairs.push_back({entry.first.first,entry.first.second,entry.second});
      if (entry.second&1u) left.push_back(index);
      if (entry.second&2u) right.push_back(index);
    }
  }

  int tasks() const {
    return int(std::min(pairs.size(),size_t(std::numeric_limits<int>::max())));
  }

  template<class Consume>
  void for_each(Consume consume,int first=0,int stride=1,
                int shell_begin=0,int shell_end=std::numeric_limits<int>::max()) const {
    // The outer canonical pair owns its entire orbit. A small rectangle visits
    // only A x B pair combinations, rather than scanning all global quartets.
    for (size_t p=size_t(first);p<pairs.size();p+=size_t(stride)) {
      const auto& outer=pairs[p];
      // Coordinate/direct callers retain their physical first-shell ranges.
      // Reject an unowned outer pair before enumerating any inner quartets.
      if (outer.first<shell_begin) continue;
      if (outer.first>=shell_end) break;
      auto visit=[&](size_t q) {
        const auto& inner=pairs[q];
        consume(std::array<int,4>{outer.first,outer.second,inner.first,inner.second});
      };
      if (outer.membership==3u) {
        for (size_t q=0;q<=p;++q) visit(q);
      } else {
        const auto& opposite=outer.membership==1u ? right:left;
        for (size_t q:opposite) {if (q>p) break;visit(q);}
      }
    }
  }
};

// The full local block already represents all AO entries of a repeated shell
// tuple. Keep one shell emission while retaining every derivative-leg term.
template<class Consume>
void eri_shell_orbit(const ERIWindow& window,const std::array<int,4>& shells,Consume consume) {
  constexpr int permutations[8][4]={{0,1,2,3},{1,0,2,3},{0,1,3,2},{1,0,3,2},
                                    {2,3,0,1},{3,2,0,1},{2,3,1,0},{3,2,1,0}};
  std::array<std::array<int,4>,8> emitted;
  size_t count=0;
  for (const auto& permutation:permutations) {
    std::array<int,4> target;
    bool inside=true;
    for (int a=0;a<4;++a) {
      target[a]=shells[permutation[a]];
      inside=inside && window.shells[2*a]<=target[a] && target[a]<window.shells[2*a+1];
    }
    if (!inside || std::find(emitted.begin(),emitted.begin()+count,target)!=emitted.begin()+count)
      continue;
    emitted[count++]=target;
    consume(permutation,target);
  }
}
