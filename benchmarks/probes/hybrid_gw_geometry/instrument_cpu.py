"""Instrument a disposable native source copy; production sources stay clean."""
import argparse
import difflib
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('snapshot', type=Path)
parser.add_argument('patch', type=Path)
args = parser.parse_args()
root = args.snapshot/'src/gradscf/integrals/_native/csrc'
changes = []
for name in ('direct_jk.cc', 'eri_geometry_entries.h'):
    path = root/name
    original = path.read_text()
    assert 'GRADSCF_CPU_' not in original
    if name == 'direct_jk.cc':
        anchor = '    auto partials=std::make_shared<std::vector<double>>('
        insert = '''#ifdef GRADSCF_LIBCINT_DIAGNOSTICS
    std::fprintf(stderr,"GRADSCF_CPU_JK requested=%d pool=%lld workers=%d private_bytes=%zu batch=%d nao=%d ops=%d\\n",
        requested_workers,(long long)pool.num_threads(),workers,
        private_outputs ? size_t(workers)*size*sizeof(double) : 0,nb,n,with_j+with_k);
#endif
'''
        updated = original.replace(anchor, insert+anchor)
        updated = '#include <cstdio>\n'+updated
    else:
        anchor = '  ERICanonicalQuartets(selected).for_each([&](const auto& shells) {'
        insert = '''#ifdef GRADSCF_LIBCINT_DIAGNOSTICS
  size_t scheduled=0;
  const auto started=std::chrono::steady_clock::now();
#endif
'''
        updated = original.replace(anchor, insert+anchor+'''
#ifdef GRADSCF_LIBCINT_DIAGNOSTICS
    ++scheduled;
#endif''')
        anchor = '}\n\ntemplate<class Accumulate>\nvoid for_each_eri_geometry_entry'
        insert = '''#ifdef GRADSCF_LIBCINT_DIAGNOSTICS
  std::fprintf(stderr,"GRADSCF_CPU_GEOMETRY first_shell=%d last_shell=%d quartets=%zu seconds=%.9f\\n",
      first_shell,last_shell,scheduled,
      std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());
#endif
'''
        updated = updated.replace(anchor, insert+anchor)
        updated = '#include <cstdio>\n#include <chrono>\n'+updated
    assert updated != original and 'GRADSCF_CPU_' in updated
    changes.extend(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
        fromfile='a/src/gradscf/integrals/_native/csrc/'+name,
        tofile='b/src/gradscf/integrals/_native/csrc/'+name))
    path.write_text(updated)
args.patch.write_text(''.join(changes))
