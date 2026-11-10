# Bounded eight-real-payload private JK amendment (2026-11-05)

The 96-atom / 880-AO shared-input complex4 spin payload becomes eight real
matrices in the FFI. The clean parent C7 kernel baseline had warm samples
80.3986, 84.5098 and 77.9871 s (median 80.3986 s), compile 2.019 s. This is a
shared-input kernel measurement, not end-to-end SCF or an optimized result.

A separate diagnostic build observed serial eri4c evaluations=65 and
cache_queries=16; requested workers4 gave evaluations=160 and cache_queries=50.
These include ten Schwarz evaluations plus 55 / 150 traversal evaluations.
The registered diagnostic gate requires equal ERI evaluation counts, while
cache queries, scheduled quartets and emissions are recorded separately.
Worker-local scratch caches need not issue equal numbers of cache queries.
Remote RED returned two failures / 36 passes in 12.71 s, exposing cap4 and
the repeated nb8 traversal. Diagnostic builds do not supply clean timings.

Only the C++ private batch limit and matching Python workspace-estimate limit
change from four to eight real matrices. The private-output byte quota,
worker reduction, ABI, AD rules and the row path above eight remain unchanged.
No zero-payload shortcut, optimizer cache or dynamic batch cap is introduced.

The complete old metadata/source-guard file is retained byte-for-byte at
`raw/fixtures/test_native_direct_parallel.pre_batch8.py`, SHA256
`a4e05572948e5e5e24c343ab3af168c73747e9cbb466c66d96d94dda754470fd`.
Only its expected C++ cap literal changes 4 to 8, and its maximum admitted
private-output formula changes 4*2*4*nao**2*8 to 4*2*8*nao**2*8 for the two
existing contraction-count cases. These are three metadata/source-guard
expectation amendments; every value, AD, tolerance, quota and >8 large-batch
regression is unchanged. New nb8 RED tests are unchanged.

Implementation is awaiting the rebuilt-library GREEN and matched clean
nb8 measurement. No performance improvement is claimed before those results.
