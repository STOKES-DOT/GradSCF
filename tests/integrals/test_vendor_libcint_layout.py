"""One vendor namespace, preserving the exact extension source bytes."""
import hashlib
import json
from pathlib import Path


NATIVE = Path(__file__).parents[2]/'src/gradscf/integrals/_native'
EXTENSIONS = (
    'extensions/eri/fill_int2e.c',
    'extensions/ecp/gto/nr_ecp.c',
    'extensions/ecp/gto/nr_ecp.h',
    'extensions/ecp/np_helper/np_helper.c',
    'extensions/ecp/np_helper/np_helper.h',
    'extensions/ecp/vhf/fblas.h',
)


def test_native_vendor_has_one_libcint_namespace():
    directories = {p.name for p in (NATIVE/'vendor').iterdir() if p.is_dir()}
    assert directories == {'libcint'}


def test_extension_sources_keep_original_content_and_headers():
    checksum = hashlib.sha256()
    for relative in EXTENSIONS:
        checksum.update((NATIVE/'vendor/libcint'/relative).read_bytes())
    assert checksum.hexdigest() == '8dd4305f0dcf0531aa97d0898888486e6c4bbf944a545dde6566c7929c472f0f'


def test_extension_build_and_manifest_use_libcint_paths():
    from gradscf.integrals._native.build import verify_vendor

    cmake = (NATIVE/'CMakeLists.txt').read_text()
    assert 'vendor/pyscf' not in cmake
    for relative in EXTENSIONS:
        if relative.endswith('.c'):
            assert 'vendor/libcint/'+relative in cmake
    manifest = json.loads((NATIVE/'vendor/manifest.json').read_text())
    assert all(d['directory'] == 'libcint' for d in manifest['dependencies'])
    assert {d['name'] for d in manifest['dependencies']} == {
        'libcint', 'libcint-eri-extension', 'libcint-ecp-extension'}
    verify_vendor(NATIVE)
