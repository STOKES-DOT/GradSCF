import importlib


def test_gradscf_dir_lists_recommended_namespaces():
    import gradscf

    for name in ("gto", "scf", "dft", "tdscf", "neural_xc", "training"):
        assert name in gradscf.__all__
        assert name in dir(gradscf)
        assert getattr(gradscf, name) is importlib.import_module(f"gradscf.{name}")


def test_gradscf_exposes_pyscf_style_namespaces():
    for name in (
        "gradscf.gto",
        "gradscf.dft",
        "gradscf.scf",
        "gradscf.tdscf",
    ):
        module = importlib.import_module(name)
        assert module is not None


def test_gradscf_reference_module_is_removed():
    try:
        importlib.import_module("gradscf.reference")
    except ModuleNotFoundError:
        return
    raise AssertionError("gradscf.reference should not import successfully")


def test_dft_namespace_exposes_ks_facades():
    from gradscf import dft, scf

    assert dft.RKS is scf.RKS
    assert dft.UKS is scf.UKS


def test_top_level_exposes_recommended_neural_xc_facades():
    import gradscf

    assert gradscf.Functional is gradscf.neural_xc.Functional


def test_top_level_removes_legacy_neural_xc_exports():
    import gradscf

    removed = (
        "Density" "NeuralXCFunctional",
        "Neural" "XCFunctional",
        "Pointwise" "MLP",
        "make_neural" "_lda_functional",
        "make_dm21" "_like_functional",
    )

    for name in removed:
        assert not hasattr(gradscf, name), f"{name} should not be exported at top level"


def test_gradscf_pyscf_style_submodules_import():
    for name in (
        "gradscf.gto.basis",
        "gradscf.gto.grid",
        "gradscf.dft.rks",
        "gradscf.dft.uks",
        "gradscf.dft.xc",
    ):
        module = importlib.import_module(name)
        assert module is not None
