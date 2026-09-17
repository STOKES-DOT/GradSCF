import pytest
from comparisons.ten_system_scf_matrix import SYSTEMS, XC_REFERENCE, cases_for, validate_resume


def test_matrix_covers_ten_systems_and_all_supported_classic_methods():
    assert len(SYSTEMS)==10
    rows=[(name,*case) for name,spec in SYSTEMS.items() for case in cases_for(spec)]
    assert len(rows)==260
    assert sum(bool(row[-1]) for row in rows)==236
    assert {row[1] for row in rows}=={"RHF","UHF","ROHF","GHF","RKS","UKS","ROKS","GKS","GKS_ncol"}
    assert set(XC_REFERENCE)=={"lda","svwn","pbe","pbe0","b3lyp"}


def test_restricted_methods_are_not_applied_to_open_shell_systems():
    for spec in SYSTEMS.values():
        for method,xc,applicable in cases_for(spec):
            assert bool(applicable)==(method not in {"RHF","RKS"} or spec["spin"]==0)
            if method=="GKS_ncol":assert xc in {"lda","svwn"}


def test_lda_reference_includes_pw_correlation():
    assert XC_REFERENCE["lda"]=="LDA_X + LDA_C_PW"
    assert "VWN_RPA" in XC_REFERENCE["b3lyp"]


def test_resume_refuses_changed_runtime_or_numerical_settings():
    identity=dict(system=SYSTEMS["H2"],basis="3-21g",grid_level=1,max_cycle=200,
                  source_hashes={"solver.py":"abc"},benchmark_sha256="xyz")
    validate_resume(identity,identity.copy())
    for key,new in [("basis","sto-3g"),("grid_level",2),("source_hashes",{}),("benchmark_sha256","new")]:
        with pytest.raises(ValueError,match="new output directory"):
            validate_resume(identity,dict(identity,**{key:new}))
