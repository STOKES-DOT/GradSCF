"""CP on/off must compare the same fixed/adaptive basis policy and geometry."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'comparisons'))


def protocol():return import_module('nnao_dimer_controls.protocols')


def inputs():
    definitions=[];records=[];bases={}
    for i,scale in enumerate((.9,1.,2.)):
        row=dict(symbols=['H']*4,atomic_numbers=[1]*4,charge=0,spin=0,nelectron=4,
            coords_angstrom=[[0.,0.,0.],[0.,0.,.7],[0.,0.,scale*3],[0.,0.,scale*3+.7]])
        point=dict(identifier=f'pair/{scale}',dataset='sample',group='pair',label='pair',
                   scale=scale,structure=row,fragment_indices=[[0,1],[2,3]])
        definitions.append(point)
        key=f'{i:020d}'
        records.append(dict(key=key,identifier=point['identifier'],basis='NNAO/MACE',
            task='curve',status='passed',uncorrected_interaction_kcal_mol=-2.,
            cp_interaction_kcal_mol=-1.,model_interaction_kcal_mol=-99.))
        bases[key]=dict(row=row,atom_shells=[[[0,[1.,scale]]]]*4)
    assets={'H':dict(orbital_shells=[[0,[1.,1.]]],polarization_shell=[1,[.8,1.]])}
    return definitions,records,bases,assets


def test_fixed_scan_uses_reference_coefficients_without_moving_geometry():
    definitions,records,bases,assets=inputs()
    before=deepcopy(bases)
    prepared=protocol().prepare_cases(definitions,records,bases,assets)
    fixed=[r for r in prepared['jobs'] if r['policy']=='fixed_r0']
    assert len(fixed)==2 and len(prepared['anchor_reuse'])==1
    for job in fixed:
        assert job['atom_shells']==bases['00000000000000000001']['atom_shells']
        point=next(d for d in definitions if d['identifier']==job['identifier'])
        assert job['row']==point['structure']
    assert bases==before
    assert len([j for j in prepared['jobs'] if j['policy']=='conventional'])==3


def test_main_plot_uses_independent_monomers_and_additive_frozen_cp():
    # The compressed HBr--NH3 point must not lose its repulsive wall through
    # subtraction of environment-conditioned monomer energies.
    record=dict(dimer_energy_hartree=-2627.201534954131,
        isolated_energies_hartree=[-2571.0236204356233,-56.17963454827534],
        own_energies_hartree=[-2571.0114314084326,-56.169625217562896],
        uncorrected_interaction_kcal_mol=-12.850344918094976,
        cp_interaction_kcal_mol=-8.784331484225152,
        bsse_kcal_mol=4.0660134338341525)
    assert protocol().plotted_energy(record,'no_cp')==pytest.approx(1.0793349747841636,abs=1e-8)
    assert protocol().plotted_energy(record,'cp')==pytest.approx(5.145348408618316,abs=1e-8)
    record['own_energies_hartree']=[-2570.,-55.]
    assert protocol().plotted_energy(record,'no_cp')==pytest.approx(1.0793349747841636,abs=1e-8)
    with pytest.raises(ValueError):protocol().plotted_energy(record,'raw_model')


def test_main_plot_rejects_missing_independent_reference():
    record=inputs()[1][0]
    with pytest.raises(ValueError,match='isolated'):
        protocol().plotted_energy(record,'no_cp')


def test_fixed_output_uses_the_same_isolated_model_reference():
    record=dict(dimer_energy_hartree=-2627.20087664274,
        uncorrected_interaction_kcal_mol=.2992080612870129,
        cp_interaction_kcal_mol=2.9201036184256384,bsse_kcal_mol=2.6208955570940384)
    reference=[-2571.0236204356233,-56.17963454827534]
    expected=(-2627.20087664274-sum(reference))*627.5094740631
    assert protocol().plotted_energy(record,'no_cp',isolated_energies=reference)==pytest.approx(expected,abs=1e-8)
    assert protocol().plotted_energy(record,'cp',isolated_energies=reference)==pytest.approx(expected+record['bsse_kcal_mol'],abs=1e-8)


def test_missing_reference_or_changed_primitive_pool_is_rejected():
    definitions,records,bases,assets=inputs()
    with pytest.raises(ValueError,match='reference'):
        protocol().prepare_cases(definitions[:1],records[:1],bases,assets)
    bases['00000000000000000000']['atom_shells']=deepcopy(bases['00000000000000000000']['atom_shells'])
    bases['00000000000000000000']['atom_shells'][0][0][1][0]=1.1
    with pytest.raises(ValueError,match='exponent'):
        protocol().prepare_cases(definitions,records,bases,assets)


def test_missing_polarization_asset_is_explicit():
    definitions,records,bases,_=inputs()
    with pytest.raises(ValueError,match='polarization'):
        protocol().prepare_cases(definitions,records,bases,{})
