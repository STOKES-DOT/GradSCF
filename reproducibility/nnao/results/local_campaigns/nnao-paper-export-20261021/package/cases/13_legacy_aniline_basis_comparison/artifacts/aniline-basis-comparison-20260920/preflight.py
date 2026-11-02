"""Resolve the saved aniline geometry and basis sizes without ERI allocation."""
import csv
import hashlib
import json
from pathlib import Path
import runpy
from gradscf import integrals
from gradscf.model.nnao import prepare_direct_basis
root=Path(__file__).resolve().parent
source=root.parent/'aniline-basis-comparison-20260916/geometry.json'
geometry=json.loads(source.read_text())
(root/'geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
atom=list(zip(geometry['symbols'],geometry['coords_angstrom']))
module=runpy.run_path('tools/compare_methane_core6_basis_sets.py')
rows=[]
for name in [*module['STANDARD_BASES'],*module['CUSTOM_FAMILIES']]:
    if name in module['CUSTOM_FAMILIES']:
        layout=prepare_direct_basis(atom,basis_family=module['CUSTOM_FAMILIES'][name],core_primitives=6)
        top=layout.topology
    else:top,_=integrals.prepare_basis(atom,name,cart=False)
    nprim=sum((2*l+1)*n for l,n in zip(top.angular_momenta,top.primitive_counts))
    rows.append(dict(basis=name,nao=top.nao,primitive_nao=nprim,fock_shape=[top.nao,top.nao],
                     contracted_eri_gib=8*top.nao**4/2**30,primitive_eri_gib=8*nprim**4/2**30))
summary=dict(**geometry,molecule=geometry['name'],geometry_metadata=geometry,basis='szp442_direct',
             core_primitive_counts=[6]*7,cartesian=False,ecp=None)
(root/'comparison-input.json').write_text(json.dumps(summary,indent=2)+'\n')
report=dict(geometry=geometry,geometry_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            nelectron=50,occupied_orbitals=25,rows=rows,
            memory_note='Tensor sizes only, not process peak. Standard calculations use contracted ERIs; current NNAO training uses primitive ERIs.')
(root/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
with (root/'preflight.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for r in rows:print(f"{r['basis']:25s} AO={r['nao']:3d} primitive={r['primitive_nao']:3d} ERI={r['contracted_eri_gib']:.3f} GiB primitiveERI={r['primitive_eri_gib']:.3f} GiB")
