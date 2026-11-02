"""Host-only definitions for matched CP and contraction-policy comparisons."""
from copy import deepcopy
import hashlib
import json
import math


FIXED='NNAO/MACE-fixed'
POLARIZED='STO-6G+P'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _pool(shells):
    return [[[shell[0],[row[0] for row in shell[1:]],len(shell[1])-1]
              for shell in atom] for atom in shells]


def conditional_energy(record,mode):
    """Original five-energy diagnostic with dimer-conditioned fragment bases."""
    fields={'no_cp':'uncorrected_interaction_kcal_mol','cp':'cp_interaction_kcal_mol'}
    if mode not in fields:raise ValueError('Plot mode must be no_cp or cp.')
    return float(record[fields[mode]])


def plotted_energy(record,mode,*,isolated_energies=None):
    """Isolated-reference energy; CP adds the frozen-parameter correction.

    The CP-adjusted quantity retains the basis-response/reference term. The
    separate conditional ghost-subtraction diagnostic remains available above.
    """
    if mode not in {'no_cp','cp'}:raise ValueError('Plot mode must be no_cp or cp.')
    reference=record.get('isolated_energies_hartree') if isolated_energies is None else isolated_energies
    if reference is None or len(reference)!=2 or not all(math.isfinite(float(x)) for x in reference):
        raise ValueError('A finite pair of isolated monomer energies is required.')
    value=(float(record['dimer_energy_hartree'])-sum(map(float,reference)))*627.5094740631
    if mode=='cp':value+=float(record['bsse_kcal_mol'])
    if not math.isfinite(value):raise ValueError('Nonfinite plotted energy.')
    return value


def prepare_cases(definitions,records,basis_exports,polarization_assets):
    good={(r['identifier'],r['basis']):r for r in records
          if r.get('task')=='curve' and r['status']=='passed'}
    anchors={}
    for point in definitions:
        if point['scale']==1.:
            group=(point['dataset'],point['group'])
            if group in anchors:raise ValueError('Duplicate reference geometry.')
            anchors[group]=point
    jobs=[];reused=[]
    for point in definitions:
        group=(point['dataset'],point['group'])
        if group not in anchors:raise ValueError('Missing scale-1.00 reference geometry.')
        anchor=anchors[group]
        current=good[point['identifier'],'NNAO/MACE']
        reference=good[anchor['identifier'],'NNAO/MACE']
        live=basis_exports[current['key']]
        fixed=basis_exports[reference['key']]
        if point['structure']['symbols']!=fixed['row']['symbols']:
            raise ValueError('Atom order differs from reference geometry.')
        if _pool(live['atom_shells'])!=_pool(fixed['atom_shells']):
            raise ValueError('Primitive exponent pool differs across the scan.')
        def job(basis,policy,shells,source_key=None):
            value=dict(identifier=point['identifier'],dataset=point['dataset'],group=point['group'],
                label=point['label'],scale=point['scale'],basis=basis,policy=policy,
                row=deepcopy(point['structure']),fragment_indices=deepcopy(point['fragment_indices']),
                atom_shells=deepcopy(shells),reference_kcal_mol=point.get('reference_kcal_mol'),
                anchor_identifier=anchor['identifier'] if policy=='fixed_r0' else None,
                source_basis_key=source_key)
            value['key']=fingerprint(value)[:20]
            return value
        if point['scale']==1.:
            shared=deepcopy(current)
            shared.update(basis=FIXED,policy='fixed_r0',source_key=current['key'],
                          anchor_identifier=anchor['identifier'])
            shared['key']=fingerprint(dict(identifier=point['identifier'],policy='fixed_r0'))[:20]
            reused.append(shared)
        else:jobs.append(job(FIXED,'fixed_r0',fixed['atom_shells'],reference['key']))
        shells=[]
        for symbol in point['structure']['symbols']:
            if symbol not in polarization_assets:raise ValueError('Missing explicit polarization asset: '+symbol)
            asset=polarization_assets[symbol]
            shells.append(deepcopy(asset['orbital_shells'])+[deepcopy(asset['polarization_shell'])])
        jobs.append(job(POLARIZED,'conventional',shells))
    return dict(jobs=jobs,anchor_reuse=reused)
