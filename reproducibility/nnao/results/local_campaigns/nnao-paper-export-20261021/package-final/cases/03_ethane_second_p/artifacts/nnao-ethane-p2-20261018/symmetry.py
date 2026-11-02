"""Reflected ethane torsion atom maps; radial coefficients transform by atom."""
import numpy as np


def mirror_atom_map(source,target):
    if abs(source['angle_deg']+target['angle_deg']-120.)>1e-12:
        raise ValueError('Expected torsion angles summing to 120 degrees.')
    symbols=np.asarray(source['symbols'])
    if source['symbols']!=target['symbols']:
        raise ValueError('Source and target element ordering differs.')
    x=np.asarray(source['coords_angstrom']);y=np.asarray(target['coords_angstrom'])
    carbons=np.flatnonzero(symbols=='C');hydrogens=np.flatnonzero(symbols=='H')
    if len(carbons)!=2 or len(hydrogens)!=6:raise ValueError('Expected ethane.')
    pivot=x[carbons[0]];axis=x[carbons[1]]-pivot;axis/=np.linalg.norm(axis)
    hydrogen=hydrogens[np.linalg.norm(x[hydrogens]-pivot,axis=1).argmin()]
    normal=np.cross(axis,x[hydrogen]-pivot);normal/=np.linalg.norm(normal)
    reflected=x-2*np.outer((x-pivot)@normal,normal)
    distances=np.linalg.norm(y[:,None]-reflected[None,:],axis=2)
    distances=np.where(symbols[:,None]==symbols[None,:],distances,np.inf)
    permutation=distances.argmin(axis=1)
    error=float(distances[np.arange(len(x)),permutation].max())
    if len(set(permutation.tolist()))!=len(x) or error>1e-11:
        raise ValueError(f'No exact reflected atom mapping: maximum residual {error}.')
    return permutation,error
