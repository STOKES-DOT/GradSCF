"""Reference exponent pools and contraction templates for NNAO initialization."""
from functools import lru_cache
from importlib.resources import files
import json
import hashlib
import numpy as np

@lru_cache(maxsize=1)
def _templates():
    return json.loads(files('gradscf.model.nnao').joinpath('data/szp3.json').read_text())['elements']


@lru_cache(maxsize=1)
def _core_reference_asset():
    raw = files('gradscf.model.nnao').joinpath('data/bse_631g_v1.json').read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def core_reference_identity(layout):
    if getattr(layout, 'core_policy', None) != '6-31g':
        return None
    return dict(policy='bse-6-31g-sz-core-sp6-d3-v1', basis='6-31G', version=1,
                sha256=_core_reference_asset()[1])


@lru_cache(maxsize=26)
def reference_core_blocks(symbol):
    """Single-zeta cores selected from original BSE 6-31G v1 records.

    Split BSE SP records into their original s and p coefficient columns.
    Inner s/p use six primitives per radial contraction. Ga-Kr retain only
    the original three-primitive d contraction as their one core d radial
    function. The separate one-primitive d block is not added to the core.
    Active polarization remains model-owned. No coefficients are refitted,
    concatenated, or taken from a 3-21G inner shell.
    """
    from gradscf.data.molecule import atomic_number
    document = _core_reference_asset()[0]
    element = document['elements'].get(str(atomic_number(symbol)))
    if element is None or element.get('ecp_potentials'):
        raise ValueError(f'No original all-electron 6-31G reference core for {symbol}.')
    grouped = {}
    for shell in element['electron_shells']:
        ls = shell['angular_momentum']
        for index, c in enumerate(shell['coefficients']):
            l = ls[index] if len(ls) > 1 else ls[0]
            rows = [[float(a), float(b)] for a, b in zip(shell['exponents'], c, strict=True)]
            grouped.setdefault(l, []).append([l, *rows])
    counts = {}
    for shell in _templates()[symbol]['shells']:
        if shell['role'] == 'core':
            counts[shell['l']] = counts.get(shell['l'], 0)+1
    for l in tuple(counts):
        if l >= 2:
            matches = [b for b in grouped.get(l, ()) if len(b)-1 == 3]
            if l != 2 or counts[l] != 1 or len(matches) != 1:
                raise ValueError(f'No unique original three-primitive core d contraction for {symbol}.')
            grouped[l] = matches
    if any(len(grouped.get(l, ())) < count for l, count in counts.items()):
        raise ValueError(f'Incomplete original core blocks for {symbol}.')
    result = tuple(b for l, count in sorted(counts.items()) for b in grouped[l][:count])
    if any(len(b)-1 != (6 if b[0] < 2 else 3) for b in result):
        raise ValueError(f'Original core blocks do not match the s/p6, d3 policy for {symbol}.')
    return result


FAMILY_GRID = {f'nnao_{label}_{tier}': (zeta, tier)
              for zeta, label in enumerate(('szp', 'dzp', 'tzp'), 1)
              for tier in ('s', 'm', 'l')}


def direct_family_dimensions(family):
    if family in FAMILY_GRID:
        zeta, tier = FAMILY_GRID[family]
        return 3*zeta, {'s': 3, 'm': 4, 'l': 6}[tier], zeta
    channels, primitives = {'szp3_direct': (2, 3), 'szp442_direct': (3, 4),
                            'szp663_direct': (3, 6)}[family]
    return channels, primitives, 1


def radial_reference_coefficients(shell, count=1):
    """Independent columns initialized in the normalized Gaussian metric.

    Keep the original contraction as the first column. Additional valence
    columns span diffuse-first primitive directions, metric-orthogonalized
    against previous columns; angular components share these radial columns.
    This is a deterministic initialization, not an atomic-SCF optimum.
    """
    c = np.asarray(shell['coefficients'], dtype=float)
    if count == 1:
        return c[:, None]
    a = np.asarray(shell['exponents'], dtype=float)
    if count > len(a):
        raise ValueError('More contractions than primitives.')
    metric = (2*np.sqrt(a[:, None]*a[None, :])/(a[:, None]+a[None, :]))**(shell['l']+1.5)
    columns = [c / np.sqrt(c@metric@c)]
    for index in np.argsort(a):
        v = np.eye(len(a))[index]
        for _ in range(2):
            for previous in columns:
                v = v-previous*(previous@metric@v)
        norm = float(v@metric@v)
        if norm > 1e-10:
            columns.append(v/np.sqrt(norm))
        if len(columns) == count:
            result = np.column_stack(columns)
            return result/np.linalg.norm(result, axis=0)
    raise ValueError('Cannot initialize independent radial contractions.')


@lru_cache(maxsize=12)
def _direct_templates(basis_family):
    from copy import deepcopy
    if basis_family in FAMILY_GRID:
        _, tier = FAMILY_GRID[basis_family]
        if tier == 'l':
            data = deepcopy(_direct_templates('szp663_direct'))
        else:
            data = deepcopy(_templates())
            for element in data.values():
                for shell in element['shells']:
                    if shell['role'] == 'core':
                        continue
                    target = (3 if tier == 's' else 4) if shell['role'].startswith('valence_') else 2
                    a, c = shell['exponents'], shell['coefficients']
                    while len(a) < target:
                        a.append(min(a)/2 if shell['l'] == 2 or shell['role'] == 'valence_p' else 2*max(a))
                        c.append(0.)
        from gradscf.data.molecule import atomic_number
        return {s: entry for s, entry in data.items() if atomic_number(s) <= 36}
    if basis_family not in {'szp3_direct','szp442_direct','szp663_direct'}:
        raise ValueError('Unknown all-electron direct basis family.')
    if basis_family=='szp663_direct':
        data=deepcopy(_direct_templates('szp442_direct'))
        for element in data.values():
            for shell in element['shells']:
                if shell['role']=='core':continue
                a=shell['exponents'];c=shell['coefficients']
                target=6 if shell['role'].startswith('valence_') else (3 if shell['l']==2 else 2)
                # Preserve old exponents, add tight/diffuse endpoints, then
                # bisect the widest logarithmic gap if one more is needed.
                additions=[2*max(a),min(a)/2]
                while len(a)<target:
                    if additions:extra=additions.pop(0)
                    else:
                        ordered=np.sort(a);i=int(np.argmax(np.diff(np.log(ordered))))
                        extra=float(np.sqrt(ordered[i]*ordered[i+1]))
                    a.append(extra);c.append(0.)
        return data
    data=deepcopy(_templates())
    if basis_family=='szp442_direct':
        for element in data.values():
            # Expand only existing p-block sp+d templates; do not add new
            # angular shells to H/He or the s-block element templates.
            if not any(s['role']=='valence_p' for s in element['shells']):continue
            for shell in element['shells']:
                if shell['role'] in {'valence_s','valence_p'} or (shell['role']=='polarization' and shell['l']==2):
                    shell['exponents'].append(min(shell['exponents'])/2.)
                    shell['coefficients'].append(0.)
    return data


def _direct_slot(shell):
    slot={'valence_s':0,'valence_p':1}.get(shell['role'],-1)
    if shell['role']=='polarization' and len(shell['exponents'])>1:
        slot=shell['l']
    return slot


def pool_channel_keys(layouts, *, min_log_spacing=1e-6):
    """Shared (element, l, role) channels, excluding all fixed contractions.

    Validate the initial pools before tracing. Repeated atoms must use the
    same exponent pool, in the same primitive order, for a shared channel.
    """
    from gradscf.data.molecule import atomic_number
    pools = {}
    for layout in layouts:
        for atom, l, role, slot, alpha in zip(layout.shell_atoms,
                layout.topology.angular_momenta, layout.roles, layout.slots,
                layout.parameters.exponents):
            if slot < 0:
                continue
            key = (layout.symbols[atom], l, role)
            alpha = np.asarray(alpha)
            if not np.isfinite(alpha).all() or np.any(alpha <= 0):
                raise ValueError(f'{key}: primitive exponents must be finite and positive.')
            if np.any(np.diff(np.sort(np.log(alpha))) <= min_log_spacing):
                raise ValueError(f'{key}: duplicate exponents or insufficient logarithmic spacing.')
            if key in pools and not np.array_equal(pools[key], alpha):
                raise ValueError(f'{key}: shared initial exponent pools differ.')
            pools[key] = alpha
    return tuple(sorted(pools, key=lambda key: (atomic_number(key[0]), key[1], key[2])))


def with_pool_parameters(layout, values, keys):
    """Positive scale/spacing deformation of the reference active exponents.

    log(alpha) = mean(log(alpha_ref)) + scale
                 + exp(log_spacing) * centered_log(alpha_ref).
    Zero parameters reproduce the reference pool; core parameters stay fixed.
    """
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    keys = tuple(keys)
    if len(set(keys)) != len(keys):
        raise ValueError('Pool channel keys must be unique.')
    values = jnp.asarray(values)
    if values.shape != (len(keys), 2):
        raise ValueError('Pool parameters require one (scale, log_spacing) pair per channel.')
    if not isinstance(values, jax.core.Tracer) and not np.isfinite(values).all():
        raise ValueError('Pool parameters must be finite.')
    indices = {key: i for i, key in enumerate(keys)}
    exponents = []
    for atom, l, role, slot, alpha in zip(layout.shell_atoms,
            layout.topology.angular_momenta, layout.roles, layout.slots,
            layout.parameters.exponents):
        if slot < 0:
            exponents.append(alpha)
            continue
        key = (layout.symbols[atom], l, role)
        if key not in indices:
            raise ValueError(f'Missing pool channel: {key}')
        scale, spacing = values[indices[key]]
        log_alpha = jnp.log(alpha)
        moved = jnp.exp(log_alpha.mean()+scale+jnp.exp(spacing)*(log_alpha-log_alpha.mean()))
        valid = jnp.all(jnp.isfinite(moved) & (moved > 0.))
        exponents.append(jnp.where(valid, moved, jnp.nan))
    return replace(layout, parameters=replace(layout.parameters, exponents=tuple(exponents)))


def bind_pool_candidate(layout, candidate):
    """Bind an A/B candidate, checking primitive order and absolute exponents.

    The saved reference must match the current template. This prevents a
    scale/spacing checkpoint from silently deforming a different initial pool.
    Fixed core coefficients and exponents are never changed.
    """
    if candidate.get('core_reference') != core_reference_identity(layout):
        raise ValueError('Candidate core reference differs; recalibrate the pool and free contractions.')
    keys = tuple(tuple(key) for key in candidate['parameters']['keys'])
    channels = {(row['element'], row['l'], row['role']): row
                for row in candidate['channels']}
    if len(channels) != len(candidate['channels']) or set(channels) != set(keys):
        raise ValueError('Candidate channels must be unique and match parameter keys.')
    moved = with_pool_parameters(layout, candidate['parameters']['values'], keys)
    for atom, l, role, slot, before, after in zip(layout.shell_atoms,
            layout.topology.angular_momenta, layout.roles, layout.slots,
            layout.parameters.exponents, moved.parameters.exponents):
        if slot < 0:
            continue
        row = channels[(layout.symbols[atom], l, role)]
        reference = np.asarray(row['reference_exponents_bohr_minus2'])
        expected = np.asarray(row['candidate_exponents_bohr_minus2'])
        if reference.shape != before.shape or not np.array_equal(reference, before):
            raise ValueError('Candidate reference exponents/order differ from the current basis.')
        if expected.shape != after.shape or not np.allclose(expected, after, rtol=1e-13, atol=0):
            raise ValueError('Absolute candidate exponents disagree with scale/spacing parameters.')
    pool_channel_keys([moved])  # Positive, distinct active exponents.
    return moved
