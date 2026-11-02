"""Reproducible shared-pilot model and exponent channel configuration."""
EXPONENT_KEYS = (('H', 0), ('H', 1), *((symbol, l) for symbol in ('C', 'N', 'O', 'F') for l in range(3)))
MODEL_CONFIG = dict(elements=(1, 6, 7, 8, 9), channels=8, num_interactions=2,
                    max_ell=1, correlation=2, zero_init=True, basis_family='szp663_direct')
PARENT_SOURCE = '7df11aecdbf295818de02c0db67da3de8d3db5b1ccef29798114308e28d08ebb'
PARENT_CHECKPOINT_SHA = 'ba93fb40efca3268962c7501f82627e188ed9509b6a73bd27a2df2170a3495de'

EXPANDED_ELEMENTS = tuple('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Ga Ge As Se Br Kr'.split())


def model_config_for_elements(elements, *, basis_family='nnao_szp_l'):
    """Explicit ordered vocabulary; never reinterpret an existing head/checkpoint."""
    from numbers import Integral
    from gradscf.data.molecule import atomic_number
    from .basis import supported_elements
    available = {atomic_number(s) for s in supported_elements(basis_family=basis_family)}
    numbers = tuple(atomic_number(e) if isinstance(e, str) else int(e)
                    if isinstance(e, Integral) and not isinstance(e, bool) else -1 for e in elements)
    if not numbers or len(set(numbers)) != len(numbers):
        raise ValueError('Model elements must be nonempty and unique.')
    if not set(numbers) <= available:
        raise ValueError('Model elements are outside the selected all-electron template family.')
    return dict(MODEL_CONFIG, elements=numbers, basis_family=basis_family)


def model_config_for_definition(data):
    """Old CHNOF definitions retain their exact vocabulary; expansions declare it."""
    from gradscf.data.molecule import atomic_number
    config = model_config_for_elements(data.get('elements', MODEL_CONFIG['elements']),
                                       basis_family=data['basis_family'])
    observed = {atomic_number(s) for row in data['structures'] for s in row['symbols']}
    if not observed <= set(config['elements']):
        raise ValueError('Reference elements exceed the declared model elements; specify elements explicitly.')
    return config


def exponent_keys_for_elements(elements, *, basis_family='nnao_szp_l'):
    from gradscf.data.molecule import atomic_number
    from .initialization import _direct_templates, _direct_slot
    config = model_config_for_elements(elements, basis_family=basis_family)
    templates = _direct_templates(basis_family)
    symbols = {atomic_number(s): s for s in templates}
    return tuple((symbols[z], l) for z in config['elements'] for l in sorted({
        shell['l'] for shell in templates[symbols[z]]['shells'] if _direct_slot(shell) >= 0}))
