"""Serializable analytic formulas evaluated as JAX expressions, never Python eval."""
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
import operator

import jax.numpy as jnp


_FUNCTIONS = {name: getattr(jnp, name) for name in ('sqrt', 'exp', 'log', 'log1p', 'tanh')}
PAIR_OPERATORS = ('pair_integral', 'pair_average', 'pair_moment')
_PAIR_DISTANCE = 'r2'
_SOURCE_PREFIX = 'source_'
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}
_FIELDS = ('n', 'grad', 'sigma', 'lap', 'weights')
_MAX_CHARACTERS = 16000
_MAX_NODES = 2048
_MAX_DEPTH = 64


def _finite(value):
    try:
        return isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _reserved(name):
    return (name.startswith(('_', _SOURCE_PREFIX))
            or name in (*_FUNCTIONS, *PAIR_OPERATORS, _PAIR_DISTANCE))


def _references(tree):
    return {node.id.removeprefix(_SOURCE_PREFIX) for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id not in
            (*_FUNCTIONS, *PAIR_OPERATORS, _PAIR_DISTANCE)}


def _pair_calls(tree):
    return {node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and node.func.id in PAIR_OPERATORS}


@dataclass(frozen=True)
class Parameter:
    name: str
    value: float
    lower: float
    upper: float

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name.isidentifier() or
                _reserved(self.name) or self.name in _FIELDS):
            raise ValueError('Parameter name must be a nonreserved identifier')
        if (not all(_finite(x) and not isinstance(x, bool) for x in (self.value, self.lower, self.upper)) or
                not self.lower < self.upper or not self.lower <= self.value <= self.upper):
            raise ValueError('Finite parameter bounds must contain its initial value')


def _parse(expression, budget):
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError('Supply a nonempty analytic formula')
    budget['characters'] += len(expression)
    if budget['characters'] > _MAX_CHARACTERS:
        raise ValueError(f'Candidate exceeds the {_MAX_CHARACTERS}-character limit')
    try:
        tree = ast.parse(expression, mode='eval').body
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('Invalid analytic formula') from exc
    def check(node, depth=0, kernel=False):
        budget['nodes'] += 1
        if budget['nodes'] > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError(f'Candidate exceeds the {_MAX_NODES}-node/{_MAX_DEPTH}-level limit')
        if isinstance(node, ast.Constant):
            if type(node.value) not in (float, int) or not _finite(node.value):
                raise ValueError('Formula constants must be finite real numbers')
        elif isinstance(node, ast.Name):
            if node.id.startswith('_') or node.id in (*_FUNCTIONS, *PAIR_OPERATORS):
                raise ValueError('Invalid feature identifier')
            if not kernel and (node.id == _PAIR_DISTANCE or node.id.startswith(_SOURCE_PREFIX)):
                raise ValueError('r2 and source_ names are available inside a pair kernel only')
            if node.id.startswith(_SOURCE_PREFIX) and (
                    not node.id.removeprefix(_SOURCE_PREFIX) or
                    _reserved(node.id.removeprefix(_SOURCE_PREFIX))):
                raise ValueError('A source_ name must identify a grid field')
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            if isinstance(node.op, ast.Pow):
                exponent = node.right
                if isinstance(exponent, ast.UnaryOp) and isinstance(exponent.op, ast.USub):
                    exponent = exponent.operand
                if (isinstance(exponent, ast.Constant) and
                        type(exponent.value) in (int, float) and abs(exponent.value) > 8):
                    raise ValueError('Literal powers must have magnitude at most eight')
            check(node.left, depth+1, kernel)
            check(node.right, depth+1, kernel)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            check(node.operand, depth+1, kernel)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
              node.func.id in _FUNCTIONS and len(node.args) == 1 and not node.keywords):
            check(node.args[0], depth+1, kernel)
        elif (not kernel and isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in PAIR_OPERATORS and not node.keywords):
            if node.func.id == 'pair_moment':
                if (len(node.args) != 3 or not isinstance(node.args[2], ast.Constant)
                        or type(node.args[2].value) is not int or node.args[2].value not in (1, 2)):
                    raise ValueError('pair_moment requires a field, kernel and literal rank 1 or 2')
                check(node.args[2], depth+1)
            elif len(node.args) != 2:
                raise ValueError('Pair quadrature requires a field and a kernel')
            check(node.args[0], depth+1)
            check(node.args[1], depth+1, kernel=True)
        else:
            raise ValueError('Only scalar arithmetic and declared math functions are allowed')
    check(tree)
    return tree


@dataclass(frozen=True)
class Candidate:
    """Pointwise potential (Ha) or energy density (Ha/bohr^3), with explicit inputs.

    n, grad, sigma and lap conventionally denote total density, gradient norm,
    squared gradient and Laplacian in atomic units. The caller supplies features
    and any regularization; no density clipping or hidden PBE is introduced.
    Other nonreserved feature names are allowed for external input providers.
    Optional ``definitions`` are ordered ``(name, expression)`` pairs; each may
    refer to earlier definitions. Only definitions used by the formula are
    evaluated, and ``features`` reports their external leaf inputs.
    """
    formula: str
    parameters: tuple[Parameter, ...] = ()
    kind: str = 'potential'
    hypothesis: str = ''
    definitions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.kind not in ('energy', 'potential') or not isinstance(self.hypothesis, str):
            raise ValueError('Declare energy or potential and a textual hypothesis')
        parameters = tuple(self.parameters)
        if (any(not isinstance(p, Parameter) for p in parameters) or
                len({p.name for p in parameters}) != len(parameters)):
            raise ValueError('Supply uniquely named Parameters')
        object.__setattr__(self, 'parameters', parameters)
        try:
            definitions = tuple(self.definitions)
        except TypeError as exc:
            raise ValueError('Definitions must be (name, expression) pairs') from exc
        reserved = {*_FIELDS, *_FUNCTIONS, *(p.name for p in parameters)}
        names = set()
        for item in definitions:
            if (not isinstance(item, (tuple, list)) or len(item) != 2 or
                    not isinstance(item[0], str) or
                    not item[0].isidentifier() or _reserved(item[0]) or
                    item[0] in reserved or item[0] in names):
                raise ValueError('Definitions require unique, nonreserved identifier names')
            names.add(item[0])
        definitions = tuple(tuple(item) for item in definitions)
        object.__setattr__(self, 'definitions', definitions)
        budget = {'characters': 0, 'nodes': 0}
        trees = {}
        for name, expression in definitions:
            tree = _parse(expression, budget)
            references = _references(tree)
            if (references & names) - trees.keys():
                raise ValueError('Definitions may refer only to earlier definitions')
            trees[name] = tree
        formula = _parse(self.formula, budget)
        parameter_names = {p.name for p in parameters}
        for tree in (formula, *trees.values()):
            if any(isinstance(node, ast.Name) and node.id.startswith(_SOURCE_PREFIX)
                   and node.id.removeprefix(_SOURCE_PREFIX) in parameter_names
                   for node in ast.walk(tree)):
                raise ValueError('Reference a kernel parameter without the source_ prefix')
        object.__setattr__(self, '_tree', formula)
        # Resolve the dependency graph iteratively, so long alias chains do not
        # introduce Python recursion beyond the individual expression budget.
        used = set()
        pending = [self._tree]
        while pending:
            for name in (_references(pending.pop()) & trees.keys()) - used:
                used.add(name)
                pending.append(trees[name])
        object.__setattr__(self, '_definition_trees',
                           tuple((name, trees[name]) for name, _ in definitions if name in used))

    @property
    def initial(self):
        return {p.name: p.value for p in self.parameters}

    @property
    def features(self):
        trees = (self._tree, *(tree for _, tree in self._definition_trees))
        return frozenset().union(*map(_references, trees)) - {
            *self.initial, *(name for name, _ in self.definitions)}

    @property
    def pair_operators(self):
        trees = (self._tree, *(tree for _, tree in self._definition_trees))
        return frozenset().union(*map(_pair_calls, trees))

    @property
    def fixed_pair_definitions(self):
        """Named quadratures that are independent of all fitted parameters."""
        dynamic = set(self.initial)
        fixed = []
        for name, tree in self._definition_trees:
            if _references(tree) & dynamic:
                dynamic.add(name)
            elif _pair_calls(tree):
                fixed.append(name)
        return tuple(fixed)

    @property
    def affine(self):
        """Conservative symbolic check for parameter-affine AO caching."""
        degrees = {name: 1 for name in self.initial}
        def degree(node):
            if isinstance(node, ast.Constant): return 0
            if isinstance(node, ast.Name): return degrees.get(node.id.removeprefix(_SOURCE_PREFIX), 0)
            if isinstance(node, ast.UnaryOp): return degree(node.operand)
            if isinstance(node, ast.Call):
                if node.func.id in PAIR_OPERATORS:
                    if degree(node.args[1]):
                        return 2
                    source = degree(node.args[0])
                    return min(2, 2*source) if node.func.id == 'pair_moment' else source
                return 2 if degree(node.args[0]) else 0
            left, right = degree(node.left), degree(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)): return max(left, right)
            if isinstance(node.op, ast.Mult): return min(2, left+right)
            if isinstance(node.op, ast.Div): return 2 if right else left
            if right: return 2
            try:
                exponent = ast.literal_eval(node.right)
            except (ValueError, TypeError):
                return 2 if left else 0
            return 0 if not left or exponent == 0 else (left if exponent == 1 else 2)
        for name, tree in self._definition_trees:
            degrees[name] = degree(tree)
        return degree(self._tree) <= 1

    def evaluate(self, params, features, *, pair=None):
        if set(params) != {p.name for p in self.parameters}:
            raise ValueError('Parameters differ from the declared formula contract')
        values = {}
        def field(name):
            if name in values:
                return values[name]
            return jnp.asarray(params[name] if name in params else features[name])

        def walk(node, lookup):
            if isinstance(node, ast.Constant): return jnp.asarray(float(node.value))
            if isinstance(node, ast.Name): return lookup(node.id)
            if isinstance(node, ast.BinOp):
                return _BINARY[type(node.op)](walk(node.left, lookup), walk(node.right, lookup))
            if isinstance(node, ast.UnaryOp):
                value = walk(node.operand, lookup)
                return -value if isinstance(node.op, ast.USub) else value
            if node.func.id in _FUNCTIONS:
                return _FUNCTIONS[node.func.id](walk(node.args[0], lookup))
            if pair is None:
                raise ValueError('A pair density operator requires a quadrature grid')

            def kernel(r2, target, source):
                def pair_field(name):
                    if name == _PAIR_DISTANCE:
                        return r2
                    if name in params:
                        return params[name]
                    if name.startswith(_SOURCE_PREFIX):
                        return source(field(name.removeprefix(_SOURCE_PREFIX)))
                    return target(field(name))
                return walk(node.args[1], pair_field)

            rank = node.args[2].value if node.func.id == 'pair_moment' else 0
            return pair(node.func.id, walk(node.args[0], lookup), kernel, rank)

        for name, tree in self._definition_trees:
            values[name] = walk(tree, field)
        return jnp.asarray(walk(self._tree, field))

    def functional(self, input_fn):
        """Bind an ENERGY-density expression to the existing dft.custom_xc port."""
        from gradscf.dft import custom_xc
        if self.kind != 'energy':
            raise ValueError('A potential-only formula has no declared scalar XC energy')
        if self.pair_operators:
            raise ValueError('Pair density operators need the grid-aware potential binding')
        return custom_xc(input_fn=input_fn,
                         energy_fn=lambda p, x: jnp.sum(x['weights']*self.evaluate(p, x)),
                         name='autoxc_'+self.digest[:12])

    def to_dict(self):
        payload = {'formula': self.formula, 'parameters': [asdict(p) for p in self.parameters],
                   'kind': self.kind, 'hypothesis': self.hypothesis}
        if self.definitions:
            payload['definitions'] = [{'name': name, 'expression': expression}
                                      for name, expression in self.definitions]
        return payload

    @classmethod
    def from_json(cls, payload):
        if isinstance(payload, str):
            payload = json.loads(payload)
        required = {'formula', 'parameters', 'kind', 'hypothesis'}
        if (not isinstance(payload, dict) or not required <= payload.keys() or
                payload.keys() - required - {'definitions'}):
            raise ValueError('Candidate JSON requires formula, parameters, kind and hypothesis')
        if 'definitions' in payload and not isinstance(payload['definitions'], list):
            raise ValueError('Candidate definitions must be a JSON array')
        try:
            definitions = []
            for item in payload.get('definitions', ()):
                if not isinstance(item, dict) or set(item) != {'name', 'expression'}:
                    raise ValueError('Definition JSON requires name and expression')
                definitions.append((item['name'], item['expression']))
            return cls(payload['formula'], tuple(Parameter(**p) for p in payload['parameters']),
                       payload['kind'], payload['hypothesis'], tuple(definitions))
        except TypeError as exc:
            raise ValueError('Malformed candidate parameter fields') from exc

    @property
    def digest(self):
        payload = self.to_dict()
        payload.pop('hypothesis')
        payload['formula'] = ast.dump(self._tree)
        if self.definitions:
            budget = {'characters': 0, 'nodes': 0}
            payload['definitions'] = [{'name': name, 'expression': ast.dump(_parse(expression, budget))}
                                      for name, expression in self.definitions]
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


PROPOSAL_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['formula', 'parameters', 'kind', 'hypothesis'],
    'properties': {
        'formula': {'type': 'string'}, 'kind': {'type': 'string', 'enum': ['energy', 'potential']},
        'hypothesis': {'type': 'string'},
        'definitions': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['name', 'expression'],
            'properties': {'name': {'type': 'string'}, 'expression': {'type': 'string'}}}},
        'parameters': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['name', 'value', 'lower', 'upper'],
            'properties': {'name': {'type': 'string'}, **{k: {'type': 'number'}
                           for k in ('value', 'lower', 'upper')}}}}}}


__all__ = ['Candidate', 'Parameter', 'PROPOSAL_SCHEMA']
