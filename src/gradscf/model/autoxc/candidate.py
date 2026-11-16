"""Serializable analytic formulas evaluated as JAX expressions, never Python eval."""
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
import operator

import jax.numpy as jnp


_FUNCTIONS = {name: getattr(jnp, name) for name in ('sqrt', 'exp', 'log', 'log1p', 'tanh')}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}
_FIELDS = ('n', 'grad', 'sigma', 'lap', 'weights')


def _finite(value):
    try:
        return isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


@dataclass(frozen=True)
class Parameter:
    name: str
    value: float
    lower: float
    upper: float

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name.isidentifier() or
                self.name.startswith('_') or self.name in (*_FIELDS, *_FUNCTIONS)):
            raise ValueError('Parameter name must be a nonreserved identifier')
        if (not all(_finite(x) and not isinstance(x, bool) for x in (self.value, self.lower, self.upper)) or
                not self.lower < self.upper or not self.lower <= self.value <= self.upper):
            raise ValueError('Finite parameter bounds must contain its initial value')


def _parse(expression):
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 2000:
        raise ValueError('Supply a nonempty formula of at most 2000 characters')
    try:
        tree = ast.parse(expression, mode='eval').body
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('Invalid analytic formula') from exc
    count = 0
    def check(node, depth=0):
        nonlocal count
        count += 1
        if count > 120 or depth > 12:
            raise ValueError('Formula exceeds the 120-node/12-level limit')
        if isinstance(node, ast.Constant):
            if type(node.value) not in (float, int) or not _finite(node.value):
                raise ValueError('Formula constants must be finite real numbers')
        elif isinstance(node, ast.Name):
            if node.id.startswith('_') or node.id in _FUNCTIONS:
                raise ValueError('Invalid feature identifier')
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            if isinstance(node.op, ast.Pow):
                exponent = node.right
                if isinstance(exponent, ast.UnaryOp) and isinstance(exponent.op, ast.USub):
                    exponent = exponent.operand
                if not isinstance(exponent, ast.Constant) or type(exponent.value) not in (int, float) or abs(exponent.value) > 8:
                    raise ValueError('Use a fixed real power with magnitude at most eight')
            check(node.left, depth+1)
            check(node.right, depth+1)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            check(node.operand, depth+1)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
              node.func.id in _FUNCTIONS and len(node.args) == 1 and not node.keywords):
            check(node.args[0], depth+1)
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
    """
    formula: str
    parameters: tuple[Parameter, ...] = ()
    kind: str = 'potential'
    hypothesis: str = ''

    def __post_init__(self):
        if self.kind not in ('energy', 'potential') or not isinstance(self.hypothesis, str):
            raise ValueError('Declare energy or potential and a textual hypothesis')
        parameters = tuple(self.parameters)
        if (len(parameters) > 4 or any(not isinstance(p, Parameter) for p in parameters) or
                len({p.name for p in parameters}) != len(parameters)):
            raise ValueError('Supply at most four uniquely named Parameters')
        object.__setattr__(self, 'parameters', parameters)
        object.__setattr__(self, '_tree', _parse(self.formula))

    @property
    def initial(self):
        return {p.name: p.value for p in self.parameters}

    @property
    def features(self):
        return frozenset(n.id for n in ast.walk(self._tree) if isinstance(n, ast.Name)) - {
            *self.initial, *_FUNCTIONS}

    @property
    def affine(self):
        """Conservative symbolic check for parameter-affine AO caching."""
        def degree(node):
            if isinstance(node, ast.Constant): return 0
            if isinstance(node, ast.Name): return int(node.id in self.initial)
            if isinstance(node, ast.UnaryOp): return degree(node.operand)
            if isinstance(node, ast.Call): return 2 if degree(node.args[0]) else 0
            left, right = degree(node.left), degree(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)): return max(left, right)
            if isinstance(node.op, ast.Mult): return left+right
            if isinstance(node.op, ast.Div): return 2 if right else left
            exponent = ast.literal_eval(node.right)
            return 0 if not left or exponent == 0 else (left if exponent == 1 else 2)
        return degree(self._tree) <= 1

    def evaluate(self, params, features):
        if set(params) != {p.name for p in self.parameters}:
            raise ValueError('Parameters differ from the declared formula contract')
        def walk(node):
            if isinstance(node, ast.Constant): return jnp.asarray(float(node.value))
            if isinstance(node, ast.Name):
                return params[node.id] if node.id in params else features[node.id]
            if isinstance(node, ast.BinOp):
                return _BINARY[type(node.op)](walk(node.left), walk(node.right))
            if isinstance(node, ast.UnaryOp):
                return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
            return _FUNCTIONS[node.func.id](walk(node.args[0]))
        return jnp.asarray(walk(self._tree))

    def functional(self, input_fn):
        """Bind an ENERGY-density expression to the existing dft.custom_xc port."""
        from gradscf.dft import custom_xc
        if self.kind != 'energy':
            raise ValueError('A potential-only formula has no declared scalar XC energy')
        return custom_xc(input_fn=input_fn,
                         energy_fn=lambda p, x: jnp.sum(x['weights']*self.evaluate(p, x)),
                         name='autoxc_'+self.digest[:12])

    def to_dict(self):
        return {'formula': self.formula, 'parameters': [asdict(p) for p in self.parameters],
                'kind': self.kind, 'hypothesis': self.hypothesis}

    @classmethod
    def from_json(cls, payload):
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict) or set(payload) != {'formula', 'parameters', 'kind', 'hypothesis'}:
            raise ValueError('Candidate JSON requires formula, parameters, kind and hypothesis')
        try:
            return cls(payload['formula'], tuple(Parameter(**p) for p in payload['parameters']),
                       payload['kind'], payload['hypothesis'])
        except TypeError as exc:
            raise ValueError('Malformed candidate parameter fields') from exc

    @property
    def digest(self):
        payload = self.to_dict()
        payload.pop('hypothesis')
        payload['formula'] = ast.dump(self._tree)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


PROPOSAL_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['formula', 'parameters', 'kind', 'hypothesis'],
    'properties': {
        'formula': {'type': 'string'}, 'kind': {'type': 'string', 'enum': ['energy', 'potential']},
        'hypothesis': {'type': 'string'},
        'parameters': {'type': 'array', 'maxItems': 4, 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['name', 'value', 'lower', 'upper'],
            'properties': {'name': {'type': 'string'}, **{k: {'type': 'number'}
                           for k in ('value', 'lower', 'upper')}}}}}}


__all__ = ['Candidate', 'Parameter', 'PROPOSAL_SCHEMA']
