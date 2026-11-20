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
_MAX_CHARACTERS = 16000
_MAX_NODES = 2048
_MAX_DEPTH = 64


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
    def check(node, depth=0):
        budget['nodes'] += 1
        if budget['nodes'] > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError(f'Candidate exceeds the {_MAX_NODES}-node/{_MAX_DEPTH}-level limit')
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
                if (isinstance(exponent, ast.Constant) and
                        type(exponent.value) in (int, float) and abs(exponent.value) > 8):
                    raise ValueError('Literal powers must have magnitude at most eight')
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
                    not item[0].isidentifier() or item[0].startswith('_') or
                    item[0] in reserved or item[0] in names):
                raise ValueError('Definitions require unique, nonreserved identifier names')
            names.add(item[0])
        definitions = tuple(tuple(item) for item in definitions)
        object.__setattr__(self, 'definitions', definitions)
        budget = {'characters': 0, 'nodes': 0}
        trees = {}
        for name, expression in definitions:
            tree = _parse(expression, budget)
            references = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            if (references & names) - trees.keys():
                raise ValueError('Definitions may refer only to earlier definitions')
            trees[name] = tree
        object.__setattr__(self, '_tree', _parse(self.formula, budget))
        # Resolve the dependency graph iteratively, so long alias chains do not
        # introduce Python recursion beyond the individual expression budget.
        used = set()
        pending = [self._tree]
        while pending:
            for node in ast.walk(pending.pop()):
                if isinstance(node, ast.Name) and node.id in trees and node.id not in used:
                    used.add(node.id)
                    pending.append(trees[node.id])
        object.__setattr__(self, '_definition_trees',
                           tuple((name, trees[name]) for name, _ in definitions if name in used))

    @property
    def initial(self):
        return {p.name: p.value for p in self.parameters}

    @property
    def features(self):
        trees = (self._tree, *(tree for _, tree in self._definition_trees))
        return frozenset(node.id for tree in trees for node in ast.walk(tree)
                         if isinstance(node, ast.Name)) - {
            *self.initial, *_FUNCTIONS, *(name for name, _ in self.definitions)}

    @property
    def affine(self):
        """Conservative symbolic check for parameter-affine AO caching."""
        degrees = {name: 1 for name in self.initial}
        def degree(node):
            if isinstance(node, ast.Constant): return 0
            if isinstance(node, ast.Name): return degrees.get(node.id, 0)
            if isinstance(node, ast.UnaryOp): return degree(node.operand)
            if isinstance(node, ast.Call): return 2 if degree(node.args[0]) else 0
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

    def evaluate(self, params, features):
        if set(params) != {p.name for p in self.parameters}:
            raise ValueError('Parameters differ from the declared formula contract')
        values = {}
        def walk(node):
            if isinstance(node, ast.Constant): return jnp.asarray(float(node.value))
            if isinstance(node, ast.Name):
                if node.id in values: return values[node.id]
                return jnp.asarray(params[node.id] if node.id in params else features[node.id])
            if isinstance(node, ast.BinOp):
                return _BINARY[type(node.op)](walk(node.left), walk(node.right))
            if isinstance(node, ast.UnaryOp):
                return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
            return _FUNCTIONS[node.func.id](walk(node.args[0]))
        for name, tree in self._definition_trees:
            values[name] = walk(tree)
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
