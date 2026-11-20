"""Compile six exchange and correlation genes into one fitted XC potential."""
import ast
from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from types import MappingProxyType

from gradscf.model.autoxc import Candidate


SLOTS = ('x_local', 'x_semilocal', 'x_nonlocal',
         'c_local', 'c_semilocal', 'c_nonlocal')
LOCAL = frozenset(('n',))
SEMILOCAL = LOCAL | {'sigma', 'grad', 's', 'u', 'lap', 'q', 'z', 'rho_occ',
                     'tau', 'tau_occ', 'alpha'}


def _rename(expression, names, *, ignore_amplitudes=False):
    def numeric_literal(node):
        if isinstance(node, ast.UnaryOp):
            node = node.operand
        return isinstance(node, ast.Constant)

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = names.get(node.id, node.id)
            return node

        def visit_BinOp(self, node):
            if ignore_amplitudes and isinstance(node.op, ast.Mult):
                left, right = numeric_literal(node.left), numeric_literal(node.right)
                if left != right:
                    a = ast.Name(id='__amplitude__', ctx=ast.Load())
                    return ast.BinOp(a, ast.Mult(), self.visit(node.right if left else node.left))
            return self.generic_visit(node)

    return ast.unparse(Rename().visit(ast.parse(expression, mode='eval')))


def _fit_name(slot, name, shared):
    return 'shared_' + name if name in shared else f'{slot}_p_{name}'


@dataclass(frozen=True)
class Genome:
    genes: Mapping[str, Candidate]
    shared: tuple[str, ...] = ()

    def __post_init__(self):
        if set(self.genes) != set(SLOTS):
            raise ValueError('A genome requires the six XC gene slots')
        if any(not isinstance(c, Candidate) or c.kind != 'potential' for c in self.genes.values()):
            raise ValueError('Every gene must define a local potential contribution')
        if any(not isinstance(name, str) or not name.isidentifier() for name in self.shared):
            raise ValueError('Shared parameters require identifier names')
        object.__setattr__(self, 'genes', MappingProxyType(dict(self.genes)))
        object.__setattr__(self, 'shared', tuple(sorted(set(self.shared))))

    def to_dict(self):
        return {'genes': {slot: self.genes[slot].to_dict() for slot in SLOTS},
                'shared': list(self.shared)}

    @classmethod
    def from_dict(cls, payload):
        return cls({slot: Candidate.from_json(gene) for slot, gene in payload['genes'].items()},
                   tuple(payload.get('shared', ())))

    def replace_gene(self, slot, gene):
        return replace(self, genes={**self.genes, slot: gene})

    def cross(self, donor, slots):
        if not slots or not set(slots) <= set(SLOTS):
            raise ValueError('Choose one or more homologous gene slots')
        if donor.shared != self.shared:
            raise ValueError('Parents use different shared-parameter contracts')
        return replace(self, genes={**self.genes, **{slot: donor.genes[slot] for slot in slots}})

    def active(self, slot):
        node = ast.parse(self.genes[slot].formula, mode='eval').body
        return not (isinstance(node, ast.Constant) and node.value == 0)

    def gene_structure(self, slot):
        gene = self.genes[slot]
        names = {p.name: f'parameter_{i}' if p.name not in self.shared else 'shared_' + p.name
                 for i, p in enumerate(gene.parameters)}
        names.update({name: f'definition_{i}' for i, (name, _) in enumerate(gene.definitions)})
        return [_rename(gene.formula, names, ignore_amplitudes=True),
                [_rename(expr, names, ignore_amplitudes=True) for _, expr in gene.definitions],
                len(gene.parameters)]

    @property
    def structure(self):
        return json.dumps([self.gene_structure(slot) for slot in SLOTS], separators=(',', ':'))

    def compile(self, registry):
        definitions, parameters, contributions = [], {}, []
        features = set()
        for slot in SLOTS:
            gene = self.genes[slot]
            allowed = (LOCAL if slot.endswith('_local') else
                       SEMILOCAL if slot.endswith('_semilocal') else registry)
            unavailable = gene.features - (allowed & registry)
            if unavailable:
                raise ValueError(f'{slot}: unavailable inputs {sorted(unavailable)}')
            if slot.endswith('_nonlocal') and self.active(slot) and not gene.features - SEMILOCAL:
                raise ValueError(f'{slot}: active gene requires a nonlocal input')
            if set(gene.initial) & registry:
                raise ValueError(f'{slot}: parameter masks an input feature')
            features.update(gene.features)
            names = {name: f'{slot}_d_{name}' for name, _ in gene.definitions}
            for param in gene.parameters:
                name = _fit_name(slot, param.name, self.shared)
                names[param.name] = name
                old = parameters.setdefault(name, replace(param, name=name))
                if (old.lower, old.upper) != (param.lower, param.upper):
                    raise ValueError(f'{slot}: incompatible bounds for shared parameter {param.name}')
            definitions.extend((names[name], _rename(expr, names)) for name, expr in gene.definitions)
            if self.active(slot):
                name = f'{slot}_potential'
                definitions.append((name, _rename(gene.formula, names)))
                contributions.append(name)
        candidate = Candidate(' + '.join(contributions) or '0', tuple(parameters.values()),
                              definitions=tuple(definitions))
        if candidate.features != features:
            raise ValueError('A compiled gene identifier masks an input feature')
        return candidate

    def with_coefficients(self, values):
        return replace(self, genes={
            slot: replace(gene, parameters=tuple(
                replace(p, value=values[_fit_name(slot, p.name, self.shared)])
                for p in gene.parameters))
            for slot, gene in self.genes.items()
        })
