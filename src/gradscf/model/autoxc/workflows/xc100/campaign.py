"""One Slurm-owned numerical worker for the 100-target PBE training campaign."""
from dataclasses import replace
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.model.autoxc import Candidate, Evaluation, Evaluator, Parameter
from gradscf.model.autoxc.history import source_digest, versions, write_json
from gradscf.dft.iks.objectives import wu_yang_density_loss
from .molecule import XC100Case, file_hash
from .checks import baseline_checks, candidate_checks
from .direct import BLOCK_SIZE, GGA_FIELDS, PROJECTION_MEMORY_MB
from .reference import reference_baseline


IDS = tuple(f'{i:03d}' for i in range(1, 101))
LDA_EXCHANGE_POTENTIAL = '-0.9847450218426965*n**0.3333333333333333'
POLICY = {'baseline': 'PBE+PW92', 'targets': 100, 'split': 'all training',
          'max_parameters': 4, 'coefficient_bounds': [-.2, .2], 'floor': 1e-12,
          'uniform_limit_tolerance': 1e-12, 'reward_scale_hartree': .001,
          'gap_tolerance_hartree': 1e-7, 'streaming': True}
POLICY['archive_occupation_tolerance'] = 1e-8
POLICY.update(kind='potential', features=sorted(GGA_FIELDS), block_size=BLOCK_SIZE,
              projection_memory_mb=PROJECTION_MEMORY_MB, parent_energy_required=False)


class CampaignReleased(Exception):
    """The controller requested release at a safe numerical checkpoint."""


def memory():
    stats = jax.devices()[0].memory_stats() or {}
    return {'rss_peak_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            'gpu_bytes_in_use': stats.get('bytes_in_use'),
            'gpu_peak_bytes': stats.get('peak_bytes_in_use')}


def validate_candidate(candidate):
    if (candidate.kind!='potential' or not candidate.affine or not candidate.features <= GGA_FIELDS or
            set(candidate.initial) & XC100Case.fields):
        raise ValueError('Use a potential-only affine expression of density/gradient features; no energy, lap, q or z')
    if any(p.lower < -.2 or p.upper > .2 for p in candidate.parameters):
        raise ValueError('Frozen campaign coefficient bounds must stay inside [-0.2,0.2]')
    n = jnp.array([1e-6, 1e-4, 1., 10.])
    uniform = {name: jnp.zeros_like(n) for name in XC100Case.fields}
    uniform['n'] = n
    parameter_sets = [{p.name: 0. for p in candidate.parameters}]
    for p in candidate.parameters:
        parameter_sets.append({q.name: float(q.name == p.name) for q in candidate.parameters})
    for params in parameter_sets:
        value = candidate.evaluate(params, uniform)
        if not bool(jnp.all(jnp.isfinite(value)) & jnp.all(jnp.abs(value) <= 1e-12)):
            raise ValueError('Correction must vanish in the sampled uniform-density limit for every coefficient')
    probe = {name: jnp.ones_like(n) for name in XC100Case.fields}
    probe['n'] = n
    for params in parameter_sets:
        if not bool(jnp.all(jnp.isfinite(candidate.evaluate(params, probe)))):
            raise ValueError('Correction is nonfinite on the declared density-domain probe')


class Campaign:
    def __init__(self, root, *, objective=None, maxiter=30, gradient_tolerance=1e-8,
                 gradient_only=False, require_convergence=False, candidate_mode='legacy',
                 max_seconds=None, unbounded_waits=False):
        objectives = {
            'legacy': ('wu_yang', 'wy_l1_l2'),
            'open': ('wu_yang',),
            'open_xc': ('wy_energy',),
            'standalone_potential': ('wu_yang',),
        }
        if candidate_mode not in objectives:
            raise ValueError(f'Unknown XC100 mode {candidate_mode}')
        allowed = objectives[candidate_mode]
        objective = allowed[0] if objective is None else objective
        if objective not in allowed:
            raise ValueError(f'{candidate_mode} requires objective from {allowed}')
        self.root = Path(root)
        self.data = self.root/'data/xc100'
        self.cases = {}
        self.rows = []
        self.objective = objective
        self.candidate_mode = candidate_mode
        if type(unbounded_waits) is not bool:
            raise ValueError('unbounded_waits must be a boolean')
        if unbounded_waits and max_seconds is not None:
            raise ValueError('Unbounded worker waits require max_seconds=null')
        self.unbounded_waits = unbounded_waits
        self.open_design = candidate_mode != 'legacy'
        self.complete_potential = candidate_mode in ('standalone_potential', 'open_xc')
        self.normalization = None
        self.energy_normalization = None
        self.fit_options = dict(maxiter=maxiter,gradient_tolerance=gradient_tolerance,
            gradient_only=gradient_only,require_convergence=require_convergence,streaming=True,max_seconds=max_seconds)
        self.policy = dict(POLICY,objective=objective,inner_objective='wu_yang',
            outer_weights=[.5,.25,.25] if objective=='wy_l1_l2' else None,
            inner_convergence=self.fit_options)
        self.policy['candidate_mode']=candidate_mode
        if unbounded_waits:
            self.policy['unbounded_waits'] = True
        if self.open_design:
            self.policy.update(max_parameters=None,coefficient_bounds=None,
                uniform_limit_tolerance=None,reference_energy_labels=False,selection='wy_first')
            if self.complete_potential:
                self.policy.update(features=sorted(XC100Case.fields), potential_is_complete=True)
            else:
                from .ingredients import AVAILABLE_INPUTS,ORBITAL_WINDOWS
                self.policy.update(features=sorted(AVAILABLE_INPUTS),
                    orbital_reference='PBE@fixed-target-density',orbital_windows_hartree=ORBITAL_WINDOWS)
        if candidate_mode == 'open_xc':
            self.policy.update(reference_energy_labels=True, selection='joint_reward',
                outer_weights=[.5, .5], energy_label='Exc_TZ', energy_path='linear_density',
                energy_density='wu_yang_output')
        self.core = source_digest()
        self.started = time.monotonic()
        self.helpers = {name:file_hash(Path(__file__).with_name(name)) for name in
                        ('campaign.py','molecule.py','direct.py','reference.py','pbe.py')}
        if self.open_design:self.helpers['ingredients.py']=file_hash(Path(__file__).with_name('ingredients.py'))

    def status(self, phase, **details):
        write_json(self.root/'status.json', {'phase': phase, 'job_id': os.environ.get('SLURM_JOB_ID'),
            'time': time.time(), 'elapsed_seconds': time.monotonic()-self.started,
            'completed_baselines': len(self.rows), **memory(), **details})

    def check_release(self):
        if (self.root/'release.flag').exists():
            raise CampaignReleased()

    def prepare(self):
        self.check_release()
        if jax.default_backend() != 'gpu' or len(jax.devices()) != 1 or not jax.config.jax_enable_x64:
            raise RuntimeError('Campaign requires the allocated single GPU and float64')
        if len(os.sched_getaffinity(0)) > 64:
            raise RuntimeError('CPU affinity exceeds the authorized allocation')
        self.status('preparing_baselines')
        receipt = json.loads((self.root/'reference_baselines.json').read_text())
        write_json(self.root/'ready.json', None)
        self.prepare_cases(IDS, receipt)
        self.finish_preparation(receipt)

    def prepare_cases(self, sample_ids, receipt, *, persist=True):
        for identifier in sample_ids:
            self.check_release()
            # rsync publishes each completed archive before its metadata. This
            # allows useful preparation inside the allocation during transfer.
            waiting = time.monotonic()
            required = [self.data/'canonical_basis.json', self.data/identifier/'density.npz',
                        self.data/identifier/'metadata.json']
            while not all(path.exists() for path in required):
                self.check_release()
                if not self.unbounded_waits and time.monotonic()-waiting > 7200:
                    raise TimeoutError(f'Qualified input transfer stalled at {identifier}')
                self.status('waiting_for_data', current_system=identifier)
                time.sleep(10)
            start = time.monotonic()
            case = XC100Case(self.data, identifier, floor=POLICY['floor'],
                             max_parameters=None if self.open_design else 4,
                             block_size=BLOCK_SIZE)
            include_energy = not self.open_design
            baseline = (reference_baseline(self.root, case, include_energy=include_energy)
                        if receipt is not None else None)
            case.prepare(baseline=baseline, include_energy=include_energy)
            if self.open_design and not self.complete_potential:
                from .ingredients import prepare_case_orbitals,features_for_candidate,AVAILABLE_INPUTS
                self.status('preparing_physical_inputs',current_system=identifier)
                try:
                    prepare_case_orbitals(case)
                    fixed_inputs=features_for_candidate(case,AVAILABLE_INPUTS)
                    if not all(bool(jnp.all(jnp.isfinite(v))) for v in fixed_inputs.values()):
                        raise ValueError('Declared physical input fields must be finite')
                except ValueError as exc:
                    raise ValueError(f'Fixed physical inputs {identifier} failed before model evaluation: {exc}') from exc
            score = case.reporter.baseline
            grid_error = abs(float(np.dot(case.arrays['grid_weights'], case.arrays['rho']))-case.target.nelec)
            if not bool(score.valid) or grid_error > 1e-6 or not np.isfinite(float(score.density_error)):
                raise ValueError(f'Baseline qualification failed for {identifier}; no sample may be omitted')
            row = {'system_id': identifier, 'name': case.metadata['symbols'],
                'nelec': case.target.nelec, 'nao': case.target.overlap.shape[0],
                'grid_points': case.grid.ngrids, 'fingerprint': case.fingerprint,
                'xc_energy_hartree': case.baseline_energy, 'W_hartree': float(score.value),
                'G_hartree': float(score.variational_gap), 'gap_hartree': float(score.gap),
                'density_error': float(score.density_error),'density_l1':float(score.density_l1),
                'grid_electron_error': grid_error,
                'seconds': time.monotonic()-start, 'cache_reused': receipt is not None,
                'reference_origin_core': receipt['origin_core'] if receipt is not None else self.core}
            if self.objective == 'wy_energy':
                reference = case.metadata.get('labels', {}).get('Exc_TZ')
                if not isinstance(reference, (int, float)) or not np.isfinite(reference):
                    raise ValueError(f'Qualified case {identifier} requires a finite Exc_TZ label')
                pbe = float(case._energy(case.baseline_functional, None, score.density))
                if not np.isfinite(pbe):
                    raise ValueError(f'Nonfinite PBE output-density XC energy for {identifier}')
                row.update(reference_xc_energy_hartree=reference, pbe_output_xc_energy_hartree=pbe)
            if receipt is not None:
                original = json.loads((self.root/'reference/cache/cases'/identifier/'baseline.json').read_text())
                fields = ('W_hartree','G_hartree','gap_hartree','density_error')
                if not all(np.isclose(row[field],original[field],atol=1e-8,rtol=1e-9) for field in fields):
                    raise ValueError(f'Imported PBE baseline score changed for {identifier}')
            if persist:
                write_json(self.root/'cache/cases'/identifier/'baseline.json', row)
            self.cases[identifier] = case
            self.rows.append(row)
            self.status('preparing_baselines', current_system=identifier)
            print('Baseline', identifier, len(self.rows), 'seconds', row['seconds'], flush=True)

    def finish_preparation(self, receipt):
        # These operators are immutable reference inputs. Reuse their original
        # independent validation, while the new baseline score is checked above.
        oracles = {}
        if receipt is not None:
            oracle_path = self.root/'reference/baseline_oracles.json'
            if file_hash(oracle_path)!=receipt['oracle_sha256']:
                raise ValueError('Reference PBE oracle receipt changed')
            oracles = json.loads(oracle_path.read_text())
            if len(oracles)!=6 or not all(row['passed'] for row in oracles.values()):
                raise ValueError('Reference PBE oracle receipt is incomplete')
            write_json(self.root/'baseline_oracles.json', {'reused_reference':True,'oracles':oracles})

        if self.objective=='wy_l1_l2' or self.open_design:
            self.normalization = (
                float(np.mean([r['G_hartree']/r['nelec'] for r in self.rows])),
                float(np.mean([r['density_l1'] for r in self.rows])),
                float(np.mean([r['density_error']**2 for r in self.rows])))
            if not all(np.isfinite(s) and s>0 for s in self.normalization):
                raise ValueError('Composite scoring needs finite positive fixed PBE scales')
        record = {'policy': self.policy, 'cases': {r['system_id']: r['fingerprint'] for r in self.rows},
                  'core_source': self.core, 'workflow':self.helpers,
                  'outer_normalization':self.normalization,
                  'reference_inputs': (file_hash(self.root/'reference_baselines.json')
                                       if receipt is not None else 'fresh_pbe_from_qualified_density_archives'),
                  'versions': versions(), 'device': jax.devices()[0].device_kind}
        if self.objective == 'wy_energy':
            self.energy_normalization = float(np.mean([
                abs(r['pbe_output_xc_energy_hartree'] - r['reference_xc_energy_hartree'])/r['nelec']
                for r in self.rows]))
            if not np.isfinite(self.energy_normalization) or self.energy_normalization <= 0:
                raise ValueError('Joint scoring needs a finite positive PBE XC energy error scale')
            record['xc_energy_normalization_hartree_per_electron'] = self.energy_normalization
        if hasattr(self, 'execution'):
            record['execution'] = self.execution
        self.fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        self.configuration = {'sample_ids': IDS, 'fit_ids': IDS, 'reward_ids': IDS,
            'fingerprint': self.fingerprint, **self.fit_options, 'worker': record}
        write_json(self.root/'baselines.json', {'rows': self.rows, 'oracles': oracles,
            'mean_density_error': float(np.mean([r['density_error'] for r in self.rows]))})

        preflight_reward = self.validate_prepared()
        ready = {'configuration': self.configuration, 'policy': self.policy, 'baseline_count': len(self.rows),
                 'source': self.core, 'job_id': os.environ.get('SLURM_JOB_ID'),
                 'memory': memory(), 'preflight_reward': preflight_reward}
        if self.open_design and self.objective != 'wy_energy':
            ready['selection']={'density_scales':self.normalization[1:],'tolerance':1e-8}
        for name in ('inbox', 'outbox'):
            (self.root / name).mkdir(exist_ok=True)
        write_json(self.root/'ready.json', ready)
        self.status('ready')

    def validate_prepared(self):
        self.status('potential_acceptance')
        base = LDA_EXCHANGE_POTENTIAL + '+' if self.complete_potential else ''
        probe = Candidate(base+'a*n**0.3333333333333333*u', kind='potential',
                          parameters=(Parameter('a', 0., -.02, .02),))
        subset = ('010','017','065')
        options = self.fit_options if self.fit_options['require_convergence'] else dict(maxiter=1,streaming=True)
        checked = Evaluator(lambda candidate:self.bind(candidate,subset),sample_ids=subset,
            fingerprint=self.fingerprint,**options).evaluate(probe)
        write_json(self.root/'preflight.json', checked)
        if not checked.valid:
            raise ValueError('Representative direct-potential acceptance failed')
        return checked.reward

    def bind(self, candidate, sample_ids=None):
        sample_ids = tuple(self.cases) if sample_ids is None else sample_ids
        if not self.open_design:validate_candidate(candidate)
        elif candidate.kind!='potential' or not candidate.features<=set(self.policy['features']):
            raise ValueError('Open iKS candidates require a local potential and declared physical inputs')
        callbacks = {}
        for index,identifier in enumerate(sample_ids,1):
            self.check_release()
            self.status('binding_potential',candidate_digest=candidate.digest,
                        current_system=identifier,bound_systems=index-1,total_systems=len(sample_ids))
            case = self.cases[identifier]
            bound = case.bind(candidate, complete_potential=True) if self.complete_potential else case.bind(candidate)
            if self.objective == 'wy_energy':
                callback = bound[identifier]
                bound[identifier] = replace(callback, sample_details=partial(case.energy_report, candidate))
            callbacks.update(bound)
        self.status('fitting',candidate_digest=candidate.digest,bound_systems=len(sample_ids))
        return callbacks

    def compose(self, result):
        """Outer selection uses fitted densities, preserving raw physical scores."""
        if self.objective=='wu_yang' or not result.valid:
            return result
        if self.objective == 'wy_energy':
            return self._compose_energy(result)
        identifiers = tuple(result.samples)
        electrons = {r['system_id']:r['nelec'] for r in self.rows}
        gap = np.array([result.samples[s]['variational_gap']/electrons[s] for s in identifiers])
        l1 = np.array([result.samples[s]['density_l1'] for s in identifiers])
        l2 = np.array([result.samples[s]['density_error']**2 for s in identifiers])
        values = np.asarray(1-wu_yang_density_loss(gap,l1,l2,scales=self.normalization))
        if not np.isfinite(values).all():
            raise ValueError('Composite density reward is nonfinite')
        samples = {s:{**result.samples[s],'composite_reward':float(v)} for s,v in zip(identifiers,values)}
        fit = {**result.fit,'inner_objective':'wu_yang','outer_objective':'wy_l1_l2',
            'outer_components':dict(zip(('gap_ratio','l1_ratio','l2_squared_ratio'),
                [float(np.mean(x)/scale) for x,scale in zip((gap,l1,l2),self.normalization)]))}
        return Evaluation(float(np.mean(values)),True,result.parameters,samples,fit,result.error)

    def _compose_energy(self, result):
        rows = {r['system_id']: r for r in self.rows}
        samples = {}
        wy_ratios, energy_errors = [], []
        wy_weight, energy_weight = self.policy['outer_weights']
        for identifier, sample in result.samples.items():
            baseline = rows[identifier]
            nelec = baseline['nelec']
            reference = baseline['reference_xc_energy_hartree']
            error = sample['xc_energy_hartree'] - reference
            error_per_electron = abs(error)/nelec
            wy_ratio = sample['delta']/(nelec*self.normalization[0])
            reward = wy_weight*wy_ratio + energy_weight*(1 - error_per_electron/self.energy_normalization)
            if not np.isfinite(reward):
                raise ValueError(f'Joint reward is nonfinite for {identifier}')
            samples[identifier] = dict(sample, reference_xc_energy_hartree=reference,
                xc_energy_error_hartree=error, xc_energy_absolute_error_hartree_per_electron=error_per_electron,
                composite_reward=reward)
            wy_ratios.append(wy_ratio)
            energy_errors.append(error_per_electron)
        fit = dict(result.fit, inner_objective='wu_yang', outer_objective='wy_energy',
            outer_components={'wu_yang_improvement_ratio': float(np.mean(wy_ratios)),
                'xc_energy_mae_hartree_per_electron': float(np.mean(energy_errors)),
                'xc_energy_error_ratio': float(np.mean(energy_errors)/self.energy_normalization)})
        reward = float(np.mean([sample['composite_reward'] for sample in samples.values()]))
        return Evaluation(reward, True, result.parameters, samples, fit, result.error)

    def evaluate(self, candidate, *, maxiter=None):
        start = time.monotonic()
        try:
            if self.candidate_mode=='legacy':validate_candidate(candidate)
            options = dict(self.fit_options)
            if maxiter is not None: options['maxiter'] = maxiter
            evaluator = Evaluator(self.bind, sample_ids=IDS, fingerprint=self.fingerprint,**options)
            result = evaluator.evaluate(candidate)
            fit = {**result.fit, 'evaluation_seconds': time.monotonic()-start,
                   'cached_affine': self.cached_affine(candidate),
                   'field_seconds':(None if self.open_design else
                       sum(case.timings.get('last_potential_fields_seconds',0.) for case in self.cases.values())),
                   'projection_seconds':(None if self.open_design else
                       sum(case.timings.get('last_potential_projection_seconds',0.) for case in self.cases.values())),
                   'active_bounds': [p.name for p in candidate.parameters if p.name in result.parameters and
                      min(abs(result.parameters[p.name]-p.lower), abs(result.parameters[p.name]-p.upper)) < 1e-8],
                   **memory()}
            return self.compose(Evaluation(result.reward, result.valid, result.parameters, result.samples, fit, result.error))
        except (ValueError, KeyError, TypeError, FloatingPointError) as exc:
            return Evaluation(None, False, candidate.initial, fit={'success': False,
                'evaluation_seconds': time.monotonic()-start}, error=f'{type(exc).__name__}: {exc}')

    def cached_affine(self, candidate):
        if self.open_design:
            return all(case.timings.get('cached_affine', False) for case in self.cases.values())
        return bool(candidate.affine)

    def serve(self):
        self.prepare()
        idle = time.monotonic()
        while not (self.root/'release.flag').exists():
            if not self.unbounded_waits and time.monotonic()-idle > 3600:
                self.status('idle_timeout')
                return
            pending = sorted((self.root/'inbox').glob('*.json'))
            for path in pending:
                target = self.root/'outbox'/path.name
                if target.exists(): continue
                idle = time.monotonic()
                request = json.loads(path.read_text())
                unchanged = all(file_hash(Path(__file__).with_name(name))==value for name,value in self.helpers.items())
                if request['fingerprint'] != self.fingerprint or source_digest() != self.core or not unchanged:
                    raise ValueError('Numerical source/data fingerprint changed')
                candidate = Candidate.from_json(request['candidate'])
                if path.stem != candidate.digest:
                    raise ValueError('Candidate identity differs from the request filename')
                self.status('evaluating', candidate_digest=candidate.digest)
                result = self.evaluate(candidate)
                write_json(target, {'fingerprint': self.fingerprint, 'digest': candidate.digest,
                                   'evaluation': result, 'seconds': time.monotonic()-idle})
                print('Candidate', candidate.digest[:12], result.reward, result.valid, flush=True)
                self.status('ready', last_candidate=candidate.digest)
            time.sleep(2)
        self.status('released')


def serve(root):
    path = Path(root)/'campaign.json'
    options = json.loads(path.read_text()) if path.exists() else {}
    campaign = Campaign(root,**options)
    try:
        campaign.serve()
    except CampaignReleased:
        campaign.status('released')
    except BaseException as exc:
        campaign.status('failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
