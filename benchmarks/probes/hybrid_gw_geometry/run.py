"""Bounded hybrid/RSH reference and geometry probes; numerical kernels stay in src."""
import argparse
import hashlib
import importlib.metadata
from inspect import getfile
import json
import os
from pathlib import Path
import resource
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=('forward', 'geometry'))
    p.add_argument('output', type=Path)
    p.add_argument('--xc', choices=('svwn', 'pbe', 'pbe0', 'wb97xd'), required=True)
    p.add_argument('--method', choices=('dft', 'g0w0', 'evgw', 'evgw0', 'qsgw', 'scgw'), default='dft')
    p.add_argument('--backend', choices=('full', 'direct', 'df'), default='full')
    p.add_argument('--molecule', choices=('h2', 'water', 'ch2o'), default='h2')
    p.add_argument('--basis', default='sto-3g')
    p.add_argument('--case-input', type=Path)
    p.add_argument('--grid', type=int, default=0)
    p.add_argument('--grid-block-size', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--screening', type=float, default=0.)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    p.add_argument('--hvp', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--restart', type=Path)
    p.add_argument('--finite-difference', action='store_true')
    p.add_argument('--ri-seed', action='store_true')
    p.add_argument('--forward', choices=('minimize','scf','reference'), default='minimize')
    p.add_argument('--reference-state', type=Path)
    p.add_argument('--gradient-record', type=Path)
    p.add_argument('--fd-step', type=float, choices=(3e-4, 1e-4))
    args = p.parse_args()
    if args.device == 'gpu' and (args.task != 'forward' or args.method not in ('g0w0','evgw','evgw0')):
        p.error('GPU probing stages CPU SCF/RI into diagonal GW; geometry and matrix methods use CPU.')
    if args.backend == 'df' and (args.task == 'geometry' or args.xc == 'wb97xd'):
        p.error('Auxiliary RI geometry and RI-JK RSH are not implemented; use direct or full.')
    if args.finite_difference and (args.task != 'geometry' or args.hvp):
        p.error('Directional finite differences validate first geometry derivatives.')
    if args.ri_seed and (args.restart is not None or args.backend != 'direct'):
        p.error('The RI seed experiment starts a direct calculation without a restart file.')
    if args.reference_state and (args.task != 'geometry' or args.method != 'dft'
            or args.backend != 'direct' or args.case_input is None or args.restart or args.ri_seed):
        p.error('A saved reference explicitly selects shared-input direct DFT geometry, with a case file.')
    if args.gradient_record and (not args.reference_state or args.hvp or args.finite_difference):
        p.error('FD-only uses a saved reference and an accepted gradient record; no HVP or repeated AD.')
    if args.forward == 'reference' and (args.task != 'geometry' or args.method != 'dft'
            or not args.reference_state or args.finite_difference or args.gradient_record):
        p.error('Reference mode is a local DFT response stage; FD-only uses forward=scf with fresh solves.')
    allowed = sorted(os.sched_getaffinity(0))
    if args.threads > len(allowed):
        raise ValueError('Requested CPUs exceed the Slurm affinity')
    os.sched_setaffinity(0, allowed[:args.threads])
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import numpy as np
    import gradscf
    from gradscf import dft, gto, gw
    from gradscf.integrals._native import register_integrals
    from gradscf_tools.molecular_response import MolecularEnergy
    from gradscf_tools.gw_response import MolecularGW
    from gradscf.scf import orbital_optimization, rks as rks_module
    from gradscf import _memory as memory_module
    from gradscf.integrals import grids as grid_module
    cpu = jax.devices('cpu')[0]
    target = jax.devices(args.device)[0]
    jax.config.update('jax_default_device', cpu)
    atoms = {'h2': 'H 0 0 -.370424; H 0 0 .370424',
        'water': 'O 0 0 .117790; H 0 .755453 -.471161; H 0 -.755453 -.471161',
        'ch2o': 'C 0 0 0; O 0 0 1.208; H 0 .936 -.584; H 0 -.936 -.584'}
    case=json.loads(args.case_input.read_text()) if args.case_input else {}
    nw=int(case.get('nw',32))
    atom=case.get('atom',atoms[args.molecule])
    args.basis=case.get('basis',args.basis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    library = Path(register_integrals()._name).resolve()
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key,item in value.items()}
        if isinstance(value, (list,tuple)):
            return [json_safe(item) for item in value]
        return None if isinstance(value,float) and not np.isfinite(value) else value
    def record(**row):
        row.setdefault('device', str(target))
        row.update(xc=args.xc, method=args.method, task=args.task, backend=args.backend,
            basis=args.basis, grid_level=args.grid, molecule=case.get('name',args.molecule),
            case_id=case.get('case_id',args.molecule), dtype='float64',
            nw=16 if args.task=='geometry' else nw,
            orbital_forward=args.forward if args.task=='geometry' else None,
            timing_scope='shared_reference_geometry' if args.reference_state else 'independent_workflow',
            affinity=sorted(os.sched_getaffinity(0)), units='Ha, bohr, s, bytes')
        with args.output.open('a') as stream:
            stream.write(json.dumps(json_safe(row), allow_nan=False)+'\n')
    manifest = dict(arguments={k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, atom=atom,
        unit='Angstrom', charge=0, spin=0, cart=True,
        source=gradscf.__file__, python=os.sys.executable,
        orbital_source=str(Path(orbital_optimization.__file__).resolve()),
        orbital_sha256=hashlib.sha256(Path(orbital_optimization.__file__).read_bytes()).hexdigest(),
        rks_source=str(Path(rks_module.__file__).resolve()),
        rks_sha256=hashlib.sha256(Path(rks_module.__file__).read_bytes()).hexdigest(),
        memory_source=str(Path(memory_module.__file__).resolve()),
        memory_sha256=hashlib.sha256(Path(memory_module.__file__).read_bytes()).hexdigest(),
        grid_source=str(Path(grid_module.__file__).resolve()),
        grid_sha256=hashlib.sha256(Path(grid_module.__file__).read_bytes()).hexdigest(),
        versions={x: importlib.metadata.version(x) for x in ('jax','jaxlib','numpy','scipy')},
        native_path=str(library), native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        restart_sha256=None if args.restart is None else hashlib.sha256(args.restart.read_bytes()).hexdigest(),
        reference_sha256=None if args.reference_state is None else hashlib.sha256(args.reference_state.read_bytes()).hexdigest(),
        reference_metadata_sha256=None if args.reference_state is None else hashlib.sha256(args.reference_state.with_suffix('.json').read_bytes()).hexdigest(),
        tools={x.__name__: hashlib.sha256(Path(getfile(x)).read_bytes()).hexdigest()
               for x in (MolecularEnergy,MolecularGW)},
        threads={k:os.getenv(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','GRADSCF_NATIVE_THREADS')},
        cache_environment={k:os.getenv(k) for k in ('JAX_ENABLE_COMPILATION_CACHE','JAX_COMPILATION_CACHE_DIR')})
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def reference():
        mol = gto.M(atom=atom, basis=args.basis, cart=True, unit='Angstrom')
        mf = dft.RKS(mol, xc=args.xc, grids_level=args.grid, conv_tol=1e-12,
            conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=200,
            grid_block_size=args.grid_block_size)
        if args.ri_seed:
            tick=time.perf_counter()
            seed_xc='pbe0' if args.xc=='wb97xd' else args.xc
            seed=dft.RKS(mol, xc=seed_xc, grids_level=args.grid, conv_tol=1e-12,
                conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=200,
                grid_block_size=args.grid_block_size)
            seed=seed.density_fit(case.get('auxbasis','weigend')).run()
            jax.block_until_ready(seed.scf_result)
            record(stage='ri_seed', device=str(cpu), wall_seconds=time.perf_counter()-tick,
                seed_xc=seed_xc, cycles=int(seed.scf_result.cycles), scientific_pass=bool(seed.converged))
            if not seed.converged:
                raise ArithmeticError('RI starting density did not converge')
            mf.init_guess=seed.make_rdm1()
        if args.restart is not None:
            with np.load(args.restart) as saved:
                mf.init_guess = np.array(saved['density'])
        if args.backend == 'direct':
            mf.direct_scf_workers, mf.direct_scf_tol = args.threads, args.screening
            mf = mf.direct_scf()
        elif args.backend == 'df':
            mf = mf.density_fit(case.get('auxbasis','weigend'))
        return mf.run()
    if args.task == 'forward':
        for repeat in range(args.repeats+1):
            start = time.perf_counter()
            mf = reference()
            jax.block_until_ready(mf.scf_result)
            scf_seconds = time.perf_counter()-start
            fit_seconds = solve_seconds = 0.
            output = dict(energy_hartree=float(mf.e_tot))
            passed = bool(mf.converged)
            record(stage='reference', repeat=repeat, wall_seconds=scf_seconds,
                nao=mf.mo_coeff.shape[0], cycles=int(mf.scf_result.cycles),
                ngrid=int(mf._scf_inputs.grid_weights.size) or None,
                grid_mode='streamed' if mf._scf_inputs.grid_ao is not None else 'dense',
                payload_ao_array_bytes=getattr(mf._scf_inputs.ao,'nbytes',None),
                payload_ao_deriv1_array_bytes=getattr(mf._scf_inputs.ao_deriv1,'nbytes',None),
                scientific_pass=passed, device=str(cpu), **output)
            if args.method != 'dft':
                targets = {'orbs': tuple(case['orbs'])} if args.method == 'g0w0' and 'orbs' in case else {}
                model = gw.GW(mf, method=args.method, nw=nw, max_cycle=100,
                    conv_tol=1e-7, damp=.3, beta=20., mixing=.5)
                if args.backend != 'df':
                    model = model.density_fit(case.get('auxbasis','weigend'))
                tick=time.perf_counter(); factors=model._df_factors(); jax.block_until_ready(factors); fit_seconds=time.perf_counter()-tick
                record(stage='ri_preparation', repeat=repeat, wall_seconds=fit_seconds, device=str(cpu))
                if args.device == 'cpu':
                    tick=time.perf_counter(); model.run(**targets); result=jax.block_until_ready(model.result); solve_seconds=time.perf_counter()-tick
                else:
                    from gradscf.gw.integrals import starting_potential
                    state=mf.scf_result
                    common=dict(mo_energy=state.mo_energy, mo_coeff=state.mo_coeff,
                        df_factors=factors, hcore_matrix=state.hcore_matrix)
                    tick=time.perf_counter()
                    if args.method in ('g0w0','evgw','evgw0'):
                        common.update(fock_matrix=state.fock_matrix, density_matrix=state.density_matrix,
                            v_mf_matrix=starting_potential(mf, state.density_matrix,
                                state.fock_matrix, state.hcore_matrix, max_memory=60000.))
                    else:
                        raise NotImplementedError('This large-system GPU probe currently targets diagonal GW.')
                    jax.block_until_ready(common)
                    record(stage='starting_potential', repeat=repeat, wall_seconds=time.perf_counter()-tick, device=str(cpu))
                    transfer_bytes=sum(x.nbytes for x in jax.tree.leaves(common))
                    tick=time.perf_counter(); common=jax.block_until_ready(jax.device_put(common,target))
                    record(stage='transfer', repeat=repeat, wall_seconds=time.perf_counter()-tick,
                        transfer_bytes=transfer_bytes, device=f'{cpu}->{target}')
                    options=dict(nocc=int(np.count_nonzero(mf.mo_occ)), nw=nw, eta=1e-3, max_memory=60000.)
                    if args.method == 'g0w0':
                        from gradscf.gw.g0w0 import _run_g0w0_numerics as driver
                        options.update(qp_tol=1e-10, resolvent_expansion=False, **targets)
                    else:
                        from gradscf.gw.evgw import evgw_cd_restricted as driver
                        options.update(max_iter=100, tol=1e-7, damping=.3, update_w=args.method=='evgw')
                    with jax.default_device(target):
                        tick=time.perf_counter(); result=jax.block_until_ready(driver(**common,**options)); solve_seconds=time.perf_counter()-tick
                passed &= bool(result.converged)
                if args.method == 'scgw':
                    output.update(energy_hartree=float(result.total_energy),
                        particle_error=float(result.particle_number_error), residual=float(result.fixed_point_residual))
                else:
                    requested=np.flatnonzero(np.asarray(result.qp_computed_mask))
                    output.update(qp_targets=requested.tolist(),
                        qp_energy_hartree=np.asarray(result.mo_energy)[requested].tolist())
                    if args.method != 'qsgw':
                        output.update(qp_residual_hartree=np.asarray(result.qp_residual)[requested].tolist(),
                            qp_weight=np.asarray(result.qp_weight)[requested].tolist(),
                            qp_branch_valid=np.asarray(result.qp_branch_valid)[requested].tolist())
                        passed &= bool(np.asarray(result.qp_branch_valid)[requested].all())
            if args.checkpoint is not None:
                inputs = mf._scf_inputs
                np.savez(args.checkpoint, mo_coeff=np.asarray(mf.mo_coeff),
                    mo_energy=np.asarray(mf.mo_energy), mo_occ=np.asarray(mf.mo_occ),
                    overlap=np.asarray(inputs.overlap), hcore=np.asarray(mf.scf_result.hcore_matrix),
                    fock=np.asarray(mf.scf_result.fock_matrix),
                    density=np.asarray(mf.scf_result.density_matrix))
            record(stage='end_to_end', repeat=repeat, wall_seconds=time.perf_counter()-start,
                scf_seconds=scf_seconds, ri_seconds=fit_seconds, gw_seconds=solve_seconds,
                nao=mf.mo_coeff.shape[0], scientific_pass=passed, **output)
    else:
        tick=time.perf_counter()
        if args.reference_state:
            metadata=json.loads(args.reference_state.with_suffix('.json').read_text())
            stage=metadata['zero_screening_scf']
            assert metadata['scf_stage_complete'] is True
            assert (stage['tolerance_energy'],stage['tolerance_density'],stage['tolerance_gradient']) == (1e-12,1e-11,1e-10)
            expected=(args.xc,args.basis.lower(),args.grid,args.grid_block_size,args.screening)
            actual=(metadata['xc'],metadata['basis'].lower(),metadata['grid_level'],
                    metadata['grid_block_size'],stage['direct_cutoff'])
            assert actual == expected and stage['converged'] and stage['residual_norm'] <= 1e-10
            assert metadata['checkpoint_sha256'] == manifest['reference_sha256']
            assert metadata['input_sha256']['case'] == hashlib.sha256(args.case_input.read_bytes()).hexdigest()
            assert metadata['native_sha256'] == manifest['native_sha256']
            mf=dft.RKS(gto.M(atom=atom,basis=args.basis,cart=True,unit='Angstrom'),
                xc=args.xc,grids_level=args.grid,conv_tol=1e-12,conv_tol_density=1e-11,
                conv_tol_grad=1e-10,max_cycle=200,grid_block_size=args.grid_block_size).direct_scf()
            mf.direct_scf_workers=args.threads
            with np.load(args.reference_state,allow_pickle=False) as saved:
                mf.mo_coeff,mf.mo_occ,mf.mo_energy=(jnp.asarray(saved[key]) for key in ('mo_coeff','mo_occ','mo_energy'))
            mf.converged,mf.e_tot=True,stage['energy']
            record(stage='reference_reuse',wall_seconds=time.perf_counter()-tick,
                prepared_reference_seconds=stage['seconds'],reference_sha256=manifest['reference_sha256'],
                energy_hartree=float(mf.e_tot),scientific_pass=True,
                notes=('Own completed SCF; rebuild live geometry inputs and attach full local implicit response.'
                       if args.forward=='reference' else
                       'Own completed SCF stage; current geometry inputs are rebuilt and every model call relaxes SCF.'))
        else:
            mf=reference(); jax.block_until_ready(mf.scf_result)
            record(stage='reference',wall_seconds=time.perf_counter()-tick,energy_hartree=float(mf.e_tot))
        if args.method == 'dft':
            model=MolecularEnergy(mf, jk_backend=args.backend, max_memory=60000., gradient_tolerance=1e-10,
                forward=args.forward)
            observable=model
        else:
            options = {} if args.method == 'scgw' else {'eta': 1e-5}
            model=MolecularGW(mf, method=args.method, nw=16,
                jk_backend=args.backend, max_memory=60000., forward=args.forward, **options)
            observable=(lambda r: model(r).total_energy) if args.method=='scgw' else (lambda r:model(r).mo_energy[model.nocc-1])
        coordinates=model.coordinates
        direction=jnp.sin(jnp.arange(coordinates.size,dtype=jnp.float64)+1).reshape(coordinates.shape)
        direction=direction/jnp.linalg.norm(direction)
        if args.gradient_record:
            previous=json.loads(args.gradient_record.with_suffix('.manifest.json').read_text())
            for key in ('tools','orbital_sha256','rks_sha256','grid_sha256','native_sha256','reference_sha256'):
                assert previous[key] == manifest[key], key
            rows=[json.loads(line) for line in args.gradient_record.read_text().splitlines()]
            accepted=next(row for row in reversed(rows) if row.get('stage')=='derivative' and row['scientific_pass'])
            for key,value in (('xc',args.xc),('basis',args.basis),('grid_level',args.grid)):
                assert accepted[key] == value, key
            assert (accepted['orbital_forward'] == args.forward or
                    (accepted['orbital_forward']=='reference' and args.forward=='scf'))
            result=tuple(jnp.asarray(value) for value in accepted['values'])
            assert result[1].shape == coordinates.shape
            record(stage='gradient_reuse',source=str(args.gradient_record),
                source_sha256=hashlib.sha256(args.gradient_record.read_bytes()).hexdigest())
        else:
            if args.forward == 'reference':
                scalar=lambda r,c: model(r,solution=c)
                function=(jax.jit(lambda r,c,v: jax.jvp(jax.grad(lambda x: scalar(x,c)),
                    (r,), (v,))[1]) if args.hvp else jax.jit(jax.value_and_grad(scalar,argnums=0)))
                inputs=(coordinates,model.coefficients,direction) if args.hvp else (coordinates,model.coefficients)
            else:
                function=jax.jit(lambda r,v:jax.jvp(jax.grad(observable),(r,),(v,))[1]) if args.hvp else jax.jit(jax.value_and_grad(observable))
                inputs=(coordinates,direction) if args.hvp else (coordinates,)
            tick=time.perf_counter(); lowered=function.lower(*inputs); lower_seconds=time.perf_counter()-tick
            record(stage='lower',wall_seconds=lower_seconds,
                derivative='hvp' if args.hvp else 'value_and_grad')
            tick=time.perf_counter(); executable=lowered.compile(); compile_seconds=time.perf_counter()-tick
            memory=executable.memory_analysis()
            record(stage='compile', lower_seconds=lower_seconds, compile_seconds=compile_seconds,
                xla_temp_bytes=getattr(memory,'temp_size_in_bytes',None), derivative='hvp' if args.hvp else 'value_and_grad')
            for repeat in range(args.repeats+1):
                tick=time.perf_counter(); result=jax.block_until_ready(executable(*inputs)); seconds=time.perf_counter()-tick
                leaves=jax.tree.leaves(result)
                error=abs(float(result[0])-float(mf.e_tot)) if args.method=='dft' and not args.hvp else None
                record(stage='derivative',repeat=repeat,wall_seconds=seconds,
                    energy_reference_error_hartree=error,values=[np.asarray(x).tolist() for x in leaves],
                    scientific_pass=bool(all(np.isfinite(x).all() for x in leaves) and (error is None or error <= 1e-7)))
        if args.finite_difference or args.gradient_record:
            tick=time.perf_counter(); evaluate=jax.jit(observable).lower(coordinates).compile()
            record(stage='fd_compile', wall_seconds=time.perf_counter()-tick)
            automatic=float(jnp.vdot(result[1], direction))
            for step in ((args.fd_step,) if args.fd_step is not None else (3e-4,1e-4)):
                tick=time.perf_counter()
                plus=jax.block_until_ready(evaluate(coordinates+step*direction))
                minus=jax.block_until_ready(evaluate(coordinates-step*direction))
                finite=float((plus-minus)/(2*step)); error=abs(automatic-finite)
                record(stage='directional_fd', step_bohr=step,
                    wall_seconds=time.perf_counter()-tick, automatic=automatic,
                    plus_energy_hartree=float(plus),minus_energy_hartree=float(minus),
                    finite_difference=finite, absolute_error=error,
                    scientific_pass=bool(np.isfinite(finite) and error <= 3e-6+4e-5*abs(finite)))
    record(stage='process_peak',cpu_peak_rss_bytes=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gpu_allocator_statistics=target.memory_stats() if args.device=='gpu' else None,
        memory_scope='ru_maxrss: whole process, including imports/preparation/compilation; GPU allocator counters separate')


if __name__=='__main__':
    main()
