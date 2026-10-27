"""Small fully relaxed excitation derivative, or NNAO native projection.

The geometry fixture rebuilds atom-centered grids, integrals, SCF and response;
it is not a frozen-orbital derivative. The NNAO fixture is a projection-product
test, and is deliberately not labeled a complete neural excitation training.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
from worker import Recorder

jax.config.update("jax_enable_x64", True)


def geometry(cfg, log):
    root=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(root/"tests"))
    from test_direct_excitation_geometry import _excitation, _global_network
    from gradscf.data.molecule import parse_molecule_spec
    spec=parse_molecule_spec("H 0 0 0; H 0 0 1.6",unit="Bohr")
    model,params=_global_network()
    xc=cfg.get("xc_override","network" if cfg.get("neural") else "pbe")
    # Same exact, un-screened direct J/K and level-0 moving grid as the
    # existing regression fixture. No new solver or stopped dependency.
    function=jax.jit(lambda bond:_excitation(spec,bond,False,xc,cfg["full"],cfg.get('direct',True),model,params))
    scalar=lambda bond:function(bond)[0]
    point=jnp.asarray(1.6)
    log.context.update(nao=2,nocc=1,nvir=1,nroots=1)
    for repeat in range(4):
        value=log.timed("excitation_value",lambda:function(point),repeat)
    if not bool(value[1]):raise RuntimeError("SCF/TD did not converge")
    vg=jax.jit(jax.value_and_grad(scalar))
    for repeat in range(4):
        energy,derivative=log.timed("excitation_value_and_grad",lambda:vg(point),repeat)
    _,tangent=log.timed("geometry_jvp",lambda:jax.jvp(scalar,(point,),(jnp.ones_like(point),)))
    for step in (2e-4,1e-4):
        plus=log.timed("fd_reconverged_plus",lambda:function(point+step))
        minus=log.timed("fd_reconverged_minus",lambda:function(point-step))
        fd=(plus[0]-minus[0])/(2*step)
        log.add("geometry_validation",None,gradient_error=float(abs(fd-derivative)),fd_step_bohr=step,
                derivative_ha_per_bohr=float(derivative),jvp_vjp_error=float(abs(tangent-derivative)),
                excitation_ha=float(energy),density_change=float(jnp.linalg.norm(plus[2]-minus[2])),
                notes="full integral/grid/orbital/XC/SCF response; nondegenerate H2 STO-3G; not def2-SVP scaling")
        assert abs(fd-derivative)<3e-6 and abs(tangent-derivative)<3e-8


def nnao(cfg,log):
    from gradscf.model.nnao import prepare_direct_basis
    from gradscf.model.nnao.integrals.context import prepare_native_context
    from gradscf.model.nnao.integrals.variable_exponents import native_projection
    atom=[('H',(0.,0.,0.)),('H',(0.,0.,.74))]
    layout=prepare_direct_basis(atom,cart=False,basis_family='szp663_direct')
    context=log.timed("nnao_context",lambda:prepare_native_context(layout,atom,'def2-universal-jkfit'))
    forward,backward=native_projection(context,layout.exponent_scale_keys)
    coefficients=layout.reference_outputs()
    beta=jnp.array([.03,-.02])
    dc=.02*jnp.sin(jnp.arange(coefficients.size).reshape(coefficients.shape)+.4)
    db=jnp.array([.07,-.04])
    for repeat in range(4):
        out=log.timed("nnao_projection",lambda:forward(coefficients,beta),repeat)
    cot=tuple(.01*jnp.cos(jnp.arange(a.size).reshape(a.shape)) for a in out)
    for repeat in range(4):
        co,cb=log.timed("nnao_basis_vjp",lambda:backward(coefficients,beta,cot),repeat)
    _,tangent=log.timed("nnao_basis_jvp",lambda:jax.jvp(forward,(coefficients,beta),(dc,db)))
    lhs=sum(jnp.vdot(a,b) for a,b in zip(tangent,cot)); rhs=jnp.vdot(co,dc)+jnp.vdot(cb,db)
    def scalar(c,b):return sum(jnp.vdot(a,t) for a,t in zip(forward(c,b),cot))
    for h in (2e-4,1e-4):
        fd=(scalar(coefficients+h*dc,beta+h*db)-scalar(coefficients-h*dc,beta-h*db))/(2*h)
        log.add("nnao_validation",None,gradient_error=float(abs(lhs-rhs)),finite_difference_error=float(abs(lhs-fd)),
                step=h,ncontracted=layout.topology.nao,naux=context['aux'].auxiliary.nao,
                notes="synthetic NNAO coefficients+log exponents; libcint 2c/3c/whitening; no trained checkpoint")
        assert abs(lhs-rhs)<1e-9 and abs(lhs-fd)<2e-7


if __name__ == '__main__':
    cfg=json.loads(Path(sys.argv[1]).read_text()); out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    log=Recorder(cfg,out)
    if cfg.get('experiment'):
        from experiments import install
        install(cfg)
    (nnao if cfg.get('probe')=='nnao' else geometry)(cfg,log)
