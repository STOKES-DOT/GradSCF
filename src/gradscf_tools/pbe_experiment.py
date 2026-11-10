"""Assessment plumbing for externally designed affine GGA corrections.

A pointwise callback returns energy density (Ha/bohr^3), with rho_spin[2,g]
and grad_spin[2,g,3] in atomic units. Molecular E and V always come from the
same GradSCF Functional; stress checks sample the callback, not a fitted table.
"""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf import dft
from .pbe_loop import fit_affine, summarize_affine
from .pbe_recording import compute_reward


CONSTRAINT_NAMES = ('finite_sampled_derivatives', 'spin_interchange_symmetry',
    'uniform_density_limit_unchanged', 'finite_fixed_reference_response')


def semilocal_functional(pointwise, *, name):
    """Bind an external energy-density formula to SCF-current spin features."""
    return dft.Functional(
        input_fn=lambda fields:(fields.rho_spin,fields.grad_rho_spin,fields.weights),
        energy_fn=lambda params,inputs:jnp.sum(inputs[2]*pointwise(params,inputs[0],inputs[1])),
        name=name)


def semilocal_checks(pointwise, theta):
    """Finite sample checks, not a proof of global exact constraints.

    Freeze rho=10^-6..10^2, s=0..5, zeta=-.8,0,.8 before model proposals.
    Both first derivatives and a Hessian-vector product must remain finite.
    """
    n,s,z = np.meshgrid(np.logspace(-6,2,9),np.linspace(0,5,6),[-.8,0,.8],indexing='ij')
    n,s,z = (jnp.asarray(v.ravel()) for v in (n,s,z))
    fractions = jnp.stack([.5*(1+z),.5*(1-z)])
    rho = fractions*n
    magnitude = 2*(3*jnp.pi**2)**(1/3)*n**(4/3)*s
    direction = jnp.array([1.,0.,0.])
    grad = fractions[:,:,None]*magnitude[None,:,None]*direction
    value = pointwise(theta,rho,grad)
    swap = pointwise(theta,rho[::-1],grad[::-1])
    uniform = pointwise(theta,rho,jnp.zeros_like(grad))
    scalar = lambda r,g:jnp.sum(pointwise(theta,r,g))
    derivative = jax.grad(scalar,argnums=(0,1))
    first,hvp = jax.jvp(derivative,(rho,grad),(jnp.ones_like(rho),jnp.ones_like(grad)))
    finite = all(bool(jnp.all(jnp.isfinite(v))) for v in (value,swap,uniform,*first,*hvp))
    swap_error = float(jnp.max(jnp.abs(value-swap)))
    uniform_error = float(jnp.max(jnp.abs(uniform)))
    measures = dict(sample_count=int(n.size),spin_swap_max_abs=swap_error,
        uniform_density_max_abs=uniform_error,energy_density_units='Ha/bohr^3',
        density_range_bohr_minus3=[1e-6,1e2],reduced_gradient_range=[0.,5.])
    checks = dict(finite_sampled_derivatives=finite,
        spin_interchange_symmetry=bool(finite and swap_error <= 1e-10),
        uniform_density_limit_unchanged=bool(finite and uniform_error <= 1e-10))
    return checks,measures


def fit_and_score(caches, *, initial, reward_config, bounds=None, pointwise=None,
                  fixed_parameters=None, maxiter=300):
    """Fit only training; sealed test caches are forbidden at this stage."""
    if set(caches) != {'train','validation'} or not all(caches.values()):
        raise ValueError('Supply only nonempty train and validation caches; test is sealed')
    if fixed_parameters is None:
        fitted = fit_affine(caches['train'],initial,bounds=bounds,maxiter=maxiter)
        theta = fitted['fitted_parameters']
        opt = fitted['optimizer']
        optimizer = dict(success=bool(opt.success),message=str(opt.message),
            iterations=int(opt.nit),evaluations=int(opt.nfev),loss=float(opt.fun),
            gradient_max=float(np.max(np.abs(opt.jac))))
    else:
        theta = list(fixed_parameters)
        optimizer = dict(success=None,message='Fixed baseline; no fit',iterations=0,evaluations=0)
    metrics = {role:summarize_affine(values,theta) for role,values in caches.items()}
    checks,measures = ({}, {}) if pointwise is None else semilocal_checks(pointwise,jnp.asarray(theta))
    checks['finite_fixed_reference_response'] = all(v['valid'] for v in metrics.values())
    reward = compute_reward(metrics['validation']['energy_wrmsd_hartree'],reward_config,checks)
    predictions=[]
    for role,values in caches.items():
        for cache in values:
            value=cache.evaluate(theta)
            predictions.append(dict(molecule_id=cache.teacher.provenance.get('molecule_id','array_oracle'),
                split=role,reference=float(cache.teacher.target.energy),prediction=float(value.total_energy),
                signed_error=float(value.energy_error),absolute_error=abs(float(value.energy_error)),units='Hartree'))
    return dict(parameters=theta,optimizer=optimizer,metrics=metrics,reward=reward,
        constraints=checks,constraint_measurements=measures,predictions=predictions)
