"""External analytic GGA proposals for the small fixed-reference iKS pilot.

The first proposal was produced by GPT-5.6 Luna/high from public literature.
This module implements its formula; GradSCF supplies the shared Functional AD.
No integral, inverse, response or optimization algorithm is implemented here.
"""
import jax.numpy as jnp

from gradscf_tools.pbe_loop import AffineTerm
from gradscf_tools.pbe_experiment import semilocal_functional


C001_FORMULA = r"""# c001: bounded spin-symmetric gradient deformation of PBE

All quantities are in atomic units. Let n=n_alpha+n_beta, g=grad(n),
n_v=1e-12 bohr^-3, r=sqrt(n^2+n_v^2), n_plus=(n+r)/2,
w=n^2/(n^2+n_v^2), s2=|g|^2/[4(3*pi^2)^(2/3)*r^(8/3)],
q=s2/(1+s2), z=(n_alpha-n_beta)/r, and
h=-0.7385587663820223*n_plus^(4/3)*w (Ha/bohr^3).

E_xc = E_xc(PBE) + integral h*(theta1*q+theta2*q^2+theta3*q*z^2) dr.
Each theta is dimensionless, initially0, bounded in[-0.5,0.5].
At theta=0 this is exactly unmodified PBE. All corrections vanish for g=0
and are invariant under alpha/beta interchange. The fixed vacuum scale breaks
strict coordinate scaling in the tail; exact exchange spin scaling and a global
Lieb-Oxford bound are not enforced. This is an unvalidated pilot proposal.
On restricted closed-shell references z=0: theta3 is unidentifiable and remains
at its initial value; a finite optimizer result does not validate that channel.

Proposal references:
- https://doi.org/10.1103/PhysRevLett.77.3865
- https://arxiv.org/html/2605.05460
- https://arxiv.org/html/2610.30022
"""


def _gradient_descriptors(rho_spin, grad_spin):
    n=rho_spin.sum(axis=0)
    grad=grad_spin.sum(axis=0)
    vacuum=1e-12
    regularized=jnp.sqrt(n*n+vacuum*vacuum)
    positive=.5*(n+regularized)
    gate=n*n/(n*n+vacuum*vacuum)
    s2=jnp.sum(grad*grad,axis=-1)/(4*(3*jnp.pi*jnp.pi)**(2/3)*regularized**(8/3))
    q=s2/(1+s2)
    z=(rho_spin[0]-rho_spin[1])/regularized
    scale=-.7385587663820223*positive**(4/3)*gate
    return scale,q,z


def term_c001_gradient(rho_spin, grad_spin):
    scale,q,z=_gradient_descriptors(rho_spin,grad_spin)
    return scale*q


def term_quartic_gradient(rho_spin, grad_spin):
    scale,q,z=_gradient_descriptors(rho_spin,grad_spin)
    return scale*q*q


def term_c001_spin(rho_spin, grad_spin):
    scale,q,z=_gradient_descriptors(rho_spin,grad_spin)
    return scale*q*z*z


def pointwise_c001(theta, rho_spin, grad_spin):
    scale,q,z=_gradient_descriptors(rho_spin,grad_spin)
    return scale*(theta[0]*q+theta[1]*q*q+theta[2]*q*z*z)


def _external_term(function):
    functional=semilocal_functional(lambda params,rho,grad:function(rho,grad),name=function.__name__)
    return AffineTerm(function.__name__,functional)


C002_FORMULA = r"""# c002: quartic-gradient deformation of PBE

Let n=n_alpha+n_beta, g=grad(n), n_v=1e-12 bohr^-3,
r=sqrt(n^2+n_v^2), n_plus=(n+r)/2, w=n^2/(n^2+n_v^2),
s2=|g|^2/[4(3*pi^2)^(2/3)*r^(8/3)], q=s2/(1+s2), and
h=-0.7385587663820223*n_plus^(4/3)*w.
Units: h in Ha/bohr^3, q and coefficients dimensionless.

E_xc = E_xc(PBE) + integral h*(theta1*q^2+theta2*q^2*(1-q)^2) dr.
Both coefficients start at0 and lie in[-0.25,0.25]. At zero coefficients,
the model is exactly PBE. Both terms start at fourth order in the density
gradient; the window also vanishes at q=1. All correction directions are
active for spin-unpolarized input, unlike the c001 polarization term.
This was proposed from the measured c001 energy/stationarity tradeoff;
improved stationarity is a hypothesis, not an established property.
The same scaling, asymptotic-potential and exact-constraint limits as c001 apply.

References: https://doi.org/10.1103/PhysRevLett.77.3865 ;
https://arxiv.org/html/2605.05460 ; https://arxiv.org/html/2610.30022 .
"""


def term_quartic_window(rho_spin, grad_spin):
    scale,q,z=_gradient_descriptors(rho_spin,grad_spin)
    return scale*q*q*(1-q)*(1-q)


def pointwise_c002(theta, rho_spin, grad_spin):
    scale,q,z=_gradient_descriptors(rho_spin,grad_spin)
    return scale*(theta[0]*q*q+theta[1]*q*q*(1-q)*(1-q))


def get_candidate(candidate_id):
    references=['https://doi.org/10.1103/PhysRevLett.77.3865',
        'https://arxiv.org/html/2605.05460','https://arxiv.org/html/2610.30022']
    if candidate_id == 'c001':
        functions=(term_c001_gradient,term_quartic_gradient,term_c001_spin)
        initial,bounds,pointwise,formula=[0.,0.,0.],[[-.5,.5]]*3,pointwise_c001,C001_FORMULA
    elif candidate_id == 'c002':
        functions=(term_quartic_gradient,term_quartic_window)
        initial,bounds,pointwise,formula=[0.,0.],[[-.25,.25]]*2,pointwise_c002,C002_FORMULA
    else:
        raise ValueError('Unknown reviewed functional proposal')
    return dict(terms=tuple(_external_term(fn) for fn in functions),
        initial=initial,bounds=bounds,pointwise=pointwise,formula=formula,references=references)
