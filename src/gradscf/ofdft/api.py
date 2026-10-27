"""Small eager facades; differentiable callers use run_ofdft with explicit inputs."""
from .types import OFDFTConfig, OFDFTInputs
from .problem import run_ofdft
from .representations import gaussian_inputs, inputs_from_cell
from .kinetic import KineticFunctional


class OFDFT:
    """Molecular/periodic OFDFT. kernel returns energy; result retains diagnostics.

    No SCF reference is needed. A cell defaults to periodic grid amplitudes; set
    representation='periodic_gaussian' to optimize a Gaussian amplitude expansion.
    """
    def __init__(self, system, *, kinetic='tfvw', xc='svwn', representation=None,
                 kinetic_params=None, xc_energy_fn=None, xc_params=None,
                 grids_level=1, **controls):
        self.system = system
        self.kinetic = KineticFunctional(kinetic) if isinstance(kinetic,str) else kinetic
        self.config = OFDFTConfig(xc=xc,**controls)
        self.representation = representation
        self.kinetic_params = kinetic_params
        self.xc_energy_fn, self.xc_params = xc_energy_fn, xc_params
        self.grids_level = grids_level
        self.inputs = self.result = self.e_tot = self.converged = None

    @property
    def integral_backend(self):
        """Read-only provenance of molecular Gaussian integrals."""
        return 'libcint'

    def kernel(self, initial=None):
        if isinstance(self.system,OFDFTInputs):
            self.inputs = self.system
        elif hasattr(self.system,'a'):
            self.inputs = inputs_from_cell(self.system,representation=self.representation or 'periodic')
        else:
            if self.representation not in (None,'gaussian'):
                raise ValueError('Molecules require the Gaussian representation.')
            self.inputs = gaussian_inputs(self.system,grids_level=self.grids_level)
        self.result = run_ofdft(self.inputs,kinetic=self.kinetic,kinetic_params=self.kinetic_params,
            xc_energy_fn=self.xc_energy_fn,xc_params=self.xc_params,initial=initial,config=self.config)
        self.e_tot, self.converged = self.result.total_energy,bool(self.result.converged)
        return self.e_tot

    def run(self, initial=None):
        self.kernel(initial)
        return self

    def nuc_grad_method(self):
        """Periodic Cell forces/stress; array inputs require an energy callback."""
        from ..pbc.grad import Gradients
        return Gradients(self)
