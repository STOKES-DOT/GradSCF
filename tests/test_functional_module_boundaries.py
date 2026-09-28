"""Generic DFT/training must not import any concrete neural model package."""
from pathlib import Path
import os
import subprocess
import sys


def test_generic_functional_and_training_run_without_model_imports():
    code = r'''
import importlib.abc
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'gradscf.model' or fullname.startswith('gradscf.model.'):
            raise AssertionError('Generic framework imported a concrete model: '+fullname)
sys.meta_path.insert(0, NoModels())
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from gradscf import dft, training, scf
from gradscf.dft.derivatives import xc_kernel_action
sys.path.insert(0, 'tests')
from test_external_neural_xc import molecule
f = dft.Functional(lambda s:s.rho, lambda p,x:.01*p['scale']*jnp.sum(x*x))
mol=molecule()
dm=mol.rdm1.sum(axis=0)
p={'scale':jnp.array(.3)}
assert f.potential(p,mol,dm).shape == dm.shape
assert f.kernel_action(p,mol,dm,jnp.eye(2)).shape == dm.shape
trainer=training.Trainer(f,params=p)
trainer.mode='implicit'
trainer.scf=dict(max_cycle=60,damping=0.,conv_tol_density=1e-10,conv_tol_energy=1e-12,eigenvalue_jitter=0.)
trainer.run([training.Sample(mol,energy=-2.1)],steps=1)
assert trainer.history['update_accepted'][-1]
assert len(trainer.history['loss'])==2
assert not any(x=='gradscf.model' or x.startswith('gradscf.model.') for x in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, PYTHONPATH='src', JAX_PLATFORMS='cpu'),
        text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr


def test_old_generic_implementation_paths_are_removed():
    root=Path(__file__).resolve().parents[1]/'src/gradscf'
    assert not (root/'model/training').exists()
    assert not (root/'model/neural_xc/external.py').exists()
    assert not (root/'scf/xc_energy.py').exists()
