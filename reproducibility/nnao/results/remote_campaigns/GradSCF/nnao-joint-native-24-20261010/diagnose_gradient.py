"""Read-only decomposition of a joint finite-difference mismatch."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import jax
import jax.numpy as jnp

source, run, output = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('native_joint_probe', source/'tools/train_nnao_joint_ad.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
config = json.loads((run/'training/metadata.json').read_text())['configuration']
args = SimpleNamespace(**config)
args.output_dir = output
output.mkdir(parents=True, exist_ok=True)
rows = m.base.pilot.read_structures(Path(args.structures), ids=['W4-11/ch4'])
theta, parent = m.base.read_parent(Path(args.checkpoint), rows)
engine = m.NativeJointEngine(rows, args, parent, None)
x = np.concatenate((theta, np.zeros(14)))
identifier = rows[0]['structure_id']
c = engine.contexts[identifier]
value, g, info = engine.molecule(x, identifier)
print('initial', value, info, flush=True)
record = dict(initial=value, diagnostics=info, directions={})
base_direction = np.sin(np.arange(x.size)+1.)
base_direction[:-14] /= np.linalg.norm(base_direction[:-14])
base_direction[-14:] *= .1
for label in ('theta', 'beta'):
    d = base_direction.copy()
    d[-14:] = 0. if label == 'theta' else d[-14:]
    if label == 'beta': d[:-14] = 0.
    analytic = float(g@d)
    values = {}
    for h in (1e-2, 1e-3, 1e-4, 1e-5):
        e = {k: engine.molecule(x+k*h*d, identifier, gradient=False)[0] for k in (-2, -1, 1, 2)}
        fd = (8*(e[1]-e[-1])-e[2]+e[-2])/(12*h)
        values[str(h)] = dict(finite_difference=fd, error=analytic-fd, energies=e)
        print(label, h, 'analytic', analytic, 'fd', fd, flush=True)
    record['directions'][label] = dict(analytic=analytic, steps=values)

theta_gpu = jax.device_put(theta, engine.device)
outputs = engine.predict(theta_gpu, c['graph'])
outputs_cpu = jax.device_put(outputs, engine.cpu)
beta_cpu = jax.device_put(np.zeros(14), engine.cpu)
physical = c['physical'](outputs_cpu, beta_cpu)
(energy, details), gp = c['gradient'](jax.device_put(physical, engine.device), c['enuc'])
gp_cpu = jax.device_put(gp, engine.cpu)
do, db = c['pullback'](outputs_cpu, beta_cpu, gp_cpu)
dbeta = jax.device_put(base_direction[-14:], engine.cpu)
with jax.default_device(engine.cpu):
    tangent = jax.jvp(lambda b: c['physical'](outputs_cpu, b), (beta_cpu,), (dbeta,))[1]
    h = 1e-4
    fs = {k:c['physical'](outputs_cpu, beta_cpu+k*h*dbeta) for k in (-2, -1, 1, 2)}
    fd = jax.tree.map(lambda p,m,p2,m2:(8*(p-m)-p2+m2)/(12*h), fs[1], fs[-1], fs[2], fs[-2])
    record['physical_beta'] = dict(
        pullback_dot=float(jnp.vdot(db, dbeta)),
        jvp_dot=float(sum(jnp.vdot(a,b) for a,b in zip(gp_cpu,tangent))),
        fd_dot=float(sum(jnp.vdot(a,b) for a,b in zip(gp_cpu,fd))),
        max_errors=[float(jnp.max(jnp.abs(a-b))) for a,b in zip(tangent,fd)])
dtheta = jax.device_put(base_direction[:-14], engine.device)
nn_tangent = jax.jvp(lambda t:engine.predict(t,c['graph']), (theta_gpu,), (dtheta,))[1]
nn_fd = (engine.predict(theta_gpu+1e-4*dtheta,c['graph'])-engine.predict(theta_gpu-1e-4*dtheta,c['graph']))/2e-4
record['network'] = dict(dtype=str(outputs.dtype), max_error=float(jnp.max(jnp.abs(nn_tangent-nn_fd))),
    jvp_dot=float(jnp.vdot(jax.device_put(do,engine.device),nn_tangent)),
    fd_dot=float(jnp.vdot(jax.device_put(do,engine.device),nn_fd)))
print(json.dumps(record, indent=2), flush=True)
(output/'diagnosis.json').write_text(json.dumps(record, indent=2)+'\n')
