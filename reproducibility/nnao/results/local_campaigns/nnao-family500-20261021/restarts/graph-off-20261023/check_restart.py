"""Check launch flags and signal/failure handling without running numerical work."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parent
scripts = list(root.glob("*.slurm"))
assert len(scripts) == 8
for script in scripts:
    text = script.read_text()
    assert '--xla_gpu_enable_command_buffer=' in text
    assert '#SBATCH --signal=B:USR1@300' in text
    assert '#SBATCH --cpus-per-task=32' in text
    assert '#SBATCH --output=artifacts/' in text
    assert '#SBATCH --time=08:00:00' in text if '--nodelist=n005' in text else '#SBATCH --time=72:00:00' in text
    subprocess.run(["bash", "-n", str(script)], check=True)

template = scripts[0].read_text()
with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    fake = directory / "scontrol"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$MOCK_LOG"\n')
    fake.chmod(0o755)
    for label, command, expected, send_signal in (
        ("success", "true", 0, False),
        ("numerical_failure", "bash -c 'exit 7'", 7, False),
        ("time_limit", "sleep 1", 0, True),
    ):
        script = directory / (label + ".sh")
        lines = template.splitlines()
        lines = [command + " &" if line.startswith("python -u ") else
                 "date" if line == "date -Is" else line for line in lines]
        script.write_text("\n".join(lines) + "\n")
        log = directory / (label + ".log")
        env = dict(os.environ, PATH=str(directory) + os.pathsep + os.environ["PATH"],
                   SLURM_JOB_ID="123", MOCK_LOG=str(log))
        process = subprocess.Popen(["bash", str(script)], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if send_signal:
            time.sleep(0.15)
            process.send_signal(signal.SIGUSR1)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == expected, (label, process.returncode, stdout, stderr)
        assert (log.read_text() == "requeue 123\n") if send_signal else not log.exists()
        print(label + ": passed")
print("Eight launch scripts and three shell-control cases passed; no SCF or GPU calls.")
