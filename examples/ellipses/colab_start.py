"""Launch the 512x512, 120-epoch experiment in the persistent Colab session."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

# Set TORCHTOMO_OUTPUT and TORCHTOMO_ARGS in the kernel before running this file to
# direct a variant run elsewhere, for example a different LPD size.
root = Path("/content/torchtomo")
output = root / "examples/ellipses" / os.environ.get("TORCHTOMO_OUTPUT", "results-512")
output.mkdir(parents=True, exist_ok=True)
pid_file = output / "training.pid"


def running_pid():
    """Return the recorded PID when that process is still running the trainer.

    A finished child stays visible as a zombie until the notebook kernel reaps it,
    and Linux recycles PIDs, so the command line is checked rather than the PID alone.
    """
    if not pid_file.exists():
        return None
    pid = int(pid_file.read_text().strip())
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return None
    state = [line for line in status.read_text().splitlines() if line.startswith("State:")]
    if state and state[0].split()[1] == "Z":
        return None
    if b"train.py" not in Path(f"/proc/{pid}/cmdline").read_bytes():
        return None
    return pid


existing = running_pid()
if existing is not None:
    raise RuntimeError(f"Training is already running with PID {existing}")

command = [
    sys.executable,
    "examples/ellipses/train.py",
    "--image-size",
    "512",
    "--angles",
    "90",
    "--batch-size",
    "5",
    "--unet-epochs",
    "120",
    "--lpd-epochs",
    "120",
    "--device",
    "cuda",
    "--output",
    str(output),
]
command += shlex.split(os.environ.get("TORCHTOMO_ARGS", ""))
if (output / "config.json").exists():
    command.append("--resume")

with (output / "console.log").open("a") as log:
    job = subprocess.Popen(
        command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
pid_file.write_text(str(job.pid))
print("Training PID:", job.pid)
print("Resuming:", "--resume" in command)
print("Output:", output)
