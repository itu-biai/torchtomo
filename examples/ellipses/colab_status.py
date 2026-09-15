"""Print progress without blocking the persistent Colab kernel."""

import json
import os
import subprocess
from pathlib import Path

output = Path("/content/torchtomo/examples/ellipses") / os.environ.get("TORCHTOMO_OUTPUT", "results-512")
pid_file = output / "training.pid"

if pid_file.exists():
    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, 0)
        print(f"Training PID {pid} is running")
    except OSError:
        print(f"Training PID {pid} is not running")
else:
    print("No training PID recorded")

print(
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
)

for name in ("fbp-unet", "lpd"):
    path = output / f"{name}-history.json"
    if path.exists():
        history = json.loads(path.read_text())
        print(name, "latest:", history[-1])
        print(name, "best:", min(history, key=lambda row: row["val_mse"]))

path = output / "console.log"
if path.exists():
    print("\n".join(path.read_text().splitlines()[-8:]))
