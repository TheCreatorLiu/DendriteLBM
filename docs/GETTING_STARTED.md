# Getting started

## 1. Install

Download the repository from GitHub or clone its published URL, then open a terminal in its root directory. Use a fresh virtual environment to keep the tested Warp version separate from other projects.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If activation is restricted, use `.\.venv\Scripts\python.exe` directly instead of changing system execution policy.

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Check the installed device list:

```bash
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
```

## 2. Run a small example

```bash
python thermal.py --smoke --flow off --output outputs/thermal-demo
python thermal.py --smoke --flow on --output outputs/thermal-flow-demo
python solutal.py --smoke --flow off --output outputs/solutal-demo
python solutal.py --smoke --flow on --output outputs/solutal-flow-demo
python thermosolutal.py --smoke --flow off --output outputs/thermosolutal-demo
python thermosolutal.py --smoke --flow on --output outputs/thermosolutal-flow-demo
```

These commands use 64×64 nodes for thermal/solutal and 128×128 for thermosolutal, retaining the physical seed radius. They exercise at least three transport updates at the default factor. These small domains are only for installation checks: their boundaries affect growth too early for scientific comparisons.

Each `--output` path must be new or empty. Omit `--output` for a generated unique name. A successful process exits with code zero and writes `"status": "completed"` to `run_status.json`; detected NaN/Inf or explosive fields fail with a nonzero process exit.

## 3. Select a paper case

Omit `--smoke` to use the full domain and manuscript target duration. All quantities are lattice quantities unless a nondimensional definition is stated.

| Model/mode | Grid | Base time step | Phase factor | Transport factor | Base steps |
|---|---:|---:|---:|---:|---:|
| Thermal, off | 512×512 | 1 | 1 | N_T=1 | 16,000 |
| Thermal, on | 512×512 | 1/15 | 15 | N_T=15 | 240,000 |
| Solutal, off | 1000×1000 | 1 | 1 | N_U=1 | 50,000 |
| Solutal, on | 1000×1000 | 1/15 | 15 | N_U=15 | 450,000 |
| Thermosolutal, either | 1500×1500 | 1 | 1 | N_U=30 | ceil(3500 d0² / DL) |

```bash
python thermal.py --flow on --transport-factor 15
python thermal.py --flow on --transport-factor 30
python thermal.py --flow on --transport-factor 45
python solutal.py --flow on --transport-factor 15
python solutal.py --flow on --transport-factor 30
python solutal.py --flow on --transport-factor 45
python thermosolutal.py --flow on --transport-factor 30
python thermosolutal.py --flow on --transport-factor 40
python thermosolutal.py --flow on --transport-factor 50
```

The transport factor changes the selected transport time step, not the flow base time step or phase time step. These are paper-aligned configurations; full-duration figure agreement has not been established in the release validation.

## 4. Control runtime and output

```bash
python thermosolutal.py --flow on --steps 600 --nx 128 --ny 128 --tip-every 30 --no-png
python thermal.py --flow on --steps 1500 --output-every 750 --dat
python solutal.py --flow on --dry-run
```

- `--steps`: number of base steps, not phase updates. Physical time is `step * dtF`.
- `--nx`, `--ny`: even dimensions; changing them changes the physical domain.
- `--tip-every`: base-step interval divisible by the phase update factor. Default thermal/solutal interval corresponds to 100 physical time units; thermosolutal uses 2,000 base steps.
- `--health-every`: field health-check interval; defaults to the tip interval. Checks also occur at snapshots and the final step.
- `--output-every`: regular snapshot interval; otherwise use the manuscript contour times plus the actual final step.
- `--no-png`: skip quick-look images. NPZ output remains enabled.
- `--dat`: add Tecplot ASCII output. This can create large files at full resolution.
- `--parameters supplied`: compare with supplied flow-script parameter conventions; see alignment notes.

Default output times are t/τ0 = 0, 8, 16, 32, 64, 128 (thermal); 0, 40, 120, 200, 400, 600, 800, 1000 (solutal off); 0, 40, 120, 200, 400, 600 (solutal on); and 0 plus the step at or immediately after t DL/d0²=3500 (thermosolutal). Add intermediate thermosolutal snapshots with `--output-every`.

## 5. Read results

```python
import csv
import numpy as np
import matplotlib.pyplot as plt

folder = 'outputs/thermal-demo'
with np.load(f'{folder}/final_state.npz') as data:
    plt.imshow(data['psi'].T, origin='lower')  # stored axes are (x, y)
    plt.colorbar(label='phase field')
    plt.show()
with open(f'{folder}/tip_velocity.csv', newline='') as stream:
    rows = list(csv.DictReader(stream))
print(rows[-1])
```

West/east correspond to negative/positive x; south/north to negative/positive y. Tips are the first outward zero crossings from the seed center, found by linear interpolation. Velocities are backward differences over the actual sampling interval. This common estimator replaces the original scripts' differing CSV schemas; it does not smooth the curves or import literature comparison data.

## 6. Run checks

```bash
python -m unittest discover -s tests -v
```

Enable the six simulation checks in PowerShell:

```powershell
$env:DENDRITE_RUN_SIM_TESTS = '1'
$env:DENDRITE_TEST_DEVICE = 'cuda:0'
python -m unittest discover -s tests -v
```

On Linux:

```bash
DENDRITE_RUN_SIM_TESTS=1 DENDRITE_TEST_DEVICE=cpu python -m unittest discover -s tests -v
```

GitHub Actions uses CPU tests; it does not validate CUDA performance or reproduce the full paper figures.
