"""Shared command-line configuration, reproducible output, and tip measurements."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time


@dataclass
class Config:
    model: str
    flow: str
    parameters: str
    device: str
    nx: int
    ny: int
    steps: int
    tip_every: int
    health_every: int
    output_steps: list[int]
    output: str
    png: bool
    dat: bool
    nF: float
    nG: float
    nT: float
    nH: float


def configure(model: str, argv=None) -> Config:
    parser = argparse.ArgumentParser(description=f'{model.title()} multirate MRT-LBM dendrite benchmark')
    parser.add_argument('--flow', choices=('off', 'on'), default='off', help='Disable/enable the hydrodynamic solver (default: off)')
    parser.add_argument('--parameters', choices=('paper', 'supplied'), default='paper', help='Paper-aligned values or supplied flow-script values; see docs/MANUSCRIPT_ALIGNMENT.md')
    parser.add_argument('--device', default='cuda:0', help='Warp device: cuda:0, cuda:1, or cpu (default: cuda:0)')
    parser.add_argument('--nx', type=int)
    parser.add_argument('--ny', type=int)
    parser.add_argument('--steps', type=int, help='Number of base steps; default reaches the final manuscript time')
    parser.add_argument('--transport-factor', type=int, help='N_T (thermal) or N_U (solutal/thermosolutal); phase time step stays fixed')
    parser.add_argument('--tip-every', type=int, help='Tip sampling interval in base steps; must align with phase updates')
    parser.add_argument('--health-every', type=int, help='Field health-check interval in base steps')
    parser.add_argument('--output-every', type=int, help='Replace manuscript snapshot times by this base-step interval')
    parser.add_argument('--output', help='New or empty output directory; existing results are never overwritten')
    parser.add_argument('--no-png', action='store_true', help='Skip PNG snapshots; NPZ and CSV are still written')
    parser.add_argument('--dat', action='store_true', help='Also write Tecplot ASCII field snapshots')
    parser.add_argument('--smoke', action='store_true', help='Small grid and short duration for installation checks, not paper reproduction')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved configuration without importing Warp or allocating arrays')
    args = parser.parse_args(argv)
    flow = args.flow == 'on'
    full_grid = {'thermal': 512, 'solutal': 1000, 'thermosolutal': 1500}[model]
    if model == 'thermosolutal':
        base, nG, factor = 1.0, 1.0, 30
        physical_times = [0, 3500 * (0.554 * 2)**2 / 0.004]
        sample = 2000
    else:
        denominator = (45 if model == 'thermal' else 30) if args.parameters == 'supplied' else 15
        base, nG, factor = (1.0 / denominator, 1.0, denominator) if flow else (1.0, 1.0, 1)
        if model == 'thermal':
            physical_times = [125 * t for t in (0, 8, 16, 32, 64, 128)]
        else:
            physical_times = [50 * t for t in ((0, 40, 120, 200, 400, 600) if flow else (0, 40, 120, 200, 400, 600, 800, 1000))]
        sample = round(100 / base)
    if args.transport_factor is not None:
        factor = args.transport_factor
    if factor < 1:
        parser.error('--transport-factor must be positive')
    nT = base * factor if model == 'thermal' else 1.0
    nH = base * factor if model != 'thermal' else 1.0
    mG = round(nG / base)
    # Avoid adding a full step when a mathematically integral endpoint is
    # represented a few ulps above that integer (Le=50 gives 1,074,206 steps).
    default_steps = math.ceil(physical_times[-1] / base - 1e-9)
    nx = args.nx if args.nx is not None else (128 if args.smoke and model == 'thermosolutal' else 64 if args.smoke else full_grid)
    ny = args.ny if args.ny is not None else nx
    steps = args.steps if args.steps is not None else (max(90, 3 * mG, 3 * factor) if args.smoke else default_steps)
    tip_every = args.tip_every if args.tip_every is not None else (mG if args.smoke else sample)
    health_every = args.health_every if args.health_every is not None else tip_every
    if min(nx, ny) < (112 if model == 'thermosolutal' else 32):
        parser.error('Grid is too small to contain the fixed benchmark seed and diffuse interface')
    if nx % 2 or ny % 2:
        parser.error('Use even grid dimensions so the seed center and sampled centerlines coincide')
    if min(steps, tip_every, health_every) < 1 or (args.output_every is not None and args.output_every < 1):
        parser.error('Steps and output/sampling intervals must be positive')
    if tip_every % mG:
        parser.error(f'--tip-every must be a multiple of the phase update factor {mG}')
    output_steps = [round(t / base) for t in physical_times if round(t / base) <= steps]
    if args.output_every is not None:
        output_steps = list(range(0, steps + 1, args.output_every))
    if args.smoke:
        output_steps = [0]
    output_steps = sorted(set(output_steps + [0, steps]))
    stamp = time.strftime('%Y%m%d-%H%M%S') + f'-{time.time_ns() % 1000000:06d}'
    output = args.output or str(Path('outputs') / f'{model}-flow-{args.flow}-{stamp}')
    cfg = Config(model, args.flow, args.parameters, args.device, nx, ny, steps, tip_every,
                 health_every, output_steps, output, not args.no_png, args.dat, base, nG, nT, nH)
    if args.dry_run:
        print(json.dumps(asdict(cfg), indent=2))
        raise SystemExit(0)
    # Ignore inherited legacy DENDRITE_* knobs: configuration has one source of truth.
    os.environ.setdefault('MPLBACKEND', 'Agg')
    return cfg


def axis_tip(line, center: int, direction: int) -> float:
    """Interpolate the first outward solid-to-liquid crossing; NaN if unresolved."""
    if line[center] < 0:
        return float('nan')
    for i in range(center, len(line) - 1 if direction > 0 else 0, direction):
        j = i + direction
        if line[i] >= 0 and line[j] < 0:
            return i + direction * float(line[i] / (line[i] - line[j]))
    return float('nan')


class Session:
    def __init__(self, cfg: Config, scope: dict):
        import numpy as np
        import warp as wp
        self.np, self.wp, self.cfg = np, wp, cfg
        self.output = Path(cfg.output)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f'Output directory is not empty: {self.output}. Choose a new --output path.')
        self.output.mkdir(parents=True, exist_ok=True)
        self.started, self.step, self.previous = time.perf_counter(), 0, None
        self.dt = scope['dtF']
        self.d0, self.tau0 = scope['d0'], scope['tau0']
        self.diffusivity = scope['alpha'] if cfg.model == 'thermal' else scope['DL']
        keys = ('W0', 'tau0', 'd0', 'lamda', 'alpha', 'DL', 'DS', 'Le', 'Pr', 'niuL', 'MC', 'U0', 'T0', 'k0', 'SEED_WIDTH', 'INLET_UX', 'mF', 'mG', 'mT', 'mH', 'dtF', 'dtG', 'dtT', 'dtH')
        device = wp.get_device()
        sources = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ('runtime.py', cfg.model + '.py')}
        metadata = {'configuration': asdict(cfg), 'physical_parameters': {key: scope[key] for key in keys if key in scope},
                    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'warp': wp.__version__},
                    'device': str(device), 'device_name': device.name, 'source_sha256': sources,
                    'boundary_conditions': 'zero-normal-gradient scalar reconstruction; flow periodic in y, inlet/outlet in x',
                    'time_star_definition': 't/tau0' if cfg.model != 'thermosolutal' else 't*DL/d0^2',
                    'velocity_star_definition': 'v*d0/alpha' if cfg.model == 'thermal' else 'v*d0/DL'}
        (self.output / 'run_config.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        self.status('running')
        print(f'{cfg.model}: flow={cfg.flow}, device={device}, grid={cfg.nx}x{cfg.ny}, steps={cfg.steps}, dt_base={self.dt:g}', flush=True)
        print(f'Output: {self.output.resolve()}', flush=True)

    def status(self, status, reason=''):
        data = {'status': status, 'completed_step': self.step, 'requested_steps': self.cfg.steps,
                'elapsed_seconds': time.perf_counter() - self.started, 'reason': reason}
        temporary = self.output / 'run_status.tmp'
        temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        temporary.replace(self.output / 'run_status.json')

    def observe(self, step, read_fields):
        self.step = step
        cfg = self.cfg
        sample = step % cfg.tip_every == 0 or step == cfg.steps
        snapshot = step in cfg.output_steps
        if not (sample or snapshot or step % cfg.health_every == 0):
            return
        fields = read_fields()
        for name, values in fields.items():
            if not self.np.isfinite(values).all():
                raise FloatingPointError(f'{name} contains NaN/Inf at step {step}')
            limit = 5 if name == 'psi' else 1000
            if float(self.np.max(self.np.abs(values))) > limit:
                raise FloatingPointError(f'{name} exceeds health limit {limit} at step {step}')
        if cfg.flow == 'off' and self.np.any(fields['velocity'] != 0):
            raise FloatingPointError('Flow is off but physical velocity is nonzero')
        if sample:
            self.record_tip(step, fields['psi'])
        if snapshot:
            self.snapshot(step, fields)
        self.status('running')

    def record_tip(self, step, psi):
        cx, cy = self.cfg.nx // 2, self.cfg.ny // 2
        tips = {name: axis_tip(line, center, direction) for name, line, center, direction in (
            ('west', psi[:, cy], cx, -1), ('east', psi[:, cy], cx, 1),
            ('south', psi[cx, :], cy, -1), ('north', psi[cx, :], cy, 1))}
        lengths = {'west': cx - tips['west'], 'east': tips['east'] - cx,
                   'south': cy - tips['south'], 'north': tips['north'] - cy}
        physical_time = step * self.dt
        time_star = physical_time / self.tau0 if self.cfg.model != 'thermosolutal' else physical_time * self.diffusivity / self.d0**2
        row = {'step': step, 'time': physical_time, 'time_star': time_star}
        for name, length in lengths.items():
            velocity = (length - self.previous[1][name]) / ((step - self.previous[0]) * self.dt) if self.previous is not None else float('nan')
            row.update({f'length_{name}': length, f'velocity_{name}': velocity, f'velocity_star_{name}': velocity * self.d0 / self.diffusivity})
        path = self.output / 'tip_velocity.csv'
        with path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if self.previous is None:
                writer.writeheader()
            writer.writerow(row)
        self.previous = (step, lengths)
        print(f'step={step}, time*={time_star:.6g}, west length={lengths["west"]:.6g}', flush=True)

    def snapshot(self, step, fields):
        np = self.np
        np.savez_compressed(self.output / f'fields_{step:09d}.npz', step=step, time=step*self.dt, **fields)
        scalar = {key: value for key, value in fields.items() if value.ndim == 2}
        speed = np.linalg.norm(fields['velocity'], axis=2)
        if self.cfg.dat:
            x, y = np.indices((self.cfg.nx, self.cfg.ny))
            names = ['X', 'Y'] + list(scalar) + ['ux', 'uy']
            arrays = [x, y] + list(scalar.values()) + [fields['velocity'][:, :, 0], fields['velocity'][:, :, 1]]
            # Tecplot I index varies fastest; Fortran flattening preserves (x,y) layout.
            header = 'TITLE="Dendrite"\nVARIABLES=' + ','.join(f'"{name}"' for name in names) + f'\nZONE I={self.cfg.nx}, J={self.cfg.ny}, F=POINT'
            np.savetxt(self.output / f'fields_{step:09d}.dat', np.column_stack([a.ravel(order='F') for a in arrays]), header=header, comments='', fmt='%.8e')
        if self.cfg.png:
            import matplotlib.pyplot as plt
            panels = dict(scalar)
            if self.cfg.flow == 'on':
                panels['speed'] = speed
            fig, axes = plt.subplots(1, len(panels), figsize=(4*len(panels), 4), constrained_layout=True, squeeze=False)
            for ax, (name, values) in zip(axes.flat, panels.items()):
                im = ax.imshow(values.T, origin='lower', aspect='equal')
                ax.contour(fields['psi'].T, levels=[0], colors='black', linewidths=0.6)
                ax.set(title=f'{name}, step={step}', xlabel='x (lattice nodes)', ylabel='y (lattice nodes)')
                fig.colorbar(im, ax=ax, shrink=0.8)
            fig.savefig(self.output / f'fields_{step:09d}.png', dpi=140)
            plt.close(fig)

    def finish(self, read_fields):
        fields = read_fields()
        self.np.savez_compressed(self.output / 'final_state.npz', step=self.step, time=self.step*self.dt, **fields)
        self.status('completed')

    def fail(self, exc):
        self.status('failed', f'{type(exc).__name__}: {exc}')
