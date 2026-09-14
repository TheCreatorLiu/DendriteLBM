# Release validation

Validated locally on 2026-09-14. This report describes short execution and source-parity checks, not full-duration reproduction of manuscript figures.

## Environment

- Windows; Python 3.12.14.
- NVIDIA Warp 1.16.0, NumPy 2.5.2, Matplotlib 3.11.0.
- NVIDIA GeForce RTX 5090 D v2; NVIDIA driver 616.56.
- CPU execution was also checked on the local host.

## Six GPU smoke cases

Each case ran 90 base steps with the default paper parameters. Thermal/solutal grids were 64×64, thermosolutal 128×128. All processes completed successfully; all final field entries were finite, the phase field evolved, and tip CSV/NPZ/PNG outputs were present. One thermal run also checked Tecplot DAT output.

| Model | Flow | Maximum absolute velocity component | Max phase change from initial state |
|---|---|---:|---:|
| thermal | off | 0 | 1.0838026 |
| thermal | on | 0.024569996 | 0.14576039 |
| solutal | off | 0 | 1.1746931 |
| solutal | on | 0.059149556 | 0.16514096 |
| thermosolutal | off | 0 | 1.1792235 |
| thermosolutal | on | 0.00040053594 | 1.1796296 |

Flow-off cases have exactly zero physical velocity. Flow-on cases have nonzero physical velocity, verifying that the switch affects hydrodynamic advancement. These short runs do not establish the late-time dendrite morphology or velocity ordering.

## Original-source parity

The original files were copied to an external validation work directory and instrumented only to reduce grid size and run duration, select an isolated output directory, use headless plotting, and save final arrays. The original scientific kernels and numerical parameters were retained. Comparisons used the release `--parameters supplied` mode and the matching flow state.

| Source path | Grid | Steps | Compared fields | Maximum absolute difference |
|---|---|---:|---|---:|
| Thermal-flow | 64×64 | 90 | psi, theta, velocity | 0 |
| Solutal-flow | 64×64 | 90 | psi, U, theta, velocity | 0 |
| Le=50 original with flow loop disabled | 128×128 | 90 | psi, U, theta, velocity | 0 |

All compared arrays were numerically identical for these short matching-source runs. This is not a comparison between the paper-aligned parameter mode and the original sources, nor a parity claim for the two dedicated original no-flow implementations.

## Automated checks

- Four configuration/measurement tests passed: interface interpolation, missing crossings, fixed-phase transport-factor sweeps, target times, and invalid-input rejection.
- All six end-to-end modes passed on CPU and all six passed on CUDA using `tests/test_simulations.py`.
- All ten included PDFs matched their source SHA-256 hashes. A contact sheet was visually inspected for figure identity and complete annotations; previews retain all plot content.
- PNG field output was visually inspected; the data-array transpose and axes agree with the documented `(x,y)` convention.
- Local Markdown links, Python syntax, archive contents and large-file exclusions were checked during packaging.

The GitHub Actions workflow is provided but has not yet run on GitHub. The local CPU result is not a claim of a completed remote CI run.

## Limits

Full 512² / 1000² / 1500² simulations to the manuscript end times were not rerun. Long-time stability, conservation, grid convergence, boundary independence and quantitative agreement with Figures 2–10 remain unverified for the paper-aligned release parameters. The supplied PDFs are reference artifacts, not newly reproduced results. See [MANUSCRIPT_ALIGNMENT.md](MANUSCRIPT_ALIGNMENT.md) before interpreting differences from the original scripts.

Machine-readable metrics: [validation_results.json](validation_results.json).
