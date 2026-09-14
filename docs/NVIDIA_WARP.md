# NVIDIA Warp

This project uses NVIDIA's scientific computing framework, not the unrelated Warp terminal application.

## Installation and devices

Install `warp-lang==1.16.0` through `requirements.txt`, then verify:

```bash
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
python thermal.py --smoke --device cuda:0
python thermal.py --smoke --device cpu
```

Python 3.10+ is required. The Windows and Linux prebuilt wheels support CPU and NVIDIA CUDA execution. The CPU option is useful for setup and CI checks. Apple Silicon CPU execution is supported upstream; this repository has not been tested on macOS.

For Warp 1.16, the default PyPI wheel uses the CUDA 12.9 runtime; the official guide specifies a 525-or-newer driver for CUDA 12.x builds and a 580-or-newer driver for CUDA 13.x builds. Follow the platform-specific official requirements when selecting a wheel. A separately installed CUDA toolkit is not required for the prebuilt wheel used here.

## Execution model

- `@wp.func` defines device-callable helpers and local equilibrium functions.
- `@wp.kernel` defines parallel lattice updates; `wp.tid()` identifies a lattice node.
- `wp.zeros` creates float32 field/distribution arrays on the selected device.
- `wp.launch` dispatches collision, streaming, boundary and reconstruction kernels.
- Each distribution uses current/next buffers. Source terms couple fields with different update intervals.

The CLI is parsed before Warp initialization. Grid and physical constants are resolved before kernels are compiled. Run a new process for each parameter set; mutating module globals after compilation is unsupported. Scalar transport uses float32 just as the supplied GPU sources do.

## Troubleshooting

Check `nvidia-smi` if CUDA initialization fails. The scripts deliberately fail when a requested CUDA device is unavailable; use `--device cpu` explicitly for CPU execution. On multi-GPU hosts choose `--device cuda:1`, etc.

The first run includes compilation overhead. Warp normally writes its kernel cache in the user's cache directory. In a restricted or read-only environment, choose a writable cache location:

```powershell
# Windows PowerShell
$env:WARP_CACHE_PATH = "$PWD\.warp_cache"
$env:MPLCONFIGDIR = "$PWD\.mpl_cache"
python thermal.py --smoke
```

```bash
# Linux/macOS shell
WARP_CACHE_PATH="$PWD/.warp_cache" MPLCONFIGDIR="$PWD/.mpl_cache" python thermal.py --smoke
```

Large NPZ/PNG/DAT outputs can cost more time than a short simulation. Use `--no-png`, avoid `--dat` when unnecessary, and select a sensible `--output-every` interval. Disabling flow skips flow kernels but currently retains their allocated arrays for a uniform code path.

## Official sources

- [Warp 1.16 installation guide](https://nvidia.github.io/warp/v1.16/user_guide/installation.html)
- [NVIDIA Warp repository and examples](https://github.com/NVIDIA/warp)
- [Warp documentation](https://nvidia.github.io/warp/)

Installation guidance checked on 2026-09-14. The runtime pin represents the locally tested version, not a claim that it is the latest release.
