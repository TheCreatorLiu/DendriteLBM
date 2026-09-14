# Attribution

The numerical kernels were adapted from the five author-provided GPU scripts listed with SHA-256 hashes in `docs/source_manifest.json`. The three unified solvers use the thermal-flow, solutal-flow and Le=50 implementations as their numerical bases. Original no-flow scripts were inspected for parameter and boundary differences, and remain in the author's original source directories.

NVIDIA Warp is developed by NVIDIA and contributors and has its own license, available in its [upstream repository](https://github.com/NVIDIA/warp/blob/main/LICENSE.md). NumPy and Matplotlib retain their respective upstream licenses. Dependencies are installed via pip; their source code and binaries are not vendored here.

The manuscript attributes benchmark comparisons to Zhan et al. (2023), Wang et al. (2021), and Ramirez et al. (2004). The original figure annotations and manuscript-derived captions are retained; see `figures/README.md`. The code license does not relicense third-party publications or reference datasets appearing in those comparisons.
