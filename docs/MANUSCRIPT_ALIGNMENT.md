# Manuscript and source alignment

Scope: sections 4.1–4.3 of the author-supplied `main0914.tex`, reviewed on 2026-09-14. Figure 1 belongs to the numerical-method section; Figures 2–10 accompany the three cases. Section 4.4 is excluded.

## Common conventions

The three entry points retain the corresponding supplied flow-script D2Q9 MRT kernels. They share CLI configuration, output, health checks and four-direction tip estimation. Physical fluid velocity is `u`; the auxiliary phase-field vector `v` is not fluid velocity. Flow off skips all hydrodynamic advancement and preserves `u=0`; it does not disable phase evolution.

`nF`, `nG`, `nT`, `nH` determine field time steps as dt_i = n_i dx², with dx=1. Update factors `mG`, `mT`, `mH` are field time steps divided by dt_base=dtF. The manuscript notation N_phi, N_T, N_U corresponds to those factors. `--transport-factor` sets N_T for thermal, N_U for the other two models, while nG and dt_base remain unchanged.

Scalar fields use the supplied flow solvers' boundary reconstruction for zero normal gradients on all four edges. The flow uses periodic transverse streaming, a prescribed inlet on the left and an extrapolated outflow on the right. This applies in both parameter modes. Corner values follow the original sequence of separate edge kernels.

## Default physical parameters

| Quantity | Thermal (§4.1) | Solutal (§4.2) | Thermosolutal (§4.3) |
|---|---:|---:|---:|
| Grid | 512² | 1000² | 1500² |
| Seed radius | 10 | 10 | 45 |
| W0 | 2.5 | 2.5 | 2 |
| τ0 | 125 | 50 | 1000 |
| Anisotropy | 0.05 | 0.02 | 0.02 |
| d0/W0 | derived ≈0.1385 | 0.2762 | 0.554 |
| θ0 | −0.55 | 0 | −0.55 |
| U0 | inactive | −0.55 | 0 |
| M_c | 0 | 1 | 0.1 |
| k_c | inactive | 0.15 | 0.15 |
| D_L | inactive | 0.25 | 0.004 |
| D_S/D_L | inactive | 0.01 | 0.01 |
| Thermal diffusivity | 0.2 | temperature inactive | 0.2 |
| Le | no solute | temperature inactive | 50 |
| Inlet when flow on | W0/τ0=0.02 | W0/τ0=0.05 | W0/(5τ0)=0.0004 |
| Kinematic viscosity | 4.62 | 11.55 (source value) | 4.62 (paper definition) |

The solutal manuscript subsection does not explicitly specify its flow viscosity. The supplied code uses `alpha=4 W0²/τ0=0.5`, `Pr=23.1`, `niuL=Pr*alpha=11.55`; the release retains that source value. It should not be inferred to be a separately stated manuscript value.

## Deliberate changes and unresolved provenance

| Item | Supplied files | Release behavior |
|---|---|---|
| Thermal no-flow grid | 513×513 | Paper configuration uses 512×512 |
| No-flow boundaries | Dedicated thermal and solutal scripts use periodic streaming without the flow scripts' edge reconstruction | Unified entry points use scalar zero-gradient reconstruction in both modes |
| Initial interface width | All inspected scripts use tanh((R−r)/sqrt(2 W0)) | `paper` uses manuscript tanh((R−r)/(sqrt(2) W0)); `supplied` retains sqrt(2 W0) |
| Thermal forced-flow base step | 1/45, N_phi=N_T=45 | `paper`: 1/15, N_phi=N_T=15; `supplied` retains 1/45 |
| Solutal forced-flow base step | 1/30, N_phi=N_U=30 | `paper`: 1/15, N_phi=N_U=15; `supplied` retains 1/30 |
| Transport-factor sweep | Supplied files differ in base-step values; exact figure-generation runs are not recorded in them | CLI sweeps only N_T or N_U at fixed dt_base and fixed phase step, following the manuscript's stated sensitivity experiment |
| Thermosolutal flow | Entire flow update block commented out, despite a nonzero inlet formula | `--flow on` activates collision, streaming, boundary and velocity reconstruction; `off` skips them |
| Thermosolutal viscosity | `niuL=Pr*DL=0.0924`, giving ν/α=0.462 | `paper`: `Pr*alpha=4.62`, giving ν/α=23.1; `supplied` retains original ν |
| Thermosolutal final time | 1,200,000 base steps by default, beyond t*=3500 | Default stops at ceil(3500 d0²/DL); actual final time is recorded |
| Tip CSV normalization | Some originals use v τ0/W0, others store dimensional velocities | Common CSV uses v d0/α (thermal) or v d0/DL (solute/thermosolutal), and paper time axes |
| Output / diagnostics | Hardcoded folders and differing file formats | Empty-directory protection, NPZ/PNG/optional DAT, JSON provenance/status and common CSV |

`--parameters supplied` is a **parameter comparison mode**, not a claim of exact reproduction of the two dedicated original no-flow implementations. Short parity comparisons are made only where the numerical paths correspond: original thermal-flow and solutal-flow against unified flow-on, and original thermosolutal with its disabled flow against unified flow-off.

The source files are identified by relative path and SHA-256 in [source_manifest.json](source_manifest.json). The user-supplied solutal-flow path contained a separator mismatch; the inspected file is `Solutal-flow/dendrite_isothermal with tip V.py`.

## Source-coupling behavior

Thermal coupling accumulates phase increments for consumption by temperature updates. Solutal coupling accumulates phase and anti-trapping contributions until the next concentration update. The thermosolutal source uses immediate phase-step injection into the temperature/concentration distributions and macroscopic fields, with coarse transport steps. The release preserves that supplied `proposed` coupling path and removes the unused `direct_coarse` ablation path from the runtime interface.

In the thermosolutal source, phase macroscopic reconstruction occurs before phase collision; that ordering is preserved. The shared recorder samples the source's current macroscopic field rather than introducing an extra reconstruction that would alter source timing. This detail matters when comparing single-step states.

## What the validation establishes

Short runs can establish kernel compilation, finite fields, evolving interfaces, functioning flow switches, source-path parity and correct output conventions. They cannot establish full-time stability, grid convergence, boundary independence, long-time conservation, or agreement with the included manuscript curves. The supplied plot PDFs alone do not contain all field histories or reference datasets needed for an automated quantitative figure comparison.

The seed-width, boundary, time-step and viscosity differences are scientifically meaningful. They remain documented rather than being hidden under a claim that the source files and manuscript were already identical. Full-size runs should be compared against the author's original figure-generation data before claiming quantitative reproduction of Figures 2–10.
