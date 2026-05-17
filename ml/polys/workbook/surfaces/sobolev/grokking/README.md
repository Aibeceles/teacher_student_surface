# Grokking / mechinterp experiments (character Sobolev student)

Follow-up notebooks to [`sobolev_student_character_periodic.ipynb`](../sobolev_student_character_periodic.ipynb): Fourier probes, dynamics, manifold ablation, modulus sweep, and grokking grids. All use JAX + `sobolev_distill_character` under `ml/polys`.

## Prerequisites

1. **JAX venv** — see [repo root README — JAX and notebook setup](../../../../../../README.md#jax-and-notebook-setup): `python -m venv .venv`, `pip install -r ml/polys/requirements-jax-gpu-wsl.txt`.
2. **Working directory**: open each notebook with **`grokking/`** (this folder) as the Jupyter cwd so `_polys_root_for_import()` resolves `ml/polys`.

```bash
# from repo root
. .venv/bin/activate
cd ml/polys/workbook/surfaces/sobolev/grokking
jupyter notebook fourier_decomp.ipynb
```

## Recommended experiment order

Runs are independent, but the narrative order matches the notebook intros:

| Step | Notebook | Default `p` |
|------|----------|-------------|
| 0 (parent dir) | [`sobolev_student_character_periodic.ipynb`](../sobolev_student_character_periodic.ipynb) | 8 |
| 1 | [`fourier_decomp.ipynb`](fourier_decomp.ipynb) | 8 |
| 2 | [`dynamics_excluded_loss.ipynb`](dynamics_excluded_loss.ipynb) | 8 |
| 3 | [`manifold_and_ablation.ipynb`](manifold_and_ablation.ipynb) | 8 |
| 4 | [`modulus_sweep.ipynb`](modulus_sweep.ipynb) | 17, 23, 113 |
| 5 | [`grokking_baseline_with_decay.ipynb`](grokking_baseline_with_decay.ipynb) | 8 (grid) + 17 (confirm) |
| 6 | [`grokking_capacity_sweep.ipynb`](grokking_capacity_sweep.ipynb) | 17 |

## How to run

Open the `.ipynb` in Jupyter or VS Code, select the `.venv` kernel, run cells top-to-bottom from the top of the notebook. Save the notebook to persist outputs.

This repo does **not** include `_run_*.py` / `_dryrun_*.py` headless runners or `_build_*.py` notebook generators; those live in the parent **Aibeceles** monorepo if you need CI-style execution or to regenerate `.ipynb` sources.

### Wall-clock hints (GPU)

| Notebook | Rough budget |
|----------|----------------|
| `fourier_decomp`, `dynamics_excluded_loss`, `manifold_and_ablation` | tens of minutes at p=8 |
| `modulus_sweep` (all three sweep cells) | ~30–50 min |
| `grokking_baseline_with_decay` | multi-hour grid |
| `grokking_capacity_sweep` | depends on grid size |

For `modulus_sweep`, you can re-open the `.ipynb` while a long sweep cell runs to inspect partial outputs.

## Setting the modulus `p`

`p` is the cyclic modulus for the character teacher \(\zeta^{(x+y) \bmod p}\) on the \(p \times p\) lattice.

| Notebook | Where to set `p` | Notes |
|----------|------------------|-------|
| `fourier_decomp`, `dynamics_excluded_loss`, `manifold_and_ablation` | Config cell: `MODULUS = 8` | Set `MAX_N` to the same value as `p`. Re-run **training** after changing `p`; DFT / summary cells only **report** `rep.p`. |
| `modulus_sweep` | Cells `sweep_p17`, `sweep_p23`, `sweep_p113`: `_p, _mesh, _ep, _ramp, _bs = ...` | The **`summary` cell does not set `p`** — it prints rows from `results` populated by the sweep cells above. |
| `modulus_sweep` | `SWEEP = [...]` in config | **Documentation only.** Changing `SWEEP` alone does not run anything; edit the `sweep_p*` cells. |
| `grokking_baseline_with_decay` | `P8_MODULUS`, `P17_MODULUS` (and `P*_MAX_N`, `P*_MESH_N`) | 27-row grid at p=8; one confirmation row at p=17. |
| `grokking_capacity_sweep` | `P_MODULUS` (default `17`) | Entire capacity grid uses that modulus. |

### `modulus_sweep` workflow

1. Set `p` (and mesh / epochs / ramp / batch) in the relevant **`sweep_p*`** cell tuple, e.g. `_p, _mesh, _ep, _ramp, _bs = 17, 64, 2000, 200, 256`.
2. Run that sweep cell (calls `_train_and_score` and stores `results['p=17'] = out`).
3. Run **`summary`** and the spectrum cells — they aggregate whatever keys exist in `results`.
4. **Skip a row**: replace the sweep cell body with `# skipped`.
5. **Add a row**: duplicate a sweep cell pattern, set a new `_p` and `results['p=<p>'] = out`.

Default hyperparameters are **paired per row** (not derived automatically from `p`):

| `p` | `mesh_n` | `epochs` | `ramp_epochs` | `batch_size` |
|-----|----------|----------|---------------|--------------|
| 17 | 64 | 2000 | 200 | 256 |
| 23 | 96 | 3000 | 300 | 256 |
| 113 | 128 | 6000 | 600 | 512 |

Roughly 4–8 mesh points per lattice cell along each axis; scale manually if you add a new modulus.

### Common pitfalls

- **`summary` prints `no sweep rows ran`** — no `sweep_p*` cell completed, or all were skipped. Re-run the sweep cells first.
- **`four_of_four` is False in `modulus_sweep`** — expected with `energy_pd=0.0` in that notebook (`pd_certificate` untrained). See the takeaway markdown at the end of `modulus_sweep.ipynb`.
- **Imports fail** — Jupyter cwd must be `grokking/` (the folder containing the notebook).

## File map (this repo)

| Kind | Files |
|------|--------|
| Notebooks | `*.ipynb` in this folder |

Parent overview: [`../README.md`](../README.md).
