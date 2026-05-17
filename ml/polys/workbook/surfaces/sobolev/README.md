# Sobolev surface experiments (character teacher / student)

JAX notebooks that distill a **character** (modular) teacher into a SIREN student on a \(p \times p\) lattice mesh. The [`grokking/`](grokking/) folder continues the same stack with mechinterp probes, modulus sweeps, and grokking grids.

## What lives here

| Path | Purpose |
|------|---------|
| `sobolev_student_character_periodic.ipynb` | Periodic-cardinal teacher; main baseline at **p = 8** |
| [`grokking/`](grokking/) | Six follow-up notebooks — see [`grokking/README.md`](grokking/README.md) |

Library code: `ml/polys/sobolev_distill_character/` (imported from notebooks via `_polys_root_for_import()`).

## Prerequisites

JAX / venv setup: [repo root README — JAX and notebook setup](../../../../../README.md#jax-and-notebook-setup).

## Setting `p` (entry notebook)

In `sobolev_student_character_periodic.ipynb`, the config cell defines:

```python
MODULUS = 8
MAX_N = 8   # lattice side; keep equal to MODULUS
```

Change `MODULUS` (and `MAX_N`), then re-run teacher build and training cells. Plots and verdict cells use `p = MODULUS` downstream.

## How to run

This extract ships **notebooks only** (no `_run_*.py` headless runners). Use Jupyter or VS Code with the notebook folder as the working directory.

```bash
# from repo root
python -m venv .venv
. .venv/bin/activate
pip install -r ml/polys/requirements-jax-gpu-wsl.txt
pip install jupyter ipykernel   # if not already installed

cd ml/polys/workbook/surfaces/sobolev
jupyter notebook sobolev_student_character_periodic.ipynb
```

Open the `.ipynb`, select a kernel from `.venv`, and run cells top-to-bottom.

Headless `_run_*.py` scripts and notebook builders live in the parent **Aibeceles** monorepo, not in this extract.

## Grokking / mechinterp follow-up

For **modulus sweep**, **Fourier probes**, **how to set `p` per notebook**, and wall-clock budgets, use:

**[grokking/README.md](grokking/README.md)**
