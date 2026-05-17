# teacher_student_surface

A self-contained extract of the math-first piece of the `surfaces/` writeup from
the parent `Aibeceles` monorepo. The repo carries the minimum codebase needed
to render and reproduce the figures in
[`00_expository.md`](ml/polys/workbook/surfaces/documentation/00_expository.md),
a third datapoint along the Welch Labs / Nanda / Anthropic mechanistic-
interpretability axis.

## Source video

The expository is structured as a probe-by-probe reply to the Welch Labs video
*The most complex AI model that we fully understand*:

- YouTube: https://www.youtube.com/watch?v=D8GOeCFFby4
- Local transcript:
  [`welsh_nanada_transcript.txt`](ml/polys/workbook/surfaces/documentation/welsh_nanada_transcript.txt)
  (auto-generated YouTube captions, lightly cleaned)

The seven probes in `00_expository.md` follow the video's walkthrough one-for-
one; the transcript is included so readers can match a passage in the video
against a section in the expository without leaving the repo.

## Layout

```
ml/polys/
├── requirements-jax-gpu-wsl.txt    # pinned deps (jax[cuda12], equinox, optax, ...)
├── graphic_zero/                   # barycentric Lagrange + Hermite-Birkhoff bases
├── graphic_zero_character/         # complex-character variant + character_birkhoff
├── sobolev_distill/                # base Sobolev-distillation package (trunk, heads, probes)
├── sobolev_distill_character/      # (Re, Im) character variant (the package the doc uses)
└── workbook/surfaces/
    ├── documentation/              # the six math chapters + figures + regen script
    │   ├── 00_expository.md        # narrative companion to chapters 01-05
    │   ├── 01_torus_and_character.md
    │   ├── 02_interpolation_cardinals_rkhs.md
    │   ├── 03_sobolev_distillation.md
    │   ├── 04_dft_pontryagin_trig_identity.md
    │   ├── 05_probes_helix_ablation.md
    │   ├── welsh_nanada_transcript.txt   # Welch Labs / Nanda video transcript
    │   ├── _export_figures.py      # regenerates figures from cache or from scratch
    │   └── figures/                # 9 PNGs + .npz / .eqx caches
    └── sobolev/                    # the 7 notebooks referenced from 00
        ├── sobolev_student_character_periodic.ipynb
        └── grokking/
            ├── modulus_sweep.ipynb
            ├── dynamics_excluded_loss.ipynb
            ├── manifold_and_ablation.ipynb
            ├── fourier_decomp.ipynb
            ├── grokking_baseline_with_decay.ipynb
            └── grokking_capacity_sweep.ipynb
```

## Reading the documentation

Open
[`ml/polys/workbook/surfaces/documentation/00_expository.md`](ml/polys/workbook/surfaces/documentation/00_expository.md)
in any markdown viewer (GitHub renders it natively). All in-doc cross-links
(figures, chapters, source modules, notebooks) resolve inside this repo.

## Regenerating the figures from the committed cache (no GPU needed)

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r ml/polys/requirements-jax-gpu-wsl.txt    # or jax-cpu for CPU-only
cd ml/polys/workbook/surfaces/documentation
PYTHONPATH=../../../.. python _export_figures.py --no-recompute
```

The `--no-recompute` flag forces use of the cached `.npz` (probe data) and
`.eqx` (trained student weights) files under `figures/cache/`. No training
happens.

## Retraining from scratch (needs JAX with a GPU for tractable wall-clock)

```bash
cd ml/polys/workbook/surfaces/documentation
PYTHONPATH=../../../.. python _export_figures.py
```

This trains the six students (4 variants at p=113, 2 architectures at p=17),
caches them as `.eqx` pytrees with JSON sidecars, and re-renders every figure.

## Scope

This repo contains *only* the documentation + the code paths it exercises.
Engineering, infrastructure, and the polynomial / Hermite-Birkhoff teacher
material referenced in the chapters' "Further reading" sections live in the
parent monorepo and are not shipped here. Cross-links inside the .md files to
files that *are* shipped here all resolve; cross-links to siblings of the
`documentation/` folder (`../bivariate/`, `../hermite_birkhoff/`, etc.) point
out of scope.
