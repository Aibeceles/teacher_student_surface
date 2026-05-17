"""Regenerate the nine figures embedded in 00_commentary.md.

Stages
------
1. Extract PNG outputs from already-executed notebooks for figures whose
   visual is best taken verbatim from the existing artefact: F2 (1-D DFT
   spectrum at p=17 / p=113), F5 (excluded-loss curve at p=8), F7
   (helix PC1-PC2 4-panel at p=8), F8 (modular accuracy heatmap at p=17).
2. Recompute the figures whose source data is not directly extractable
   or whose modulus is being upgraded:
   - F1 (linear probes R^2 at p=113)            -- shared baseline_siren_p113 run
   - F3 (2-D FFT channel fractions at p=113)    -- shared baseline_siren_p113 run
   - F6 (causal ablation grouped bar at p=113)  -- four variants at p=113
   - F4 (sum-of-angles histogram at p=17 bottleneck) -- (8, 16) trunk
   - F9 (gradient-field comparison at p=17)     -- canonical (32, 64) trunk
3. Render each figure as a PNG with consistent matplotlib style.

The script is idempotent: trained students are cached as ``.eqx`` pytrees
plus a JSON config sidecar; per-figure data is cached as ``.npz``.
Re-running with all caches present skips both training and probing.

Usage
-----
::

    python _export_figures.py                    # full pipeline
    python _export_figures.py --only=f01,f06    # render only specific figures
    python _export_figures.py --no-recompute    # use cache only; never train
    python _export_figures.py --no-extract      # skip stage 1 too

The script is meant to be run from anywhere; paths are anchored to the
file's parent directory (the documentation/ folder).
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

# matplotlib import is deferred to avoid loading it for `--list` usage.


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------


HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE / "figures"
CACHE_DIR = FIGURES_DIR / "cache"
STUDENTS_P113_DIR = CACHE_DIR / "students_p113"
STUDENTS_P17_DIR = CACHE_DIR / "students_p17"
STUDENTS_P8_DIR = CACHE_DIR / "students_p8"

NB_DIR = HERE.parent  # surfaces/
GROK_DIR = NB_DIR / "sobolev" / "grokking"
SOBOLEV_DIR = NB_DIR / "sobolev"

DPI = 150
PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "tertiary": "#2ca02c",
    "quaternary": "#d62728",
    "muted": "#7f7f7f",
}


# ---------------------------------------------------------------------------
# polys imports (deferred until stage 2 so --no-recompute does not require
# the GPU stack to be present)
# ---------------------------------------------------------------------------


def _polys_root() -> Path:
    here = HERE
    candidates = [
        here.parent.parent.parent / "graphic_zero_character",
        here.parent.parent.parent.parent / "graphic_zero_character",
        here.parent.parent.parent.parent.parent / "graphic_zero_character",
    ]
    for g in candidates:
        if g.is_dir() and (g / "__init__.py").exists():
            return g.parent
    raise FileNotFoundError("Could not locate ml/polys.")


def _ensure_polys_on_path() -> None:
    root = _polys_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


# ---------------------------------------------------------------------------
# matplotlib style
# ---------------------------------------------------------------------------


def _setup_mpl():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    })
    return plt


# ---------------------------------------------------------------------------
# Stage 1 -- extract PNG outputs from notebooks
# ---------------------------------------------------------------------------


def _read_nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_png_from_cell(cell: dict) -> bytes | None:
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        png_b64 = data.get("image/png")
        if png_b64:
            return base64.b64decode(png_b64)
    return None


def _find_cell(nb: dict, cell_id: str) -> dict | None:
    for cell in nb.get("cells", []):
        if cell.get("id") == cell_id:
            return cell
    return None


def _find_cell_with_image(nb: dict, cell_id_prefix: str | None = None) -> dict | None:
    """Find the first cell whose id starts with ``cell_id_prefix`` (or any cell
    if None) that contains a PNG in its outputs."""
    for cell in nb.get("cells", []):
        if cell_id_prefix is not None and not cell.get("id", "").startswith(cell_id_prefix):
            continue
        if _extract_png_from_cell(cell) is not None:
            return cell
    return None


def stage1_extract(only: set[str] | None) -> None:
    extractions = [
        ("f02", "f02_dft_spectrum.png", GROK_DIR / "modulus_sweep.ipynb", "spectrum_plot"),
        ("f05", "f05_excluded_loss_curve.png", GROK_DIR / "dynamics_excluded_loss.ipynb", None),
        ("f07", "f07_helix_pc12_4panel.png", GROK_DIR / "manifold_and_ablation.ipynb", "helix_plot"),
        ("f08", "f08_modular_accuracy_heatmap.png", SOBOLEV_DIR / "sobolev_student_character_periodic.ipynb", "modular"),
    ]

    for fid, out_name, nb_path, cell_id in extractions:
        if only is not None and fid not in only:
            continue
        out_path = FIGURES_DIR / out_name
        if out_path.exists():
            print(f"  [{fid}] already exists at {out_path.name}, skipping")
            continue
        if not nb_path.exists():
            print(f"  [{fid}] notebook missing: {nb_path.name}, skipping")
            continue
        nb = _read_nb(nb_path)
        cell = _find_cell(nb, cell_id) if cell_id else None
        if cell is None:
            cell = _find_cell_with_image(nb)
        if cell is None:
            print(f"  [{fid}] no image cell found in {nb_path.name}, skipping")
            continue
        png = _extract_png_from_cell(cell)
        if png is None:
            print(f"  [{fid}] cell has no PNG in {nb_path.name}, skipping")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png)
        print(f"  [{fid}] extracted {len(png) // 1024} KB -> {out_path.name}")


# ---------------------------------------------------------------------------
# Stage 2 -- training and probe helpers
# ---------------------------------------------------------------------------


def _serialise_student(student, cfg_dict: dict, dest_dir: Path, name: str) -> None:
    import equinox as eqx

    dest_dir.mkdir(parents=True, exist_ok=True)
    eqx_path = dest_dir / f"{name}.eqx"
    cfg_path = dest_dir / f"{name}.config.json"
    eqx.tree_serialise_leaves(eqx_path, student)
    cfg_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")


def _deserialise_student(dest_dir: Path, name: str, make_template):
    import equinox as eqx

    eqx_path = dest_dir / f"{name}.eqx"
    cfg_path = dest_dir / f"{name}.config.json"
    if not eqx_path.exists() or not cfg_path.exists():
        return None, None
    cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    template = make_template(cfg_dict)
    student = eqx.tree_deserialise_leaves(eqx_path, template)
    return student, cfg_dict


def _student_cfg_to_dict(scfg) -> dict:
    return {f.name: getattr(scfg, f.name) for f in dataclasses.fields(scfg)}


def _train_student(
    *,
    p: int,
    mesh_n: int,
    epochs: int,
    ramp: int,
    batch: int,
    student_cfg_overrides: dict,
    weight_decay: float = 0.0,
    weights_overrides: dict | None = None,
    seed: int = 0,
    label: str = "",
):
    """Single training run; returns (student, ds, teacher, train_seconds)."""
    import jax
    import jax.numpy as jnp
    from sobolev_distill_character import (
        CharacterStudentConfig,
        CharacterTrainConfig,
        LinearRampSchedule,
        LossWeights,
        build_character_dataset,
        build_character_teacher_mesh_periodic,
        make_character_student,
        train_student_character_scheduled,
    )
    from sobolev_distill_character.model import CoordMeta

    nodes = np.arange(p, dtype=np.float64)
    teacher = build_character_teacher_mesh_periodic(
        nodes_x=nodes, nodes_y=nodes, p=p, mesh_n=mesh_n, lam=1.0,
    )
    ds = build_character_dataset(teacher)

    base_cfg = dict(
        trunk_hidden=64,
        trunk_depth=3,
        embed_dim=32,
        head_hidden=32,
        activation="siren",
        omega_0=2.5,
        axis_probe=False,
    )
    base_cfg.update(student_cfg_overrides)
    scfg = CharacterStudentConfig(**base_cfg)

    base_weights = dict(
        value=1.0, grad=0.0, hess=0.0, hess_reg=1e-4,
        unit_circle=0.5, axis=0.0, energy_value=0.5,
        energy_pd=0.0, pd_pos_weight=1.0,
    )
    if weights_overrides:
        base_weights.update(weights_overrides)
    weights = LossWeights(**base_weights)

    coord_meta = CoordMeta.from_dataset(ds) if scfg.trunk_kind == "fourier" else None
    student = make_character_student(jax.random.PRNGKey(seed), scfg, coord_meta=coord_meta)

    cfg = CharacterTrainConfig(
        epochs=epochs, batch_size=batch,
        lr_init=1e-3, lr_min=1e-5, weight_decay=weight_decay, grad_clip=1.0,
        lattice_frac=0.4, chebyshev_frac=0.4,
        weights=weights, log_every=max(50, epochs // 20), seed=seed,
    )
    schedule = LinearRampSchedule(
        base=weights, field="grad", start=0.0, end=0.05, ramp_epochs=ramp,
    )

    print(f"    training {label} (p={p}, mesh={mesh_n}, ep={epochs}, batch={batch}) ...", flush=True)
    t0 = time.time()
    student, _hist = train_student_character_scheduled(student, ds, cfg, schedule)
    train_s = time.time() - t0
    print(f"    {label} done in {train_s:.1f}s", flush=True)

    return student, ds, teacher, train_s, scfg, weights, coord_meta


def _config_signature(scfg, weight_decay: float, weights, p: int, mesh_n: int, epochs: int, batch: int, seed: int) -> dict:
    return {
        "scfg": _student_cfg_to_dict(scfg),
        "weight_decay": float(weight_decay),
        "weights": {f.name: float(getattr(weights, f.name)) for f in dataclasses.fields(weights)},
        "p": int(p),
        "mesh_n": int(mesh_n),
        "epochs": int(epochs),
        "batch": int(batch),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Stage 2a / 2b -- p=113 students
# ---------------------------------------------------------------------------


P113 = dict(p=113, mesh_n=128, epochs=6000, ramp=600, batch=512)


VARIANT_CONFIGS = {
    "baseline_siren": dict(
        scfg_overrides={"axis_probe": False},
        weights_overrides={"axis": 0.0},
    ),
    "A_axis_loss": dict(
        scfg_overrides={"axis_probe": True},
        weights_overrides={"axis": 1.0},
    ),
    "B_fourier": dict(
        scfg_overrides={"trunk_kind": "fourier", "fourier_K": 4, "activation": "gelu", "axis_probe": False},
        weights_overrides={"axis": 0.0},
    ),
    "C_factored": dict(
        scfg_overrides={"trunk_kind": "factored", "axis_emb": 16, "activation": "gelu", "axis_probe": False},
        weights_overrides={"axis": 0.0},
    ),
}


def _train_or_load_p113_variant(name: str, *, no_recompute: bool):
    """Train (or load from cache) one variant at p=113."""
    eqx_path = STUDENTS_P113_DIR / f"{name}.eqx"
    cfg_path = STUDENTS_P113_DIR / f"{name}.config.json"

    cfg = VARIANT_CONFIGS[name]

    if eqx_path.exists() and cfg_path.exists():
        print(f"  [variant {name}] cache hit; loading", flush=True)
        return _load_cached_p113(name)

    if no_recompute:
        raise SystemExit(f"--no-recompute set but variant {name} not cached at {eqx_path}")

    student, ds, teacher, train_s, scfg, weights, coord_meta = _train_student(
        p=P113["p"], mesh_n=P113["mesh_n"],
        epochs=P113["epochs"], ramp=P113["ramp"], batch=P113["batch"],
        student_cfg_overrides=cfg["scfg_overrides"],
        weights_overrides=cfg["weights_overrides"],
        weight_decay=0.0,
        seed=0,
        label=f"{name}_p113",
    )
    sig = _config_signature(scfg, 0.0, weights, P113["p"], P113["mesh_n"], P113["epochs"], P113["batch"], 0)
    sig["train_seconds"] = train_s
    _serialise_student(student, sig, STUDENTS_P113_DIR, name)
    return student, ds, teacher, sig, coord_meta


def _load_cached_p113(name: str):
    """Reload a cached p=113 student plus its dataset + teacher (rebuilt deterministically)."""
    import jax
    from sobolev_distill_character import (
        CharacterStudentConfig,
        build_character_dataset,
        build_character_teacher_mesh_periodic,
        make_character_student,
    )
    from sobolev_distill_character.model import CoordMeta

    cfg_path = STUDENTS_P113_DIR / f"{name}.config.json"
    sig = json.loads(cfg_path.read_text(encoding="utf-8"))
    scfg = CharacterStudentConfig(**sig["scfg"])

    nodes = np.arange(sig["p"], dtype=np.float64)
    teacher = build_character_teacher_mesh_periodic(
        nodes_x=nodes, nodes_y=nodes, p=sig["p"], mesh_n=sig["mesh_n"], lam=1.0,
    )
    ds = build_character_dataset(teacher)
    coord_meta = CoordMeta.from_dataset(ds) if scfg.trunk_kind == "fourier" else None
    template = make_character_student(jax.random.PRNGKey(sig["seed"]), scfg, coord_meta=coord_meta)

    import equinox as eqx
    student = eqx.tree_deserialise_leaves(STUDENTS_P113_DIR / f"{name}.eqx", template)
    return student, ds, teacher, sig, coord_meta


# ---------------------------------------------------------------------------
# Figure data caching: per-figure .npz with whatever arrays / scalars are
# needed to render
# ---------------------------------------------------------------------------


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_npz(path: Path) -> dict:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


# ---------------------------------------------------------------------------
# Per-figure data computation
# ---------------------------------------------------------------------------


def compute_f01(no_recompute: bool) -> dict:
    """F1 -- linear probes R^2 at p=113."""
    cache_path = CACHE_DIR / "f01_linear_probes_p113.npz"
    cached = _load_npz(cache_path)
    if cached:
        return cached
    if no_recompute:
        raise SystemExit("--no-recompute set but f01 cache missing")
    from sobolev_distill_character import compute_latents_character, linear_probes_character

    student, ds, teacher, sig, _ = _train_or_load_p113_variant("baseline_siren", no_recompute=no_recompute)
    latents = compute_latents_character(student, ds, teacher)
    probes = linear_probes_character(latents)
    targets = list(probes.r2.keys())
    r2 = np.array([probes.r2[t] for t in targets], dtype=np.float64)
    data = {"targets": np.array(targets), "r2": r2, "p": np.array([sig["p"]])}
    _save_npz(cache_path, **data)
    return data


def compute_f03(no_recompute: bool) -> dict:
    """F3 -- 2-D FFT channel fractions at p=113."""
    cache_path = CACHE_DIR / "f03_channel_fractions_p113.npz"
    cached = _load_npz(cache_path)
    if cached:
        return cached
    if no_recompute:
        raise SystemExit("--no-recompute set but f03 cache missing")
    from sobolev_distill_character import fft2_neuron_surface

    student, ds, teacher, sig, _ = _train_or_load_p113_variant("baseline_siren", no_recompute=no_recompute)
    fft2 = fft2_neuron_surface(student, ds, teacher)
    keys = ("cos_cos", "sin_sin", "cos_sin", "sin_cos")
    totals = {k: float(fft2.channel_energy[k].sum()) for k in keys}
    norm = sum(totals.values()) or 1.0
    fracs = np.array([totals[k] / norm for k in keys], dtype=np.float64)
    data = {"channel_keys": np.array(keys), "fractions": fracs, "p": np.array([sig["p"]])}
    _save_npz(cache_path, **data)
    return data


def compute_f06(no_recompute: bool) -> dict:
    """F6 -- causal ablation grouped bar at p=113.

    Four variants x four ablations: baseline, readout cos/sin axes, random
    matched-norm, top-2 PC of helix.
    """
    cache_path = CACHE_DIR / "f06_ablation_grouped_bar_p113.npz"
    cached = _load_npz(cache_path)
    if cached:
        return cached
    if no_recompute:
        raise SystemExit("--no-recompute set but f06 cache missing")
    from sobolev_distill_character import (
        ablate_subspace_and_score,
        compute_latents_character,
        helix_pca,
        linear_probes_character,
    )

    variants = ["baseline_siren", "A_axis_loss", "B_fourier", "C_factored"]
    rows = {v: {} for v in variants}
    rng = np.random.default_rng(0)

    for v in variants:
        student, ds, teacher, sig, _ = _train_or_load_p113_variant(v, no_recompute=no_recompute)
        latents = compute_latents_character(student, ds, teacher)
        probes_rep = linear_probes_character(latents)
        D = int(latents.H_lat.shape[1])

        coef_keys = ("Re zeta^i", "Im zeta^i", "Re zeta^j", "Im zeta^j")
        readout_rows = []
        for k in coef_keys:
            coef = probes_rep.coefficients.get(k)
            if coef is None:
                continue
            coef = coef[:-1]  # drop bias
            if np.linalg.norm(coef) > 1e-12:
                readout_rows.append(coef.astype(np.float64))
        readout_subspace = np.stack(readout_rows, axis=0) if readout_rows else np.zeros((0, D))
        rand_subspace = rng.normal(size=readout_subspace.shape) if readout_subspace.size > 0 else np.zeros((0, D))

        rep_i = helix_pca(latents, axis="i", n_components=6)
        pc_subspace = rep_i.components[:2]

        rep_readout = ablate_subspace_and_score(student, ds, teacher, subspace=readout_subspace, name=f"{v}/readout")
        rep_rand = ablate_subspace_and_score(student, ds, teacher, subspace=rand_subspace, name=f"{v}/rand")
        rep_pc = ablate_subspace_and_score(student, ds, teacher, subspace=pc_subspace, name=f"{v}/pc12")

        rows[v] = {
            "baseline": float(rep_readout.baseline_modular_accuracy),
            "readout": float(rep_readout.ablated_modular_accuracy),
            "random":  float(rep_rand.ablated_modular_accuracy),
            "pc12":    float(rep_pc.ablated_modular_accuracy),
        }
        print(f"  [F6] {v}: baseline={rows[v]['baseline']:.3f} readout={rows[v]['readout']:.3f} "
              f"random={rows[v]['random']:.3f} pc12={rows[v]['pc12']:.3f}", flush=True)

    cols = ["baseline", "readout", "random", "pc12"]
    arr = np.array([[rows[v][c] for c in cols] for v in variants], dtype=np.float64)
    data = {"variants": np.array(variants), "columns": np.array(cols), "values": arr,
            "p": np.array([P113["p"]])}
    _save_npz(cache_path, **data)
    return data


def _train_or_load_p17_bottleneck(no_recompute: bool):
    """The (8, 16) bottleneck student at p=17 used for F4 (sum-of-angles)."""
    eqx_path = STUDENTS_P17_DIR / "bottleneck_8x16.eqx"
    cfg_path = STUDENTS_P17_DIR / "bottleneck_8x16.config.json"

    if eqx_path.exists() and cfg_path.exists():
        print("  [p17 bottleneck (8, 16)] cache hit; loading", flush=True)
        return _load_cached_p17_bottleneck()

    if no_recompute:
        raise SystemExit("--no-recompute set but p17 bottleneck student not cached")

    student, ds, teacher, train_s, scfg, weights, _ = _train_student(
        p=17, mesh_n=64, epochs=8000, ramp=800, batch=256,
        student_cfg_overrides={"embed_dim": 8, "trunk_hidden": 16, "axis_probe": False},
        weights_overrides={"axis": 0.0},
        weight_decay=0.0, seed=0, label="p17_bottleneck",
    )
    sig = _config_signature(scfg, 0.0, weights, 17, 64, 8000, 256, 0)
    sig["train_seconds"] = train_s
    _serialise_student(student, sig, STUDENTS_P17_DIR, "bottleneck_8x16")
    return student, ds, teacher, sig


def _load_cached_p17_bottleneck():
    import jax
    from sobolev_distill_character import (
        CharacterStudentConfig,
        build_character_dataset,
        build_character_teacher_mesh_periodic,
        make_character_student,
    )

    cfg_path = STUDENTS_P17_DIR / "bottleneck_8x16.config.json"
    sig = json.loads(cfg_path.read_text(encoding="utf-8"))
    scfg = CharacterStudentConfig(**sig["scfg"])

    nodes = np.arange(sig["p"], dtype=np.float64)
    teacher = build_character_teacher_mesh_periodic(
        nodes_x=nodes, nodes_y=nodes, p=sig["p"], mesh_n=sig["mesh_n"], lam=1.0,
    )
    ds = build_character_dataset(teacher)
    template = make_character_student(jax.random.PRNGKey(sig["seed"]), scfg, coord_meta=None)

    import equinox as eqx
    student = eqx.tree_deserialise_leaves(STUDENTS_P17_DIR / "bottleneck_8x16.eqx", template)
    return student, ds, teacher, sig


def compute_f04(no_recompute: bool) -> dict:
    """F4 -- per-neuron sum-of-angles histogram at p=17 bottleneck."""
    cache_path = CACHE_DIR / "f04_soa_histogram_p17_bottleneck.npz"
    cached = _load_npz(cache_path)
    if cached:
        return cached
    if no_recompute:
        raise SystemExit("--no-recompute set but f04 cache missing")
    from sobolev_distill_character import fft2_neuron_surface

    student, ds, teacher, sig = _train_or_load_p17_bottleneck(no_recompute=no_recompute)
    fft2 = fft2_neuron_surface(student, ds, teacher)
    soa = np.asarray(fft2.sum_of_angles_score)
    data = {"soa": soa, "median": np.median(soa),
            "frac_gt_half": float((soa > 0.5).mean()),
            "p": np.array([sig["p"]])}
    _save_npz(cache_path, **data)
    return data


def _train_or_load_p17_canonical(no_recompute: bool):
    """The canonical (32, 64) baseline_siren student at p=17 used for F9."""
    eqx_path = STUDENTS_P17_DIR / "canonical_32x64.eqx"
    cfg_path = STUDENTS_P17_DIR / "canonical_32x64.config.json"

    if eqx_path.exists() and cfg_path.exists():
        print("  [p17 canonical (32, 64)] cache hit; loading", flush=True)
        return _load_cached_p17_canonical()

    if no_recompute:
        raise SystemExit("--no-recompute set but p17 canonical student not cached")

    student, ds, teacher, train_s, scfg, weights, _ = _train_student(
        p=17, mesh_n=64, epochs=8000, ramp=800, batch=256,
        student_cfg_overrides={"axis_probe": False},
        weights_overrides={"axis": 0.0},
        weight_decay=0.0, seed=0, label="p17_canonical",
    )
    sig = _config_signature(scfg, 0.0, weights, 17, 64, 8000, 256, 0)
    sig["train_seconds"] = train_s
    _serialise_student(student, sig, STUDENTS_P17_DIR, "canonical_32x64")
    return student, ds, teacher, sig


def _load_cached_p17_canonical():
    import jax
    from sobolev_distill_character import (
        CharacterStudentConfig,
        build_character_dataset,
        build_character_teacher_mesh_periodic,
        make_character_student,
    )

    cfg_path = STUDENTS_P17_DIR / "canonical_32x64.config.json"
    sig = json.loads(cfg_path.read_text(encoding="utf-8"))
    scfg = CharacterStudentConfig(**sig["scfg"])

    nodes = np.arange(sig["p"], dtype=np.float64)
    teacher = build_character_teacher_mesh_periodic(
        nodes_x=nodes, nodes_y=nodes, p=sig["p"], mesh_n=sig["mesh_n"], lam=1.0,
    )
    ds = build_character_dataset(teacher)
    template = make_character_student(jax.random.PRNGKey(sig["seed"]), scfg, coord_meta=None)

    import equinox as eqx
    student = eqx.tree_deserialise_leaves(STUDENTS_P17_DIR / "canonical_32x64.eqx", template)
    return student, ds, teacher, sig


def compute_f09(no_recompute: bool) -> dict:
    """F9 -- gradient field comparison: |grad T|, |grad f|, |grad T - grad f| at p=17."""
    cache_path = CACHE_DIR / "f09_grad_field_p17.npz"
    cached = _load_npz(cache_path)
    if cached:
        return cached
    if no_recompute:
        raise SystemExit("--no-recompute set but f09 cache missing")

    import jax
    import jax.numpy as jnp
    from sobolev_distill_character.model import f_arith_character

    student, ds, teacher, sig = _train_or_load_p17_canonical(no_recompute=no_recompute)
    p = int(sig["p"])

    # Mesh of (x, y) raw coords on T^2 (in [0, p)^2), then normalise to feed the trunk.
    nx = ny = 64
    xs = np.linspace(0, p, nx, endpoint=False, dtype=np.float64)
    ys = np.linspace(0, p, ny, endpoint=False, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    raw = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    norm = ds.norm
    cx = np.array([norm.x_center, norm.y_center], dtype=np.float64)
    sc = np.array([norm.x_scale, norm.y_scale], dtype=np.float64)
    xy_norm = ((raw - cx) / sc).astype(np.float32)

    # Teacher analytic gradient: T(x,y) = exp(2 pi i (x+y)/p), so
    #   d/dx Re T = -(2 pi / p) sin(2 pi (x+y) / p) = d/dy Re T,
    #   d/dx Im T =  (2 pi / p) cos(2 pi (x+y) / p) = d/dy Im T.
    omega = 2.0 * math.pi / p
    s = np.sin(omega * (XX + YY))
    c = np.cos(omega * (XX + YY))
    grad_T_re = np.stack([-omega * s, -omega * s], axis=-1)  # shape (nx, ny, 2)
    grad_T_im = np.stack([ omega * c,  omega * c], axis=-1)
    grad_T_norm = np.sqrt((grad_T_re ** 2).sum(axis=-1) + (grad_T_im ** 2).sum(axis=-1))

    # Student gradient via autodiff on f_arith_character (returns (Re, Im) on
    # NORMALISED inputs; the student stores its own normalisation, so we must
    # un-normalise the gradient by 1/scale for comparison with the analytic teacher).
    def _re_norm(s, xy):
        return f_arith_character(s, xy)[0]

    def _im_norm(s, xy):
        return f_arith_character(s, xy)[1]

    g_re = jax.vmap(jax.grad(lambda xy: _re_norm(student, xy)))(jnp.asarray(xy_norm))
    g_im = jax.vmap(jax.grad(lambda xy: _im_norm(student, xy)))(jnp.asarray(xy_norm))
    g_re = np.asarray(g_re).reshape(nx, ny, 2)
    g_im = np.asarray(g_im).reshape(nx, ny, 2)
    # Un-normalise: d/d(x_raw) = (1/scale) d/d(x_norm)
    sc_arr = np.array([norm.x_scale, norm.y_scale], dtype=np.float64)
    g_re = g_re / sc_arr[None, None, :]
    g_im = g_im / sc_arr[None, None, :]

    # Un-normalise the value, too: predicted (Re, Im) lives in normalised units;
    # the gradient relation has the same scale factor, so once we divide by sc
    # above we are in raw-units gradient of normalised value. We separately need
    # to multiply by the value-normalisation std.
    g_re = g_re * float(norm.v_re_std)
    g_im = g_im * float(norm.v_im_std)
    grad_f_norm = np.sqrt((g_re ** 2).sum(axis=-1) + (g_im ** 2).sum(axis=-1))
    grad_diff_norm = np.sqrt(((g_re - grad_T_re) ** 2).sum(axis=-1) + ((g_im - grad_T_im) ** 2).sum(axis=-1))

    data = {
        "xs": xs, "ys": ys,
        "grad_T_norm": grad_T_norm, "grad_f_norm": grad_f_norm, "grad_diff_norm": grad_diff_norm,
        "p": np.array([p]),
    }
    _save_npz(cache_path, **data)
    return data


# ---------------------------------------------------------------------------
# Stage 3 -- rendering
# ---------------------------------------------------------------------------


def render_f01(plt, data: dict) -> Path:
    targets = [str(t) for t in data["targets"]]
    r2 = data["r2"]
    p = int(data["p"][0])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = np.arange(len(targets))
    bars = ax.barh(y, r2, color=PALETTE["primary"], edgecolor="black", linewidth=0.5)
    ax.axvline(1.0, color=PALETTE["muted"], linestyle="--", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(targets)
    ax.invert_yaxis()
    ax.set_xlabel(r"linear-probe test $R^2$")
    ax.set_xlim(min(min(r2), 0.0) - 0.05, 1.05)
    ax.set_title(f"F1: linear probes for canonical Fourier features (p = {p}, baseline_siren)")
    for yi, ri in zip(y, r2):
        ax.text(ri + 0.01, yi, f"{ri:+.3f}", va="center", fontsize=8)
    fig.tight_layout()
    out = FIGURES_DIR / "f01_linear_probes_R2.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def render_f03(plt, data: dict) -> Path:
    keys = [str(k) for k in data["channel_keys"]]
    fracs = data["fractions"]
    p = int(data["p"][0])

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["tertiary"], PALETTE["quaternary"]]
    bars = ax.bar(keys, fracs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("channel-energy fraction")
    ax.set_ylim(0, 1)
    ax.set_title(f"F3: 2-D FFT channel decomposition (p = {p}, baseline_siren, axis-loss-off)")
    for k, fr in zip(keys, fracs):
        ax.text(k, fr + 0.02, f"{fr:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = FIGURES_DIR / "f03_fft_channel_fractions.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def render_f06(plt, data: dict) -> Path:
    variants = [str(v) for v in data["variants"]]
    cols = [str(c) for c in data["columns"]]
    arr = data["values"]
    p = int(data["p"][0])

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    n_var = len(variants); n_col = len(cols)
    width = 0.18
    x = np.arange(n_var)
    palette = [PALETTE["muted"], PALETTE["quaternary"], PALETTE["tertiary"], PALETTE["secondary"]]
    for j, c in enumerate(cols):
        offsets = (j - (n_col - 1) / 2) * width
        ax.bar(x + offsets, arr[:, j], width, label=c, color=palette[j], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("modular accuracy after ablation")
    ax.axhline(1.0, color=PALETTE["muted"], linestyle="--", linewidth=0.7)
    ax.set_title(f"F6: causal subspace ablation across variants (p = {p})")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    out = FIGURES_DIR / "f06_causal_ablation_grouped_bar.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def render_f04(plt, data: dict) -> Path:
    soa = data["soa"]
    median = float(data["median"])
    frac = float(data["frac_gt_half"])
    p = int(data["p"][0])

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(soa, bins=24, range=(0, 1), color=PALETTE["primary"], edgecolor="black", linewidth=0.5)
    ax.axvline(median, color=PALETTE["quaternary"], linestyle="--", linewidth=1.2,
               label=f"median = {median:.2f}")
    ax.set_xlabel("sum-of-angles score")
    ax.set_ylabel("neuron count")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left")
    ax.set_title(f"F4: per-neuron trig-identity score at p = {p} bottleneck (embed_dim=8, trunk_hidden=16); "
                 f"frac > 0.5 = {frac:.2f}")
    fig.tight_layout()
    out = FIGURES_DIR / "f04_soa_histogram_bottleneck.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def render_f09(plt, data: dict) -> Path:
    xs = data["xs"]; ys = data["ys"]
    gT = data["grad_T_norm"]; gF = data["grad_f_norm"]; gD = data["grad_diff_norm"]
    p = int(data["p"][0])
    extent = [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())]

    vmax = float(max(gT.max(), gF.max()))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    titles = [r"$|\nabla T|$ (analytic teacher)",
              r"$|\nabla f_\theta|$ (student)",
              r"$|\nabla T - \nabla f_\theta|$ (mismatch, same scale)"]
    fields = [gT, gF, gD]
    for ax, title, field in zip(axes, titles, fields):
        im = ax.imshow(field.T, origin="lower", extent=extent, aspect="auto",
                       cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(title)
        ax.text(0.03, 0.97,
                f"max = {float(field.max()):.3f}\nmean = {float(field.mean()):.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.55))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"F9: gradient-field comparison at p = {p} (canonical 32x64 baseline_siren)")
    fig.tight_layout()
    out = FIGURES_DIR / "f09_grad_field_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


COMPUTE_RENDER = {
    "f01": (compute_f01, render_f01),
    "f03": (compute_f03, render_f03),
    "f04": (compute_f04, render_f04),
    "f06": (compute_f06, render_f06),
    "f09": (compute_f09, render_f09),
}

EXTRACTED_NO_RENDER = {"f02", "f05", "f07", "f08"}
ALL_FIG_IDS = sorted(set(COMPUTE_RENDER.keys()) | EXTRACTED_NO_RENDER)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="comma-separated figure ids to render (default: all)")
    parser.add_argument("--no-recompute", action="store_true",
                        help="never run training; fail if cache is missing")
    parser.add_argument("--no-extract", action="store_true",
                        help="skip stage 1 PNG extraction from notebooks")
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None
    if only is not None:
        only = {f.strip() for f in only}

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"figures dir: {FIGURES_DIR}")
    print(f"cache dir:   {CACHE_DIR}")
    print(f"only:        {only or 'all'}")
    print(f"no-recompute: {args.no_recompute}")
    print()

    t_start = time.time()

    # Stage 1: extract from notebooks (no compute, no GPU)
    if not args.no_extract:
        print("=== Stage 1: extract PNGs from notebooks ===")
        stage1_extract(only)
        print()

    # Stage 2 + 3: compute + render
    needed = [f for f in COMPUTE_RENDER if (only is None or f in only)]
    if needed:
        print("=== Stage 2 + 3: compute and render figures ===")
        _ensure_polys_on_path()
        plt = _setup_mpl()
        # Print JAX device info up front so the user sees the GPU pickup
        try:
            import jax
            print(f"jax devices: {jax.devices()}")
        except Exception as exc:
            print(f"jax import failed: {exc}")
            raise

        for fid in needed:
            print(f"--- {fid} ---")
            compute_fn, render_fn = COMPUTE_RENDER[fid]
            data = compute_fn(args.no_recompute)
            out = render_fn(plt, data)
            print(f"  rendered -> {out.name} ({out.stat().st_size // 1024} KB)")
            print()

    elapsed = time.time() - t_start
    print(f"total elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
