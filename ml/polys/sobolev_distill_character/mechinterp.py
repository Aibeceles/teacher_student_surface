"""Mechanistic-interpretability probes for the character Sobolev student.

Follow-up to the structural probes in :mod:`sobolev_distill_character.probes`.
Where ``probes.py`` answers "did the student fit?", this module answers
"*how* did the student fit?" -- the Nanda-style Fourier-feature picture.

Public surface:

- :func:`dft_trunk_along_axis` -- per-neuron 1-D DFT of trunk activations
  along one input axis with the other held at zero (mirrors the "fix y=0,
  sweep x" plot from the Welch Labs / Nanda video).
- :func:`fft2_neuron_surface` -- 2-D FFT of one (or every) neuron's lattice
  ``(x, y)`` surface, decomposed into the four product channels
  ``{cos x cos y, sin x sin y, cos x sin y, sin x cos y}`` so the
  sum-of-angles identity is testable as a magnitude ratio.
- :func:`excluded_loss_at_freqs` -- Nanda excluded-loss replay. Projects
  the named Fourier modes out of either the readout's expansion of the
  ``(p, p)`` truth table or the trunk-latent matrix ``H_lat`` and
  re-evaluates value MSE / modular accuracy.
- :func:`helix_pca` -- PCA on ``H_lat`` ordered by one axis index;
  reports angular wrap and helix-fit R^2 against the canonical
  ``(cos 2 pi k i / p, sin 2 pi k i / p)`` curves.
- :func:`ablate_subspace_and_score` -- generalised projection-out
  ablation. Removes a user-supplied subspace from the trunk output and
  pushes through ``head_a`` to compute modular accuracy and value MSE.

All helpers are pure numpy / JAX-numpy, consume the existing
``CharacterStudent`` / ``CharacterSobolevDataset`` / ``CharacterMeshTeacher``
tuple or the cached :class:`CharacterLatentArrays`, and do not mutate the
student.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .dataset import CharacterSobolevDataset
from .model import CharacterStudent, f_arith_character_batched
from .probes import CharacterLatentArrays, compute_latents_character
from .teacher import CharacterMeshTeacher


# ---------------------------------------------------------------------------
# 1) 1-D DFT of trunk neurons along one axis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DFTReport:
    """Per-neuron 1-D DFT spectrum along one input axis.

    The trunk is evaluated at the lattice points ``(i, 0)`` for ``axis=0``
    (sweep x with y held at zero) or ``(0, j)`` for ``axis=1`` (sweep y
    with x held at zero).  ``spectrum[n, k]`` is the standard ``np.fft.fft``
    coefficient for neuron ``n`` at frequency index ``k`` (length ``p``).

    Attributes
    ----------
    axis
        ``0`` for x-sweep, ``1`` for y-sweep.
    p
        Modulus.
    H_along_axis
        ``(p, D)`` raw trunk activations along the chosen axis at lattice
        points, in modulus order ``i = 0..p-1``.
    spectrum
        ``(D, p)`` complex DFT spectrum per neuron.
    magnitude
        ``|spectrum|`` -- absolute value, ``(D, p)``.
    top_modes
        ``(D,)`` int array of the dominant non-DC frequency index for
        each neuron (argmax of ``magnitude[:, 1:p//2 + 1]`` shifted by 1
        for length-``p`` parity).
    mode_concentration
        ``(D,)`` array of energy ratio of the dominant non-DC mode to the
        total non-DC energy.  Close to 1 means single-frequency neuron.
    dominant_freq_histogram
        ``(p // 2 + 1,)`` int array counting how many neurons peak at
        each non-DC frequency bin in ``1..p//2``.
    """

    axis: int
    p: int
    H_along_axis: np.ndarray  # (p, D)
    spectrum: np.ndarray      # (D, p) complex
    magnitude: np.ndarray     # (D, p)
    top_modes: np.ndarray     # (D,) int
    mode_concentration: np.ndarray  # (D,)
    dominant_freq_histogram: np.ndarray  # (p // 2 + 1,) int


def _trunk_along_axis_raw(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    teacher: CharacterMeshTeacher,
    *,
    axis: int,
) -> np.ndarray:
    """Evaluate ``student.trunk`` at lattice points along one axis.

    For ``axis == 0`` we form points ``(i, nodes_y[0])`` for ``i = 0..p-1``;
    for ``axis == 1`` we form points ``(nodes_x[0], j)``.  Returned in
    *normalised* input coords because that is what the trunk expects.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    p = int(teacher.modulus)
    nodes_x = np.asarray(teacher.nodes_x, dtype=np.float64)
    nodes_y = np.asarray(teacher.nodes_y, dtype=np.float64)
    if axis == 0:
        xs = nodes_x[:p]
        ys = np.full((p,), nodes_y[0], dtype=np.float64)
    else:
        xs = np.full((p,), nodes_x[0], dtype=np.float64)
        ys = nodes_y[:p]
    raw = np.stack([xs, ys], axis=-1)
    norm = dataset.norm
    cx = np.array([norm.x_center, norm.y_center], dtype=np.float64)
    sc = np.array([norm.x_scale, norm.y_scale], dtype=np.float64)
    xy_norm = ((raw - cx) / sc).astype(np.float32)
    trunk_fn = jax.jit(jax.vmap(student.trunk))
    return np.asarray(trunk_fn(jnp.asarray(xy_norm)))  # (p, D)


def dft_trunk_along_axis(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    teacher: CharacterMeshTeacher,
    *,
    axis: int,
) -> DFTReport:
    """Per-neuron 1-D DFT of the trunk activations along one input axis.

    The trunk is queried at the lattice ``(i, 0)`` (or ``(0, j)``); the
    DFT is taken across the ``p`` lattice values per neuron.  The result
    surfaces which discrete frequencies ``2 pi k / p`` each neuron picked
    up after training, mirroring the central plot in Nanda's modular
    arithmetic analysis.
    """
    H = _trunk_along_axis_raw(student, dataset, teacher, axis=axis)  # (p, D)
    p = int(teacher.modulus)
    if H.shape[0] != p:
        raise RuntimeError(f"expected p={p} samples along axis, got {H.shape[0]}")
    D = int(H.shape[1])
    spectrum = np.fft.fft(H, axis=0).T  # (D, p)
    magnitude = np.abs(spectrum)
    nyq = p // 2
    if nyq < 1:
        top_modes = np.zeros((D,), dtype=np.int64)
        mode_conc = np.zeros((D,), dtype=np.float64)
    else:
        slice_mag = magnitude[:, 1 : nyq + 1]  # (D, nyq)
        local_argmax = np.argmax(slice_mag, axis=1)
        top_modes = (local_argmax + 1).astype(np.int64)
        peak_e = slice_mag[np.arange(D), local_argmax] ** 2
        total_e = (slice_mag ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            mode_conc = np.where(total_e > 1e-30, peak_e / np.maximum(total_e, 1e-30), 0.0)
    hist = np.bincount(top_modes, minlength=nyq + 1)
    if hist.size > nyq + 1:
        hist = hist[: nyq + 1]
    return DFTReport(
        axis=int(axis),
        p=p,
        H_along_axis=H,
        spectrum=spectrum,
        magnitude=magnitude,
        top_modes=top_modes,
        mode_concentration=mode_conc.astype(np.float64),
        dominant_freq_histogram=hist.astype(np.int64),
    )


# ---------------------------------------------------------------------------
# 2) 2-D FFT of one neuron's (x, y) lattice surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Surface2DReport:
    """2-D FFT decomposition of a trunk neuron's lattice surface.

    For each neuron index ``n``, the trunk output ``H[i, j, n]`` over the
    ``p x p`` lattice is FFT'd in both axes; the per-frequency complex
    coefficients are then projected onto the four real product channels
    ``{cos x cos y, sin x sin y, cos x sin y, sin x cos y}``.

    Attributes
    ----------
    H_grid
        ``(p, p, D)`` lattice activations.
    spectrum
        ``(D, p, p)`` complex 2-D DFT.
    channel_energy
        Mapping ``{"cos_cos", "sin_sin", "cos_sin", "sin_cos"}`` to an
        ``(D, p // 2 + 1, p // 2 + 1)`` real array giving the energy of
        each ``(kx, ky)`` mode pair in that channel.  ``kx = 0`` / ``ky = 0``
        rows / cols correspond to the DC component on that axis.
    sum_of_angles_score
        ``(D,)`` array of
        ``min(cos_cos, sin_sin) / max(cos_cos, sin_sin, 1e-30)`` summed
        over the top non-DC ``(kx, ky)`` mode pair, where larger means
        "closer to the cos(x+y) trig identity".  Per-neuron diagnostic.
    top_mode
        ``(D, 2)`` int array of the dominant non-DC ``(kx, ky)`` pair per
        neuron.
    """

    H_grid: np.ndarray
    spectrum: np.ndarray
    channel_energy: dict[str, np.ndarray]
    sum_of_angles_score: np.ndarray
    top_mode: np.ndarray


def _trunk_on_lattice_grid(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    teacher: CharacterMeshTeacher,
) -> np.ndarray:
    """Evaluate ``student.trunk`` on the full ``p x p`` lattice grid.

    Returns a ``(p, p, D)`` numpy array indexed by ``(i, j, neuron)``.
    """
    p = int(teacher.modulus)
    nodes_x = np.asarray(teacher.nodes_x, dtype=np.float64)[:p]
    nodes_y = np.asarray(teacher.nodes_y, dtype=np.float64)[:p]
    XX, YY = np.meshgrid(nodes_x, nodes_y, indexing="ij")
    raw = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    norm = dataset.norm
    cx = np.array([norm.x_center, norm.y_center], dtype=np.float64)
    sc = np.array([norm.x_scale, norm.y_scale], dtype=np.float64)
    xy_norm = ((raw - cx) / sc).astype(np.float32)
    trunk_fn = jax.jit(jax.vmap(student.trunk))
    H_flat = np.asarray(trunk_fn(jnp.asarray(xy_norm)))   # (p*p, D)
    D = int(H_flat.shape[1])
    return H_flat.reshape(p, p, D)


def fft2_neuron_surface(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    teacher: CharacterMeshTeacher,
    *,
    neuron_idx: int | None = None,
) -> Surface2DReport:
    """2-D FFT of each (or one) trunk neuron's lattice ``(x, y)`` surface.

    ``neuron_idx`` is honoured only by ``top_mode`` ranking (the report
    always contains every neuron).  The energy split across the four
    product channels follows directly from the parity of the (kx, ky)
    coefficient and the identity
    ``H[i, j] = a cos(kx i) cos(ky j) + b sin(kx i) sin(ky j)``
    ``       + c cos(kx i) sin(ky j) + d sin(kx i) cos(ky j)``
    with ``a, b, c, d`` extracted from the symmetric / antisymmetric parts
    of ``spectrum[kx, ky]``.
    """
    H_grid = _trunk_on_lattice_grid(student, dataset, teacher)  # (p, p, D)
    p = int(teacher.modulus)
    D = int(H_grid.shape[-1])
    if H_grid.shape[:2] != (p, p):
        raise RuntimeError(
            f"expected (p, p, D) lattice grid; got {H_grid.shape}"
        )

    # FFT independently per neuron.
    H_pn = H_grid.transpose(2, 0, 1)              # (D, p, p)
    spec = np.fft.fft2(H_pn, axes=(1, 2))         # (D, p, p) complex
    nyq = p // 2
    K = nyq + 1

    # For each (kx, ky) with kx, ky in [0, nyq], combine spec[kx, ky] and its
    # reflection partners to recover the four real product channels.
    # For real input,  Re(c1) + Re(c2) maps to cos cos;
    # Re(c1) - Re(c2) maps to sin sin;
    # Im(c1) + Im(c2) maps to cos sin;
    # -Im(c1) + Im(c2) maps to sin cos.  Where c1 = spec[kx, ky],
    # c2 = spec[kx, -ky].  Indices are taken mod p.
    channel_keys = ("cos_cos", "sin_sin", "cos_sin", "sin_cos")
    channel_energy: dict[str, np.ndarray] = {
        k: np.zeros((D, K, K), dtype=np.float64) for k in channel_keys
    }

    # Use real-valued amplitudes: divide by the right number for DC vs interior.
    # We treat channel "energy" as squared real coefficients normalised by p**2.
    for kx in range(K):
        for ky in range(K):
            c1 = spec[:, kx, ky]
            c2 = spec[:, kx, (-ky) % p]
            re1, im1 = c1.real, c1.imag
            re2, im2 = c2.real, c2.imag
            cc = 0.5 * (re1 + re2) / (p * p)
            ss = 0.5 * (re2 - re1) / (p * p)
            cs = 0.5 * (im1 + im2) / (p * p)
            sc = 0.5 * (im2 - im1) / (p * p)
            channel_energy["cos_cos"][:, kx, ky] = cc * cc
            channel_energy["sin_sin"][:, kx, ky] = ss * ss
            channel_energy["cos_sin"][:, kx, ky] = cs * cs
            channel_energy["sin_cos"][:, kx, ky] = sc * sc

    # Top non-DC (kx, ky) per neuron (sum across channels).
    total = sum(channel_energy[k] for k in channel_keys)  # (D, K, K)
    total_nodc = total.copy()
    total_nodc[:, 0, 0] = 0.0
    flat_idx = total_nodc.reshape(D, -1).argmax(axis=1)
    kx_top, ky_top = np.unravel_index(flat_idx, (K, K))
    top_mode = np.stack([kx_top, ky_top], axis=-1).astype(np.int64)

    # sum-of-angles score at the dominant mode per neuron.
    cc_top = channel_energy["cos_cos"][np.arange(D), kx_top, ky_top]
    ss_top = channel_energy["sin_sin"][np.arange(D), kx_top, ky_top]
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.maximum(cc_top + ss_top, 1e-30)
        score = (2.0 * np.sqrt(cc_top * ss_top)) / denom
        score = np.where(cc_top + ss_top > 1e-30, score, 0.0)

    return Surface2DReport(
        H_grid=H_grid,
        spectrum=spec,
        channel_energy=channel_energy,
        sum_of_angles_score=score.astype(np.float64),
        top_mode=top_mode,
    )


# ---------------------------------------------------------------------------
# 3) Excluded-loss replay (Nanda-style)
# ---------------------------------------------------------------------------


def _denormalise_predictions(
    pred_norm: np.ndarray, dataset: CharacterSobolevDataset
) -> np.ndarray:
    """``(N, 2)`` normalised ``(re, im)`` -> ``(N,)`` complex in raw units."""
    re = pred_norm[:, 0] * dataset.norm.v_re_std + dataset.norm.v_re_mean
    im = pred_norm[:, 1] * dataset.norm.v_im_std + dataset.norm.v_im_mean
    return re + 1j * im


def _decode_recovered(z: np.ndarray, p: int) -> np.ndarray:
    return (np.round(p * np.angle(z) / (2.0 * math.pi)).astype(int)) % p


def excluded_loss_at_freqs(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    teacher: CharacterMeshTeacher,
    *,
    modes: list[int] | tuple[int, ...],
    component: str = "trunk",
) -> dict[str, float]:
    """Re-evaluate value MSE / modular accuracy after removing Fourier modes.

    Parameters
    ----------
    modes
        Set of frequency indices ``k`` (``1 <= k <= p // 2``) to *remove*.
        Empty list means "no ablation" and the result equals the baseline
        lattice diagnostics.
    component
        ``"trunk"`` projects ``H_lat`` (and ``H_along_axis(0)`` / ``(1)``)
        orthogonal to the column space of the canonical Fourier columns
        for the given modes, then re-pushes through the head.
        ``"readout"`` instead applies the rank reduction directly to the
        ``(p, p)`` predicted lattice table by zeroing the FFT coefficients
        at those modes.  Both reproduce the qualitative behaviour from
        Nanda's paper; "trunk" matches the manuscript more literally.

    Returns
    -------
    dict
        Keys: ``baseline_value_mse``, ``ablated_value_mse``,
        ``baseline_modular_accuracy``, ``ablated_modular_accuracy``,
        ``baseline_train_minus_holdout``, ``ablated_train_minus_holdout``,
        ``n_modes_removed``.
    """
    if component not in ("trunk", "readout"):
        raise ValueError(f"component must be 'trunk' or 'readout', got {component}")
    p = int(teacher.modulus)
    modes_arr = np.asarray(sorted(set(int(m) for m in modes)), dtype=np.int64)
    if modes_arr.size > 0 and (modes_arr.min() < 1 or modes_arr.max() > p // 2):
        raise ValueError(
            f"modes must be in [1, p//2={p // 2}], got {modes_arr.tolist()}"
        )

    is_node = np.asarray(dataset.is_node, dtype=bool)
    if not is_node.any():
        return {
            "baseline_value_mse": float("nan"),
            "ablated_value_mse": float("nan"),
            "baseline_modular_accuracy": float("nan"),
            "ablated_modular_accuracy": float("nan"),
            "baseline_train_minus_holdout": float("nan"),
            "ablated_train_minus_holdout": float("nan"),
            "n_modes_removed": int(modes_arr.size),
        }

    nodes_x = np.asarray(teacher.nodes_x, dtype=np.float64)[:p]
    nodes_y = np.asarray(teacher.nodes_y, dtype=np.float64)[:p]
    xy_lat_raw = np.asarray(dataset.xy_raw)[is_node]
    i_idx = np.argmin(np.abs(xy_lat_raw[:, 0:1] - nodes_x[None, :]), axis=1).astype(int)
    j_idx = np.argmin(np.abs(xy_lat_raw[:, 1:2] - nodes_y[None, :]), axis=1).astype(int)

    # Baseline lattice predictions.
    xy_lat = jnp.asarray(np.asarray(dataset.xy)[is_node])
    pred_norm = np.asarray(f_arith_character_batched(student, xy_lat))  # (K, 2)
    z_base = _denormalise_predictions(pred_norm, dataset)               # (K,)
    truth_lat = (i_idx + j_idx) % p

    target_re = np.asarray(dataset.V_re)[is_node]
    target_im = np.asarray(dataset.V_im)[is_node]
    target_re_raw = target_re * dataset.norm.v_re_std + dataset.norm.v_re_mean
    target_im_raw = target_im * dataset.norm.v_im_std + dataset.norm.v_im_mean

    z_grid_base = np.full((p, p), np.nan + 1j * np.nan, dtype=np.complex128)
    z_grid_base[i_idx, j_idx] = z_base
    target_grid = np.full((p, p), np.nan + 1j * np.nan, dtype=np.complex128)
    target_grid[i_idx, j_idx] = target_re_raw + 1j * target_im_raw

    if component == "readout":
        z_grid_ab = _project_out_grid_modes(z_grid_base, modes_arr, p=p)
    else:
        # Build a (p, p, 2) per-channel lattice prediction grid via the head
        # acting on H_lat that has been projected.
        latents = compute_latents_character(student, dataset, teacher)
        H_lat_grid = latents.H_lat_grid  # (p, p, D)
        H_proj = _project_out_lattice_grid(H_lat_grid, modes_arr, p=p)
        valid = np.isfinite(H_lat_grid).all(axis=-1)
        head_apply = jax.jit(jax.vmap(student.head_a))
        D = H_proj.shape[-1]
        H_flat = H_proj.reshape(-1, D)
        finite = np.isfinite(H_flat).all(axis=-1)
        pred_flat = np.full((H_flat.shape[0], 2), np.nan, dtype=np.float64)
        if finite.any():
            pred_flat[finite] = np.asarray(
                head_apply(jnp.asarray(H_flat[finite].astype(np.float32)))
            )
        pred_grid_norm = pred_flat.reshape(p, p, 2)
        re_raw = (
            pred_grid_norm[..., 0] * dataset.norm.v_re_std + dataset.norm.v_re_mean
        )
        im_raw = (
            pred_grid_norm[..., 1] * dataset.norm.v_im_std + dataset.norm.v_im_mean
        )
        z_grid_ab = re_raw + 1j * im_raw
        z_grid_ab[~valid] = np.nan + 1j * np.nan

    # Lattice-wide value MSE (against the analytic-raw target zeta^(i+j)).
    def _mse(grid: np.ndarray) -> float:
        diff = grid - target_grid
        with np.errstate(invalid="ignore"):
            sq = np.abs(diff) ** 2
        m = np.nanmean(sq)
        return float(m) if np.isfinite(m) else float("nan")

    base_mse = _mse(z_grid_base)
    ab_mse = _mse(z_grid_ab)

    # Modular accuracy from each grid.
    valid = np.isfinite(z_grid_base.real) & np.isfinite(z_grid_base.imag)
    z_b_flat = z_grid_base[valid]
    z_a_flat = z_grid_ab[np.isfinite(z_grid_ab.real) & np.isfinite(z_grid_ab.imag)]
    rec_b = _decode_recovered(z_grid_base[i_idx, j_idx], p)
    rec_a_full = _decode_recovered(z_grid_ab[i_idx, j_idx], p)
    base_acc = float((rec_b == truth_lat).mean())
    ab_acc = float((rec_a_full == truth_lat).mean())

    # Train / holdout split (sort lex by (i, j)).
    order = np.lexsort((j_idx.astype(np.int64), i_idx.astype(np.int64)))
    n = int(i_idx.shape[0])
    if n >= 5:
        n_hold = max(1, int(math.ceil(0.2 * n)))
        n_hold = min(n_hold, n - 1)
        hold_mask = np.zeros(n, dtype=bool)
        hold_mask[order[-n_hold:]] = True
        train_mask = ~hold_mask
        base_gap = float((rec_b[train_mask] == truth_lat[train_mask]).mean()) - float(
            (rec_b[hold_mask] == truth_lat[hold_mask]).mean()
        )
        ab_gap = float((rec_a_full[train_mask] == truth_lat[train_mask]).mean()) - float(
            (rec_a_full[hold_mask] == truth_lat[hold_mask]).mean()
        )
    else:
        base_gap = float("nan")
        ab_gap = float("nan")
    # Silence flake8 "unused" warnings.
    del z_b_flat, z_a_flat

    return {
        "baseline_value_mse": base_mse,
        "ablated_value_mse": ab_mse,
        "baseline_modular_accuracy": base_acc,
        "ablated_modular_accuracy": ab_acc,
        "baseline_train_minus_holdout": base_gap,
        "ablated_train_minus_holdout": ab_gap,
        "n_modes_removed": int(modes_arr.size),
    }


def _project_out_grid_modes(
    grid: np.ndarray, modes: np.ndarray, *, p: int
) -> np.ndarray:
    """Zero ``modes`` (and their negatives) in the 2-D FFT of a ``(p, p)`` grid.

    Operates on the complex-valued lattice prediction grid; non-finite
    entries are preserved.  Treats every ``kx`` or ``ky`` in ``modes`` as a
    band to zero -- i.e. removes the rows / columns of the FFT with that
    spatial frequency on *either* axis.
    """
    finite = np.isfinite(grid.real) & np.isfinite(grid.imag)
    if not finite.all():
        # Fill nans with zero before FFT, then re-apply mask at the end.
        filled = np.where(finite, grid, 0.0 + 0.0j)
    else:
        filled = grid
    F = np.fft.fft2(filled)
    mask = np.ones((p, p), dtype=bool)
    for k in modes:
        k_int = int(k)
        for kk in (k_int % p, (-k_int) % p):
            mask[kk, :] = False
            mask[:, kk] = False
    F_proj = np.where(mask, F, 0.0 + 0.0j)
    out = np.fft.ifft2(F_proj)
    out = out.astype(np.complex128)
    out[~finite] = np.nan + 1j * np.nan
    return out


def _project_out_lattice_grid(
    H_grid: np.ndarray, modes: np.ndarray, *, p: int
) -> np.ndarray:
    """Project trunk lattice activations orthogonal to specified Fourier modes.

    For each neuron channel ``n``, the surface ``H_grid[:, :, n]`` is
    FFT'd; FFT coefficients at any ``kx in modes`` or ``ky in modes``
    (modulo ``p`` symmetry) are zeroed; the inverse FFT yields the
    projected channel.  Non-finite rows are left untouched (re-applied).
    """
    if H_grid.shape[0] != p or H_grid.shape[1] != p:
        raise RuntimeError(f"H_grid is not (p, p, D): {H_grid.shape}")
    finite = np.isfinite(H_grid).all(axis=-1)  # (p, p)
    filled = np.where(finite[..., None], H_grid, 0.0)
    D = filled.shape[-1]
    out = np.empty_like(filled)
    for n in range(D):
        F = np.fft.fft2(filled[..., n])
        mask = np.ones((p, p), dtype=bool)
        for k in modes:
            k_int = int(k)
            for kk in (k_int % p, (-k_int) % p):
                mask[kk, :] = False
                mask[:, kk] = False
        F_proj = np.where(mask, F, 0.0 + 0.0j)
        out[..., n] = np.real(np.fft.ifft2(F_proj))
    out = np.where(finite[..., None], out, np.nan)
    return out


# ---------------------------------------------------------------------------
# 4) PCA / helix probe on H_lat
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HelixReport:
    """Helix-PCA report for one input axis.

    ``H_lat`` (lattice trunk activations) is averaged over the *other*
    axis to get a per-i (or per-j) ``(p, D)`` matrix, then projected onto
    its top ``n_components`` principal components.  The trace in
    ``(PC1, PC2)`` should be circular if the trunk learned a single
    Fourier mode along that axis; the wrap angle reports the total
    rotation.
    """

    axis: str
    p: int
    components: np.ndarray            # (n_components, D)
    coords: np.ndarray                # (p, n_components)
    explained_variance: np.ndarray    # (n_components,)
    explained_variance_ratio: np.ndarray  # (n_components,)
    wrap_angle_radians: float
    helix_r2_per_k: dict[int, float]
    best_k: int
    best_r2: float


def helix_pca(
    latents: CharacterLatentArrays,
    *,
    axis: str = "i",
    n_components: int = 6,
) -> HelixReport:
    """PCA on the trunk's lattice activations ordered by one axis index.

    Workflow:

    1. Bin ``H_lat`` by ``i_idx`` (or ``j_idx``); average across the
       *other* axis so the result is ``(p, D)`` -- one mean activation
       per axis index.
    2. PCA -> ``(p, n_components)`` coordinates.
    3. Fit each ``(cos 2 pi k i / p, sin 2 pi k i / p)`` curve into the
       PC1-PC2 subspace via linear regression and record the joint
       ``R^2``; report the dominant ``k``.
    4. Compute the wrap angle in ``(PC1, PC2)`` as the total signed
       angular sweep around the centroid.
    """
    if axis not in ("i", "j"):
        raise ValueError(f"axis must be 'i' or 'j', got {axis!r}")
    p = int(latents.modulus)
    idx = (latents.i_idx if axis == "i" else latents.j_idx).astype(np.int64)
    H_lat = latents.H_lat.astype(np.float64)
    D = int(H_lat.shape[1])
    H_per_idx = np.zeros((p, D), dtype=np.float64)
    counts = np.zeros((p,), dtype=np.int64)
    for h, k in zip(H_lat, idx):
        H_per_idx[int(k)] += h
        counts[int(k)] += 1
    valid = counts > 0
    H_per_idx[valid] /= counts[valid, None]
    if not valid.all():
        # Fall back: replicate global mean into empty bins so the SVD has
        # full rank in i.
        H_per_idx[~valid] = H_lat.mean(axis=0)

    centered = H_per_idx - H_per_idx.mean(axis=0, keepdims=True)
    # SVD; we want left singular vectors as components.
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n_keep = min(int(n_components), Vt.shape[0])
    components = Vt[:n_keep]                            # (n_keep, D)
    explained_var = (S[:n_keep] ** 2) / max(1, p - 1)
    total_var = (S ** 2).sum() / max(1, p - 1)
    ratio = explained_var / total_var if total_var > 0 else np.zeros_like(explained_var)
    coords = centered @ components.T                    # (p, n_keep)

    # Helix R^2 per frequency: fit (cos, sin) of frequency k against PC1, PC2.
    r2_per_k: dict[int, float] = {}
    best_k = 1
    best_r2 = -np.inf
    omega = 2.0 * math.pi / p
    if n_keep >= 2:
        pc12 = coords[:, :2]                            # (p, 2)
        ii = np.arange(p, dtype=np.float64)
        for k in range(1, p // 2 + 1):
            X = np.stack([np.cos(omega * k * ii), np.sin(omega * k * ii)], axis=-1)
            X_aug = np.concatenate([X, np.ones((p, 1))], axis=-1)
            # Fit both PCs jointly.
            sol, *_ = np.linalg.lstsq(X_aug, pc12, rcond=None)
            pred = X_aug @ sol
            ss_res = float(((pc12 - pred) ** 2).sum())
            ss_tot = float(((pc12 - pc12.mean(axis=0, keepdims=True)) ** 2).sum())
            r2 = 1.0 - ss_res / max(ss_tot, 1e-30) if ss_tot > 0 else float("nan")
            r2_per_k[int(k)] = r2
            if r2 > best_r2:
                best_r2 = r2
                best_k = int(k)
        # Wrap angle: cumulative signed angle increments around centroid,
        # including the closing step ``p-1 -> 0`` so a single-mode helix at
        # frequency k reports exactly ``2 pi k`` (an integer number of turns).
        cx, cy = float(pc12[:, 0].mean()), float(pc12[:, 1].mean())
        ang = np.arctan2(pc12[:, 1] - cy, pc12[:, 0] - cx)
        ang_closed = np.concatenate([ang, ang[:1]])  # close the loop
        diff = np.diff(np.unwrap(ang_closed))
        wrap = float(np.sum(diff))
    else:
        wrap = float("nan")

    return HelixReport(
        axis=axis,
        p=p,
        components=components,
        coords=coords,
        explained_variance=explained_var.astype(np.float64),
        explained_variance_ratio=np.asarray(ratio, dtype=np.float64),
        wrap_angle_radians=wrap,
        helix_r2_per_k=r2_per_k,
        best_k=int(best_k),
        best_r2=float(best_r2 if np.isfinite(best_r2) else float("nan")),
    )


# ---------------------------------------------------------------------------
# 5) Causal subspace ablation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationReport:
    """Causal subspace-projection ablation report.

    The trunk output at every lattice point is projected orthogonal to
    ``subspace`` (a ``(r, D)`` matrix of stacked directions), then fed
    into ``head_a``.  ``modular_accuracy`` and ``unit_circle_residual``
    are recomputed and compared to the unablated baseline.
    """

    name: str
    rank: int
    baseline_modular_accuracy: float
    ablated_modular_accuracy: float
    baseline_unit_circle_residual: float
    ablated_unit_circle_residual: float
    baseline_value_mse: float
    ablated_value_mse: float
    delta_modular_accuracy: float


def _orthonormalise(subspace: np.ndarray, D: int) -> np.ndarray:
    """Drop zero-norm rows and Gram-Schmidt orthonormalise the remainder."""
    if subspace.ndim == 1:
        subspace = subspace[None, :]
    if subspace.shape[1] != D:
        raise ValueError(
            f"subspace last-axis dim {subspace.shape[1]} != trunk D {D}"
        )
    if subspace.shape[0] == 0:
        return np.zeros((0, D), dtype=np.float64)
    sub = np.asarray(subspace, dtype=np.float64)
    norms = np.linalg.norm(sub, axis=1)
    keep = norms > 1e-12
    sub = sub[keep]
    if sub.shape[0] == 0:
        return np.zeros((0, D), dtype=np.float64)
    Q, _ = np.linalg.qr(sub.T)
    return Q.T  # (r, D)


def ablate_subspace_and_score(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    teacher: CharacterMeshTeacher,
    *,
    subspace: np.ndarray,
    name: str = "ablation",
) -> AblationReport:
    """Project the trunk output orthogonal to ``subspace`` and re-score.

    ``subspace`` can be any ``(r, D)`` matrix (rows are directions in the
    trunk's embedding space).  Zero-norm rows are dropped; the remaining
    rows are Gram-Schmidt'd to an orthonormal basis before projection.
    ``head_a`` is then applied to the projected lattice activations and
    modular accuracy / value MSE are computed against the analytic
    teacher.

    A ``subspace=zeros((0, D))`` call reproduces the unablated baseline
    and is the natural sanity check.
    """
    is_node = np.asarray(dataset.is_node, dtype=bool)
    if not is_node.any():
        nan = float("nan")
        return AblationReport(
            name=name,
            rank=int(subspace.shape[0]) if subspace.ndim == 2 else 1,
            baseline_modular_accuracy=nan,
            ablated_modular_accuracy=nan,
            baseline_unit_circle_residual=nan,
            ablated_unit_circle_residual=nan,
            baseline_value_mse=nan,
            ablated_value_mse=nan,
            delta_modular_accuracy=nan,
        )

    p = int(teacher.modulus)
    xy_lat = np.asarray(dataset.xy)[is_node]
    trunk_fn = jax.jit(jax.vmap(student.trunk))
    H_lat = np.asarray(trunk_fn(jnp.asarray(xy_lat)))  # (K, D)
    D = int(H_lat.shape[1])
    Q = _orthonormalise(np.asarray(subspace), D)

    if Q.shape[0] > 0:
        # H_proj = H - H Q^T Q
        proj_coeffs = H_lat @ Q.T  # (K, r)
        H_proj = H_lat - proj_coeffs @ Q
    else:
        H_proj = H_lat.copy()

    head_apply = jax.jit(jax.vmap(student.head_a))
    pred_base = np.asarray(head_apply(jnp.asarray(H_lat.astype(np.float32))))
    pred_ab = np.asarray(head_apply(jnp.asarray(H_proj.astype(np.float32))))

    z_base = _denormalise_predictions(pred_base, dataset)
    z_ab = _denormalise_predictions(pred_ab, dataset)

    nodes_x = np.asarray(teacher.nodes_x, dtype=np.float64)[:p]
    nodes_y = np.asarray(teacher.nodes_y, dtype=np.float64)[:p]
    xy_lat_raw = np.asarray(dataset.xy_raw)[is_node]
    i_idx = np.argmin(np.abs(xy_lat_raw[:, 0:1] - nodes_x[None, :]), axis=1).astype(int)
    j_idx = np.argmin(np.abs(xy_lat_raw[:, 1:2] - nodes_y[None, :]), axis=1).astype(int)
    truth = (i_idx + j_idx) % p

    base_acc = float((_decode_recovered(z_base, p) == truth).mean())
    ab_acc = float((_decode_recovered(z_ab, p) == truth).mean())
    base_unit = float(np.abs(np.abs(z_base) - 1.0).mean())
    ab_unit = float(np.abs(np.abs(z_ab) - 1.0).mean())

    target_re = np.asarray(dataset.V_re)[is_node]
    target_im = np.asarray(dataset.V_im)[is_node]
    target_re_raw = target_re * dataset.norm.v_re_std + dataset.norm.v_re_mean
    target_im_raw = target_im * dataset.norm.v_im_std + dataset.norm.v_im_mean
    target_z = target_re_raw + 1j * target_im_raw

    base_mse = float(np.mean(np.abs(z_base - target_z) ** 2))
    ab_mse = float(np.mean(np.abs(z_ab - target_z) ** 2))

    return AblationReport(
        name=name,
        rank=int(Q.shape[0]),
        baseline_modular_accuracy=base_acc,
        ablated_modular_accuracy=ab_acc,
        baseline_unit_circle_residual=base_unit,
        ablated_unit_circle_residual=ab_unit,
        baseline_value_mse=base_mse,
        ablated_value_mse=ab_mse,
        delta_modular_accuracy=ab_acc - base_acc,
    )


__all__ = [
    "AblationReport",
    "DFTReport",
    "HelixReport",
    "Surface2DReport",
    "ablate_subspace_and_score",
    "dft_trunk_along_axis",
    "excluded_loss_at_freqs",
    "fft2_neuron_surface",
    "helix_pca",
]
