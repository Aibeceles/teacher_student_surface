"""Structural probes for the character Sobolev student.

Mirrors :mod:`sobolev_distill.probes` with a (Re, Im) split and a new
``modular_addition_recovery_probe`` that decodes the predicted character
into ``(i + j) mod p`` and reports top-1 accuracy + confusion matrix.

Helpers (``_pca_block``, ``_ridge_fit_predict``, ``_ridge_probe_one``,
``_factorisation_index``, ``_enumerate_axis_pairs``, ``_r2``,
``_participation_ratio``, ``_effective_rank_at``) are reused from
:mod:`sobolev_distill.probes` because they are pure functions of the
trunk activations / target arrays.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from sobolev_distill.probes import (
    LinearProbeReport,
    PCAReport,
    _factorisation_index,
    _enumerate_axis_pairs,
    _pca_block,
    _ridge_probe_one,
)

from .dataset import CharacterSobolevDataset
from .model import CharacterStudent, f_arith_character
from .teacher import CharacterMeshTeacher
from .train import CharacterDiagnosticsReport


# ---------------------------------------------------------------------------
# Latent cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharacterLatentArrays:
    """Trunk activations + lattice index recovery + character labels."""

    H: np.ndarray              # (N, D) trunk output on every kept mesh point
    H_lat: np.ndarray          # (K_lat, D) trunk output at lattice nodes
    H_lat_grid: np.ndarray     # (Kx, Ky, D) lattice latents scattered
    lattice_mask: np.ndarray   # (N,) bool
    i_idx: np.ndarray          # (K_lat,) int
    j_idx: np.ndarray          # (K_lat,) int
    Kx: int
    Ky: int
    embed_dim: int
    T_re_at_lat: np.ndarray    # (K_lat,) Re T[i,j]
    T_im_at_lat: np.ndarray    # (K_lat,) Im T[i,j]
    is_pd_lat: np.ndarray      # (K_lat,) bool
    V_re_lat_norm: np.ndarray  # (K_lat,) normalised
    V_im_lat_norm: np.ndarray  # (K_lat,)
    xy_lat_norm: np.ndarray    # (K_lat, 2)
    xy_lat_raw: np.ndarray     # (K_lat, 2)
    xy_full_norm: np.ndarray   # (N, 2)
    xy_full_raw: np.ndarray    # (N, 2)
    is_pd_full: np.ndarray     # (N,) bool
    V_re_full_norm: np.ndarray # (N,)
    V_im_full_norm: np.ndarray # (N,)
    modulus: int


def _recover_lattice_indices(
    xy_lat_raw: np.ndarray,
    nodes_x: np.ndarray,
    nodes_y: np.ndarray,
    *,
    atol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    diffs_x = np.abs(xy_lat_raw[:, 0:1] - nodes_x[None, :])
    diffs_y = np.abs(xy_lat_raw[:, 1:2] - nodes_y[None, :])
    i_idx = np.argmin(diffs_x, axis=1)
    j_idx = np.argmin(diffs_y, axis=1)
    n = xy_lat_raw.shape[0]
    if n > 0 and (
        diffs_x[np.arange(n), i_idx].max() > atol
        or diffs_y[np.arange(n), j_idx].max() > atol
    ):
        raise RuntimeError(
            "Lattice points do not align with teacher node grid within atol; "
            "check dataset/teacher consistency."
        )
    return i_idx.astype(np.int64), j_idx.astype(np.int64)


def compute_latents_character(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    teacher: CharacterMeshTeacher,
) -> CharacterLatentArrays:
    """Evaluate ``student.trunk`` over ``dataset.xy`` once and cache results."""
    trunk_fn = jax.vmap(student.trunk)
    H = np.asarray(trunk_fn(dataset.xy))                                   # (N, D)

    is_node = np.asarray(dataset.is_node, dtype=bool)
    H_lat = H[is_node]
    xy_full_norm = np.asarray(dataset.xy)
    xy_full_raw = np.asarray(dataset.xy_raw)
    xy_lat_raw = xy_full_raw[is_node]
    xy_lat_norm = xy_full_norm[is_node]
    nodes_x = np.asarray(teacher.nodes_x)
    nodes_y = np.asarray(teacher.nodes_y)
    i_idx, j_idx = _recover_lattice_indices(xy_lat_raw, nodes_x, nodes_y)

    T_re = np.asarray(teacher.T_re)
    T_im = np.asarray(teacher.T_im)
    T_re_at_lat = T_re[i_idx, j_idx]
    T_im_at_lat = T_im[i_idx, j_idx]
    is_pd_full = np.asarray(dataset.is_pd, dtype=bool)
    V_re_full_norm = np.asarray(dataset.V_re, dtype=np.float64)
    V_im_full_norm = np.asarray(dataset.V_im, dtype=np.float64)
    is_pd_lat = is_pd_full[is_node]
    V_re_lat_norm = V_re_full_norm[is_node]
    V_im_lat_norm = V_im_full_norm[is_node]

    Kx = int(nodes_x.size)
    Ky = int(nodes_y.size)
    embed_dim = int(H.shape[1])

    H_lat_grid = np.full((Kx, Ky, embed_dim), np.nan, dtype=H.dtype)
    H_lat_grid[i_idx, j_idx] = H_lat

    return CharacterLatentArrays(
        H=H,
        H_lat=H_lat,
        H_lat_grid=H_lat_grid,
        lattice_mask=is_node,
        i_idx=i_idx,
        j_idx=j_idx,
        Kx=Kx,
        Ky=Ky,
        embed_dim=embed_dim,
        T_re_at_lat=T_re_at_lat,
        T_im_at_lat=T_im_at_lat,
        is_pd_lat=is_pd_lat,
        V_re_lat_norm=V_re_lat_norm,
        V_im_lat_norm=V_im_lat_norm,
        xy_lat_norm=xy_lat_norm,
        xy_lat_raw=xy_lat_raw,
        xy_full_norm=xy_full_norm,
        xy_full_raw=xy_full_raw,
        is_pd_full=is_pd_full,
        V_re_full_norm=V_re_full_norm,
        V_im_full_norm=V_im_full_norm,
        modulus=int(dataset.modulus),
    )


# ---------------------------------------------------------------------------
# PCA / linear probes
# ---------------------------------------------------------------------------


def latent_pca_character(latents: CharacterLatentArrays) -> PCAReport:
    """Mirror of :func:`sobolev_distill.probes.latent_pca` with character colour keys."""
    s_full, frac_full, er_full, pr_full, top2_full = _pca_block(latents.H)
    s_lat, frac_lat, er_lat, pr_lat, top2_lat = _pca_block(latents.H_lat)
    p = latents.modulus
    color_keys: dict[str, np.ndarray] = {
        "i": latents.i_idx.astype(np.float64),
        "j": latents.j_idx.astype(np.float64),
        "(i+j) mod p": ((latents.i_idx + latents.j_idx) % p).astype(np.float64),
        "Re T[i,j]": latents.T_re_at_lat.astype(np.float64),
        "Im T[i,j]": latents.T_im_at_lat.astype(np.float64),
        "is_pd": latents.is_pd_lat.astype(np.float64),
    }
    return PCAReport(
        spectrum_full=s_full,
        frac_variance_full=frac_full,
        effective_rank_99_full=er_full,
        participation_ratio_full=pr_full,
        top2_coords_full=top2_full,
        spectrum_lat=s_lat,
        frac_variance_lat=frac_lat,
        effective_rank_99_lat=er_lat,
        participation_ratio_lat=pr_lat,
        top2_coords_lat=top2_lat,
        lattice_color_keys=color_keys,
    )


def linear_probes_character(
    latents: CharacterLatentArrays,
    *,
    ridge: float = 1e-3,
    seed: int = 0,
    test_frac: float = 0.2,
) -> LinearProbeReport:
    """Ridge ``R^2`` from ``H`` onto a character-aware target panel.

    Targets:
    - mesh: ``x``, ``y``, ``is_pd``, ``V_re``, ``V_im``
    - lattice: ``Re T[i,j]``, ``Im T[i,j]``, ``Re zeta^i``, ``Im zeta^i``,
      ``Re zeta^j``, ``Im zeta^j``, ``(i+j) mod p``
    """
    H_full = latents.H.astype(np.float64)
    H_lat = latents.H_lat.astype(np.float64)
    p = latents.modulus
    omega = 2.0 * math.pi / p

    targets_used: dict[str, str] = {}
    r2_test: dict[str, float] = {}
    r2_train: dict[str, float] = {}
    coefs: dict[str, np.ndarray] = {}

    mesh_targets: dict[str, np.ndarray] = {
        "x": latents.xy_full_norm[:, 0].astype(np.float64),
        "y": latents.xy_full_norm[:, 1].astype(np.float64),
        "is_pd": latents.is_pd_full.astype(np.float64),
        "V_re": latents.V_re_full_norm.astype(np.float64),
        "V_im": latents.V_im_full_norm.astype(np.float64),
    }
    re_zi = np.cos(omega * latents.i_idx).astype(np.float64)
    im_zi = np.sin(omega * latents.i_idx).astype(np.float64)
    re_zj = np.cos(omega * latents.j_idx).astype(np.float64)
    im_zj = np.sin(omega * latents.j_idx).astype(np.float64)
    lat_targets: dict[str, np.ndarray] = {
        "Re T[i,j]": latents.T_re_at_lat.astype(np.float64),
        "Im T[i,j]": latents.T_im_at_lat.astype(np.float64),
        "Re zeta^i": re_zi,
        "Im zeta^i": im_zi,
        "Re zeta^j": re_zj,
        "Im zeta^j": im_zj,
        "(i+j) mod p": ((latents.i_idx + latents.j_idx) % p).astype(np.float64),
    }

    for offset, (name, y) in enumerate(mesh_targets.items()):
        r2_te, r2_tr, coef = _ridge_probe_one(
            H_full, y, ridge=ridge, seed=seed + offset, test_frac=test_frac
        )
        targets_used[name] = "mesh"
        r2_test[name] = r2_te
        r2_train[name] = r2_tr
        coefs[name] = coef
    for offset, (name, y) in enumerate(lat_targets.items()):
        r2_te, r2_tr, coef = _ridge_probe_one(
            H_lat, y, ridge=ridge, seed=seed + 1000 + offset, test_frac=test_frac
        )
        targets_used[name] = "lattice"
        r2_test[name] = r2_te
        r2_train[name] = r2_tr
        coefs[name] = coef

    return LinearProbeReport(
        r2=r2_test,
        r2_train=r2_train,
        coefficients=coefs,
        targets_used=targets_used,
        ridge=float(ridge),
    )


# ---------------------------------------------------------------------------
# Activation patching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharacterPatchingReport:
    """Subspace activation patching across lattice rows and columns.

    Same metric as :class:`sobolev_distill.probes.PatchingReport` but the
    head returns ``(re, im)`` so MSE is summed over both channels.
    """

    no_swap_row_mse: float
    swap_row_mse: float
    full_swap_row_mse: float
    row_factorisation_index: float
    no_swap_col_mse: float
    swap_col_mse: float
    full_swap_col_mse: float
    col_factorisation_index: float
    n_row_pairs: int
    n_col_pairs: int


def patching_probe_character(
    student: CharacterStudent,
    latents: CharacterLatentArrays,
    probes_report: LinearProbeReport,
) -> CharacterPatchingReport:
    """Subspace patching using the rank-1 ``Re zeta^j`` / ``Re zeta^i`` directions."""
    H_grid = latents.H_lat_grid
    valid = np.isfinite(H_grid).all(axis=-1)

    V_re_grid = np.full((latents.Kx, latents.Ky), np.nan, dtype=np.float64)
    V_im_grid = np.full((latents.Kx, latents.Ky), np.nan, dtype=np.float64)
    V_re_grid[latents.i_idx, latents.j_idx] = latents.V_re_lat_norm
    V_im_grid[latents.i_idx, latents.j_idx] = latents.V_im_lat_norm

    head_apply = jax.jit(jax.vmap(student.head_a))

    pred_lat = np.asarray(head_apply(jnp.asarray(latents.H_lat)))
    pred_re_grid = np.full((latents.Kx, latents.Ky), np.nan, dtype=np.float64)
    pred_im_grid = np.full((latents.Kx, latents.Ky), np.nan, dtype=np.float64)
    pred_re_grid[latents.i_idx, latents.j_idx] = pred_lat[:, 0]
    pred_im_grid[latents.i_idx, latents.j_idx] = pred_lat[:, 1]

    coef_y = probes_report.coefficients["Re zeta^j"][:-1].astype(np.float64)
    norm_y = float(np.linalg.norm(coef_y))
    u_y = np.zeros_like(coef_y) if norm_y < 1e-12 else (coef_y / norm_y)
    coef_x = probes_report.coefficients["Re zeta^i"][:-1].astype(np.float64)
    norm_x = float(np.linalg.norm(coef_x))
    u_x = np.zeros_like(coef_x) if norm_x < 1e-12 else (coef_x / norm_x)

    def _patch_along(u: np.ndarray, axis: int):
        a_idx, b_idx = _enumerate_axis_pairs(valid, axis)
        if a_idx.shape[0] == 0:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        h_a = H_grid[a_idx[:, 0], a_idx[:, 1]]
        h_b = H_grid[b_idx[:, 0], b_idx[:, 1]]
        u_dt = u.astype(h_a.dtype)
        delta_proj = (h_b - h_a) @ u_dt
        h_swap = h_a + delta_proj[:, None] * u_dt[None, :]
        v_re_target = V_re_grid[b_idx[:, 0], b_idx[:, 1]]
        v_im_target = V_im_grid[b_idx[:, 0], b_idx[:, 1]]
        v_re_no = pred_re_grid[a_idx[:, 0], a_idx[:, 1]]
        v_im_no = pred_im_grid[a_idx[:, 0], a_idx[:, 1]]
        v_re_full = pred_re_grid[b_idx[:, 0], b_idx[:, 1]]
        v_im_full = pred_im_grid[b_idx[:, 0], b_idx[:, 1]]
        v_swap = np.asarray(head_apply(jnp.asarray(h_swap)))
        no_mse = float(
            np.mean((v_re_no - v_re_target) ** 2 + (v_im_no - v_im_target) ** 2)
        )
        sw_mse = float(
            np.mean((v_swap[:, 0] - v_re_target) ** 2 + (v_swap[:, 1] - v_im_target) ** 2)
        )
        full_mse = float(
            np.mean((v_re_full - v_re_target) ** 2 + (v_im_full - v_im_target) ** 2)
        )
        return no_mse, sw_mse, full_mse, _factorisation_index(sw_mse, no_mse, full_mse), int(a_idx.shape[0])

    no_row, sw_row, full_row, fi_row, n_row = _patch_along(u_y, axis=1)
    no_col, sw_col, full_col, fi_col, n_col = _patch_along(u_x, axis=0)

    return CharacterPatchingReport(
        no_swap_row_mse=no_row,
        swap_row_mse=sw_row,
        full_swap_row_mse=full_row,
        row_factorisation_index=fi_row,
        no_swap_col_mse=no_col,
        swap_col_mse=sw_col,
        full_swap_col_mse=full_col,
        col_factorisation_index=fi_col,
        n_row_pairs=n_row,
        n_col_pairs=n_col,
    )


# ---------------------------------------------------------------------------
# Hessian alignment (Re and Im)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HessianAlignmentChannel:
    """Per-channel (Re or Im) eigen-alignment between student and teacher."""

    student_min_eig: np.ndarray
    teacher_min_eig: np.ndarray
    student_max_eig: np.ndarray
    teacher_max_eig: np.ndarray
    both_pd_fraction: float
    eigvec_cosine_top: np.ndarray
    eigvec_cosine_bot: np.ndarray
    eigval_ratio_top: np.ndarray
    eigval_ratio_bot: np.ndarray


@dataclass(frozen=True)
class CharacterHessianAlignmentReport:
    """Re and Im eigen-alignment reports."""

    re: HessianAlignmentChannel
    im: HessianAlignmentChannel


def _hessian_alignment_one_channel(
    student_H: np.ndarray,
    teacher_H: np.ndarray,
) -> HessianAlignmentChannel:
    if student_H.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return HessianAlignmentChannel(
            student_min_eig=empty,
            teacher_min_eig=empty,
            student_max_eig=empty,
            teacher_max_eig=empty,
            both_pd_fraction=float("nan"),
            eigvec_cosine_top=empty,
            eigvec_cosine_bot=empty,
            eigval_ratio_top=empty,
            eigval_ratio_bot=empty,
        )
    s_sym = 0.5 * (student_H + np.swapaxes(student_H, -1, -2))
    t_sym = 0.5 * (teacher_H + np.swapaxes(teacher_H, -1, -2))
    s_evals, s_evecs = np.linalg.eigh(s_sym)
    t_evals, t_evecs = np.linalg.eigh(t_sym)
    s_min = s_evals[:, 0]
    s_max = s_evals[:, 1]
    t_min = t_evals[:, 0]
    t_max = t_evals[:, 1]
    both_pd = (s_min > 0.0) & (t_min > 0.0)
    both_pd_fraction = float(np.mean(both_pd.astype(np.float64))) if both_pd.size else float("nan")
    cos_bot = np.abs(np.einsum("kd,kd->k", s_evecs[:, :, 0], t_evecs[:, :, 0]))
    cos_top = np.abs(np.einsum("kd,kd->k", s_evecs[:, :, 1], t_evecs[:, :, 1]))
    safe_top = np.where(np.abs(t_max) > 1e-12, t_max, np.nan)
    safe_bot = np.where(np.abs(t_min) > 1e-12, t_min, np.nan)
    return HessianAlignmentChannel(
        student_min_eig=s_min,
        teacher_min_eig=t_min,
        student_max_eig=s_max,
        teacher_max_eig=t_max,
        both_pd_fraction=both_pd_fraction,
        eigvec_cosine_top=cos_top,
        eigvec_cosine_bot=cos_bot,
        eigval_ratio_top=s_max / safe_top,
        eigval_ratio_bot=s_min / safe_bot,
    )


def hessian_alignment_character(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
) -> CharacterHessianAlignmentReport:
    """Per-channel Hessian alignment at lattice nodes."""
    is_node = np.asarray(dataset.is_node, dtype=bool)
    if not is_node.any():
        empty_channel = _hessian_alignment_one_channel(
            np.zeros((0, 2, 2)), np.zeros((0, 2, 2))
        )
        return CharacterHessianAlignmentReport(re=empty_channel, im=empty_channel)

    xy_lat = jnp.asarray(np.asarray(dataset.xy)[is_node])

    # Per-channel student Hessian via index closures (channel as concrete int).
    def _hess_re(s, x):
        return jax.hessian(lambda inp: f_arith_character(s, inp)[0])(x)

    def _hess_im(s, x):
        return jax.hessian(lambda inp: f_arith_character(s, inp)[1])(x)

    student_H_re = np.asarray(jax.vmap(_hess_re, in_axes=(None, 0))(student, xy_lat))
    student_H_im = np.asarray(jax.vmap(_hess_im, in_axes=(None, 0))(student, xy_lat))

    Hxx_re = np.asarray(dataset.Hxx_re)[is_node]
    Hxx_im = np.asarray(dataset.Hxx_im)[is_node]
    Hxy_re = np.asarray(dataset.Hxy_re)[is_node]
    Hxy_im = np.asarray(dataset.Hxy_im)[is_node]
    Hyy_re = np.asarray(dataset.Hyy_re)[is_node]
    Hyy_im = np.asarray(dataset.Hyy_im)[is_node]
    teacher_H_re = np.stack(
        [np.stack([Hxx_re, Hxy_re], axis=-1), np.stack([Hxy_re, Hyy_re], axis=-1)],
        axis=-2,
    )
    teacher_H_im = np.stack(
        [np.stack([Hxx_im, Hxy_im], axis=-1), np.stack([Hxy_im, Hyy_im], axis=-1)],
        axis=-2,
    )
    return CharacterHessianAlignmentReport(
        re=_hessian_alignment_one_channel(student_H_re, teacher_H_re),
        im=_hessian_alignment_one_channel(student_H_im, teacher_H_im),
    )


# ---------------------------------------------------------------------------
# Modular addition recovery probe (NEW)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModularRecoveryReport:
    """Top-1 accuracy + confusion matrix of decoded ``(i+j) mod p``."""

    top1_accuracy: float
    confusion_matrix: np.ndarray   # (p, p) integer counts of (truth -> recovered)
    unit_circle_residual_mean: float
    unit_circle_residual_max: float
    n_lattice: int


def modular_addition_recovery_probe(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    teacher: CharacterMeshTeacher,
) -> ModularRecoveryReport:
    """Decode lattice predictions to ``(i+j) mod p`` and report accuracy.

    Predictions are denormalised back to raw character units before
    ``atan2``-decoding.  ``unit_circle_residual_*`` give a sanity bound on
    how far predictions drift off the unit circle (closer to 0 is better).
    """
    is_node = np.asarray(dataset.is_node, dtype=bool)
    if not is_node.any():
        return ModularRecoveryReport(
            top1_accuracy=float("nan"),
            confusion_matrix=np.zeros((teacher.modulus, teacher.modulus), dtype=np.int64),
            unit_circle_residual_mean=float("nan"),
            unit_circle_residual_max=float("nan"),
            n_lattice=0,
        )

    xy_lat = dataset.xy[is_node]
    pred = np.asarray(jax.vmap(lambda x: f_arith_character(student, x))(xy_lat))  # (K, 2)
    re = pred[:, 0] * dataset.norm.v_re_std + dataset.norm.v_re_mean
    im = pred[:, 1] * dataset.norm.v_im_std + dataset.norm.v_im_mean
    z = re + 1j * im

    p = int(teacher.modulus)
    nodes_x = np.asarray(teacher.nodes_x)
    nodes_y = np.asarray(teacher.nodes_y)
    xy_lat_raw = np.asarray(dataset.xy_raw)[is_node]
    i_idx = np.argmin(np.abs(xy_lat_raw[:, 0:1] - nodes_x[None, :]), axis=1)
    j_idx = np.argmin(np.abs(xy_lat_raw[:, 1:2] - nodes_y[None, :]), axis=1)
    truth = (i_idx.astype(int) + j_idx.astype(int)) % p
    recovered = (np.round(p * np.angle(z) / (2.0 * math.pi)).astype(int)) % p

    top1 = float((recovered == truth).mean())
    cm = np.zeros((p, p), dtype=np.int64)
    for t, r in zip(truth, recovered):
        cm[int(t), int(r)] += 1

    abs_z = np.abs(z)
    return ModularRecoveryReport(
        top1_accuracy=top1,
        confusion_matrix=cm,
        unit_circle_residual_mean=float(np.abs(abs_z - 1.0).mean()),
        unit_circle_residual_max=float(np.abs(abs_z - 1.0).max()),
        n_lattice=int(xy_lat_raw.shape[0]),
    )


# ---------------------------------------------------------------------------
# Bundle / driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharacterProbeBundle:
    """Aggregate of every probe report plus optional diagnostics."""

    latents: CharacterLatentArrays
    pca: PCAReport
    probes: LinearProbeReport
    patching: CharacterPatchingReport
    hessian: CharacterHessianAlignmentReport
    modular_recovery: ModularRecoveryReport
    diagnostics: CharacterDiagnosticsReport | None = None

    def verdict(
        self,
        *,
        max_unit_circle_residual: float = 0.1,
        min_pd_auroc: float = 0.95,
        min_modular_acc: float = 0.95,
        min_arith_r2: float = 0.9,
        max_modular_train_holdout_gap: float = 0.1,
        min_holdout_lattice: int = 5,
    ) -> dict[str, bool]:
        """Four-criterion test extending the real version with modular recovery.

        Adds ``modular_generalisation_ok`` from
        ``diagnostics.modular_acc_train_minus_holdout`` (requires
        :func:`~sobolev_distill_character.train.evaluate_diagnostics_character`
        to be passed as ``diagnostics``).  Emits :class:`UserWarning` when
        diagnostics are missing, the holdout split is too small, or the
        train-minus-holdout gap exceeds ``max_modular_train_holdout_gap``.

        ``all_four`` is unchanged (original four criteria).  ``all_five`` is
        ``all_four and modular_generalisation_ok``.
        """
        if self.diagnostics is None:
            unit_ok = bool(self.modular_recovery.unit_circle_residual_mean < max_unit_circle_residual)
            pd_ok = False
        else:
            unit_ok = bool(
                np.isfinite(self.diagnostics.unit_circle_residual)
                and self.diagnostics.unit_circle_residual < max_unit_circle_residual
            )
            pd_ok = bool(
                np.isfinite(self.diagnostics.energy_pd_auroc)
                and self.diagnostics.energy_pd_auroc > min_pd_auroc
            )
        r2_re_T = self.probes.r2.get("Re T[i,j]", float("nan"))
        r2_im_T = self.probes.r2.get("Im T[i,j]", float("nan"))
        r2_T = max(
            r2_re_T if np.isfinite(r2_re_T) else -np.inf,
            r2_im_T if np.isfinite(r2_im_T) else -np.inf,
        )
        arith_ok = bool(r2_T > min_arith_r2)
        modular_ok = bool(self.modular_recovery.top1_accuracy > min_modular_acc)
        all_four = bool(unit_ok and pd_ok and arith_ok and modular_ok)

        d = self.diagnostics
        gen_ok = False
        if d is None:
            warnings.warn(
                "CharacterProbeBundle.verdict: diagnostics missing; "
                "modular_generalisation_ok is False (cannot evaluate train vs holdout gap).",
                UserWarning,
                stacklevel=2,
            )
        else:
            tr = d.modular_recovery_accuracy_train
            ho = d.modular_recovery_accuracy_holdout
            gap = d.modular_acc_train_minus_holdout
            finite = np.isfinite(tr) and np.isfinite(ho) and np.isfinite(gap)
            n_lat = int(self.modular_recovery.n_lattice)
            n_hold = int(np.ceil(0.2 * n_lat)) if n_lat >= 5 else 0
            if not finite:
                warnings.warn(
                    "CharacterProbeBundle.verdict: train/holdout modular metrics are "
                    "non-finite (e.g. no lattice or empty holdout split); "
                    "modular_generalisation_ok is False.",
                    UserWarning,
                    stacklevel=2,
                )
            elif n_hold < min_holdout_lattice:
                warnings.warn(
                    f"CharacterProbeBundle.verdict: holdout lattice count {n_hold} "
                    f"< min_holdout_lattice={min_holdout_lattice}; "
                    "modular_generalisation_ok is False.",
                    UserWarning,
                    stacklevel=2,
                )
            elif gap > max_modular_train_holdout_gap:
                warnings.warn(
                    "Memorisation / uneven fit suspected: "
                    f"modular train_acc={tr:.4f}, holdout_acc={ho:.4f}, "
                    f"train_minus_holdout={gap:.4f} "
                    f"(threshold {max_modular_train_holdout_gap}).",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                gen_ok = True

        return {
            "unit_circle_following": unit_ok,
            "pd_certificate": pd_ok,
            "arithmetic_axes_linear": arith_ok,
            "modular_recovery_ok": modular_ok,
            "modular_generalisation_ok": gen_ok,
            "all_four": all_four,
            "all_five": bool(all_four and gen_ok),
        }


def run_all_probes_character(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    teacher: CharacterMeshTeacher,
    *,
    diagnostics: CharacterDiagnosticsReport | None = None,
    ridge: float = 1e-3,
    seed: int = 0,
) -> CharacterProbeBundle:
    """Run all six probes in dependency order."""
    latents = compute_latents_character(student, dataset, teacher)
    pca = latent_pca_character(latents)
    probes_rep = linear_probes_character(latents, ridge=ridge, seed=seed)
    patching = patching_probe_character(student, latents, probes_rep)
    hessian = hessian_alignment_character(student, dataset)
    modular = modular_addition_recovery_probe(student, dataset, teacher)
    return CharacterProbeBundle(
        latents=latents,
        pca=pca,
        probes=probes_rep,
        patching=patching,
        hessian=hessian,
        modular_recovery=modular,
        diagnostics=diagnostics,
    )


__all__ = [
    "CharacterHessianAlignmentReport",
    "CharacterLatentArrays",
    "CharacterPatchingReport",
    "CharacterProbeBundle",
    "HessianAlignmentChannel",
    "LinearProbeReport",
    "ModularRecoveryReport",
    "PCAReport",
    "compute_latents_character",
    "hessian_alignment_character",
    "latent_pca_character",
    "linear_probes_character",
    "modular_addition_recovery_probe",
    "patching_probe_character",
    "run_all_probes_character",
]
