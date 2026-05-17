"""Structural probes for the Sobolev student's latent representation.

Each probe answers a different aspect of the question "did the trunk learn
arithmetic structure or just memorise lattice values?":

- :func:`latent_pca` -- effective rank of ``trunk(xy)`` and lattice-restricted
  top-2 PCs (a structural fingerprint of the latent geometry).
- :func:`linear_probes` -- closed-form ridge ``R^2`` from latents onto a panel
  of arithmetic targets (``x``, ``y``, ``T[i, j]``, ``i + j``, ``i - j``,
  ``i * j``, ``is_pd``).
- :func:`patching_probe` -- subspace activation patching across lattice rows
  and columns, using the linear-probe directions as the patching subspace.
- :func:`hessian_alignment` -- per-lattice-node alignment between the student's
  Hessian (``jax.hessian``) and the teacher's Hessian, in eigenvalue and
  eigenvector terms.

The five reports plus an optional :class:`~train.DiagnosticsReport` flow into
:class:`ProbeBundle`, whose :meth:`ProbeBundle.verdict` implements the
three-criterion structural-resolution test:

1. ``node_grad_angle_deg < max_grad_angle_deg`` (the student follows the
   teacher's gradient field at lattice nodes).
2. ``energy_pd_auroc > min_pd_auroc`` (the energy head separates PD from
   non-PD lattice nodes).
3. ``max(R^2(i + j), R^2(T[i, j])) > min_arith_r2`` (the addition operation
   is linearly accessible from the latent).

All probes are pure functions; the single trunk evaluation done in
:func:`compute_latents` is reused across every downstream report.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .dataset import SobolevDataset
from .model import Student, f_arith
from .teacher import MeshTeacher
from .train import DiagnosticsReport


# ---------------------------------------------------------------------------
# Latent cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentArrays:
    """Cache of trunk activations on the mesh plus lattice index recovery.

    ``H`` is the full-mesh trunk output (one row per kept point in
    ``dataset.xy``).  ``H_lat`` selects the lattice rows; ``i_idx`` and
    ``j_idx`` give their lattice coordinates so any external array of shape
    ``(Kx, Ky, ...)`` can be gathered via ``arr[i_idx, j_idx]``.

    The dataclass is intentionally self-contained: every probe consumes only
    :class:`LatentArrays` (with the exception of :func:`hessian_alignment`,
    which needs the teacher Hessian fields stored on
    :class:`~dataset.SobolevDataset`).  This keeps the trunk evaluation a
    one-shot cost.
    """

    H: np.ndarray              # (N, D) trunk output on every kept mesh point
    H_lat: np.ndarray          # (K_lat, D) trunk output at lattice nodes
    H_lat_grid: np.ndarray     # (Kx, Ky, D) lattice latents scattered into a grid
    lattice_mask: np.ndarray   # (N,) bool mirror of dataset.is_node
    i_idx: np.ndarray          # (K_lat,) int index into teacher.nodes_x
    j_idx: np.ndarray          # (K_lat,) int index into teacher.nodes_y
    Kx: int
    Ky: int
    embed_dim: int
    T_at_lat: np.ndarray       # (K_lat,) raw teacher T at each lattice point
    is_pd_lat: np.ndarray      # (K_lat,) bool from dataset.is_pd
    V_lat_norm: np.ndarray     # (K_lat,) normalised V (head_a target frame)
    xy_lat_norm: np.ndarray    # (K_lat, 2) normalised inputs at lattice
    xy_lat_raw: np.ndarray     # (K_lat, 2) original inputs at lattice
    xy_full_norm: np.ndarray   # (N, 2) normalised inputs over full mesh
    xy_full_raw: np.ndarray    # (N, 2) original inputs over full mesh
    is_pd_full: np.ndarray     # (N,) bool over full mesh
    V_full_norm: np.ndarray    # (N,) normalised V over full mesh


def _recover_lattice_indices(
    xy_lat_raw: np.ndarray,
    nodes_x: np.ndarray,
    nodes_y: np.ndarray,
    *,
    atol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Match each lattice point's raw coordinates to ``(i, j)`` indices.

    Mirrors ``teacher._node_indices_on_mesh`` in spirit: take the nearest
    node index per axis and assert the residual is within ``atol`` (the
    teacher mesh always inserts exact node coordinates, so the residual is
    zero up to float-double rounding).
    """
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


def compute_latents(
    student: Student,
    dataset: SobolevDataset,
    teacher: MeshTeacher,
) -> LatentArrays:
    """Evaluate ``student.trunk`` over ``dataset.xy`` once and cache the result.

    All downstream probes consume :class:`LatentArrays` so the (potentially
    large) trunk evaluation happens exactly once per call.
    """
    trunk_fn = jax.vmap(student.trunk)
    H = np.asarray(trunk_fn(dataset.xy))                                  # (N, D)

    is_node = np.asarray(dataset.is_node, dtype=bool)
    H_lat = H[is_node]
    xy_full_norm = np.asarray(dataset.xy)
    xy_full_raw = np.asarray(dataset.xy_raw)
    xy_lat_raw = xy_full_raw[is_node]
    xy_lat_norm = xy_full_norm[is_node]
    nodes_x = np.asarray(teacher.nodes_x)
    nodes_y = np.asarray(teacher.nodes_y)
    i_idx, j_idx = _recover_lattice_indices(xy_lat_raw, nodes_x, nodes_y)

    T_arr = np.asarray(teacher.T)
    T_at_lat = T_arr[i_idx, j_idx]
    is_pd_full = np.asarray(dataset.is_pd, dtype=bool)
    V_full_norm = np.asarray(dataset.V, dtype=np.float64)
    is_pd_lat = is_pd_full[is_node]
    V_lat_norm = V_full_norm[is_node]

    Kx = int(nodes_x.size)
    Ky = int(nodes_y.size)
    embed_dim = int(H.shape[1])

    H_lat_grid = np.full((Kx, Ky, embed_dim), np.nan, dtype=H.dtype)
    H_lat_grid[i_idx, j_idx] = H_lat

    return LatentArrays(
        H=H,
        H_lat=H_lat,
        H_lat_grid=H_lat_grid,
        lattice_mask=is_node,
        i_idx=i_idx,
        j_idx=j_idx,
        Kx=Kx,
        Ky=Ky,
        embed_dim=embed_dim,
        T_at_lat=T_at_lat,
        is_pd_lat=is_pd_lat,
        V_lat_norm=V_lat_norm,
        xy_lat_norm=xy_lat_norm,
        xy_lat_raw=xy_lat_raw,
        xy_full_norm=xy_full_norm,
        xy_full_raw=xy_full_raw,
        is_pd_full=is_pd_full,
        V_full_norm=V_full_norm,
    )


# ---------------------------------------------------------------------------
# PCA / effective rank
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PCAReport:
    """SVD-based latent geometry summary, full-mesh and lattice-only."""

    spectrum_full: np.ndarray            # (D,) singular values (descending)
    frac_variance_full: np.ndarray       # (D,) cumulative explained variance
    effective_rank_99_full: int
    participation_ratio_full: float
    top2_coords_full: np.ndarray         # (N, 2) top-2 PC coordinates
    spectrum_lat: np.ndarray             # (D,) lattice-only singular values
    frac_variance_lat: np.ndarray
    effective_rank_99_lat: int
    participation_ratio_lat: float
    top2_coords_lat: np.ndarray          # (K_lat, 2) lattice-only top-2 PCs
    lattice_color_keys: dict[str, np.ndarray]


def _participation_ratio(s: np.ndarray) -> float:
    """``(sum s_i^2)^2 / sum s_i^4`` -- a continuous rank surrogate in ``[1, D]``."""
    s2 = (s * s).astype(np.float64)
    denom = float(np.sum(s2 * s2))
    if denom <= 0.0:
        return 0.0
    num = float(np.sum(s2)) ** 2
    return num / denom


def _effective_rank_at(frac_var: np.ndarray, threshold: float) -> int:
    """Smallest ``k`` such that cumulative variance ``>= threshold`` (1-indexed)."""
    if frac_var.size == 0:
        return 0
    idx = int(np.searchsorted(frac_var, threshold, side="left") + 1)
    return int(min(idx, frac_var.size))


def _pca_block(
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray]:
    """SVD of ``H - mean(H)`` returning ``(spectrum, frac_var, eff_rank, pr, top2)``."""
    if H.shape[0] == 0:
        zeros = np.zeros((0,), dtype=np.float64)
        return zeros, zeros, 0, 0.0, np.zeros((0, 2), dtype=np.float64)
    Hc = (H - H.mean(axis=0, keepdims=True)).astype(np.float64)
    U, s, _vt = np.linalg.svd(Hc, full_matrices=False)
    var = s * s
    total = float(var.sum())
    frac = np.cumsum(var) / total if total > 0.0 else np.zeros_like(var)
    eff_rank = _effective_rank_at(frac, 0.99)
    pr = _participation_ratio(s)

    if s.size >= 2:
        top2 = U[:, :2] * s[:2][None, :]
    elif s.size == 1:
        top2 = np.concatenate(
            [U[:, :1] * s[:1][None, :], np.zeros((H.shape[0], 1))], axis=-1
        )
    else:
        top2 = np.zeros((H.shape[0], 2), dtype=np.float64)
    return s, frac, eff_rank, pr, top2


def latent_pca(latents: LatentArrays) -> PCAReport:
    """PCA on full-mesh and lattice-restricted latents, plus colour keys.

    The colour keys are pre-computed integer/boolean lattice arrays that the
    notebook can pass straight to ``matplotlib`` ``c=...`` for the lattice
    scatter; this keeps the notebook free of teacher access.
    """
    s_full, frac_full, er_full, pr_full, top2_full = _pca_block(latents.H)
    s_lat, frac_lat, er_lat, pr_lat, top2_lat = _pca_block(latents.H_lat)

    color_keys: dict[str, np.ndarray] = {
        "i": latents.i_idx.astype(np.float64),
        "j": latents.j_idx.astype(np.float64),
        "i+j": (latents.i_idx + latents.j_idx).astype(np.float64),
        "i-j": (latents.i_idx - latents.j_idx).astype(np.float64),
        "T[i,j]": latents.T_at_lat.astype(np.float64),
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


# ---------------------------------------------------------------------------
# Linear probes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearProbeReport:
    """Held-out ``R^2`` of closed-form ridge regression from ``H`` to targets.

    Targets are split into two groups by domain:

    - **mesh** targets are evaluated on every kept mesh point (the rows of
      ``latents.H``).  Examples: ``x``, ``y``, ``is_pd``, ``V``.
    - **lattice** targets are evaluated only at lattice nodes (the rows of
      ``latents.H_lat``).  Examples: ``T[i, j]``, ``i + j``, ``i - j``,
      ``i * j``.

    Coefficients are stored with the bias appended as the last element so
    downstream callers (notably :func:`patching_probe`) can drop it via
    ``coef[:-1]`` to get the latent-direction vector.
    """

    r2: dict[str, float]                  # held-out R^2 per target
    r2_train: dict[str, float]            # in-sample R^2 per target (for context)
    coefficients: dict[str, np.ndarray]   # (D + 1,) including bias as last element
    targets_used: dict[str, str]          # "mesh" or "lattice"
    ridge: float


def _ridge_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form ridge with bias.

    Adds a constant column for the bias term and a ``sqrt(ridge) * I`` block
    that penalises the non-bias coefficients.  Solving the augmented system
    via :func:`numpy.linalg.lstsq` is equivalent to
    ``(X^T X + ridge * I_partial)^{-1} X^T y`` while remaining numerically
    stable for rank-deficient ``X``.
    """
    n, d = X_train.shape
    ones_train = np.ones((n, 1), dtype=X_train.dtype)
    ones_test = np.ones((X_test.shape[0], 1), dtype=X_test.dtype)
    X_train_b = np.concatenate([X_train, ones_train], axis=1)
    X_test_b = np.concatenate([X_test, ones_test], axis=1)
    pen = np.sqrt(ridge) * np.eye(d + 1, dtype=X_train.dtype)
    pen[-1, -1] = 0.0  # do not penalise the bias
    A = np.concatenate([X_train_b, pen], axis=0)
    b = np.concatenate(
        [y_train, np.zeros(d + 1, dtype=y_train.dtype)], axis=0
    )
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    y_pred_train = X_train_b @ coef
    y_pred_test = X_test_b @ coef
    return coef, y_pred_train, y_pred_test


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _ridge_probe_one(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float,
    seed: int,
    test_frac: float,
) -> tuple[float, float, np.ndarray]:
    """80/20 ridge probe; returns ``(r2_test, r2_train, coef_with_bias)``."""
    n = X.shape[0]
    if n < 4:
        return float("nan"), float("nan"), np.zeros(X.shape[1] + 1, dtype=X.dtype)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    coef, y_pred_train, y_pred_test = _ridge_fit_predict(
        X[train_idx], y[train_idx], X[test_idx], ridge
    )
    return (
        _r2(y[test_idx], y_pred_test),
        _r2(y[train_idx], y_pred_train),
        coef,
    )


def linear_probes(
    latents: LatentArrays,
    *,
    ridge: float = 1e-3,
    seed: int = 0,
    test_frac: float = 0.2,
) -> LinearProbeReport:
    """Ridge ``R^2`` from ``H`` onto a panel of arithmetic targets.

    Each target uses an independent train/test split (deterministic seed
    derived from ``seed`` plus a per-target offset) so the held-out ``R^2``
    values are not artefacts of a single shared split.
    """
    H_full = latents.H.astype(np.float64)
    H_lat = latents.H_lat.astype(np.float64)

    targets_used: dict[str, str] = {}
    r2_test: dict[str, float] = {}
    r2_train: dict[str, float] = {}
    coefs: dict[str, np.ndarray] = {}

    mesh_targets: dict[str, np.ndarray] = {
        "x": latents.xy_full_norm[:, 0].astype(np.float64),
        "y": latents.xy_full_norm[:, 1].astype(np.float64),
        "is_pd": latents.is_pd_full.astype(np.float64),
        "V": latents.V_full_norm.astype(np.float64),
    }
    lat_targets: dict[str, np.ndarray] = {
        "T[i,j]": latents.T_at_lat.astype(np.float64),
        "i+j": (latents.i_idx + latents.j_idx).astype(np.float64),
        "i-j": (latents.i_idx - latents.j_idx).astype(np.float64),
        "i*j": (latents.i_idx * latents.j_idx).astype(np.float64),
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
class PatchingReport:
    """Subspace activation patching across lattice rows and columns.

    For a row pair ``((i, j), (i, k))`` (same ``i``, different columns), we
    project ``h(a) = h(i, j)`` and ``h(b) = h(i, k)`` onto the linear-probe
    direction for ``y`` (the rank-1 ``y``-subspace) and replace ``h(a)``'s
    component along that direction with ``h(b)``'s.  Running ``head_a`` on
    the patched embedding gives ``v_swap``; we compare it against the teacher
    value at ``b``.

    The factorisation index

    ``(swap_mse - full_swap_mse) / (no_swap_mse - full_swap_mse)``

    is roughly in ``[0, 1]``: ``0`` means the rank-1 subspace patch perfectly
    recovers the row swap (the trunk has factorised ``(x, y)`` along the
    probe direction), ``1`` means no improvement over not patching at all.
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


def _factorisation_index(swap: float, no_swap: float, full_swap: float) -> float:
    denom = no_swap - full_swap
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return float("nan")
    return float((swap - full_swap) / denom)


def _enumerate_axis_pairs(
    valid: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate index pairs that differ only along ``axis`` (0 or 1).

    Returns two ``(P, 2)`` int arrays of grid indices for ``a`` and ``b``.
    ``axis=1`` gives row pairs (same ``i``, different ``j``), ``axis=0``
    gives column pairs.
    """
    Kx, Ky = valid.shape
    a_list: list[tuple[int, int]] = []
    b_list: list[tuple[int, int]] = []
    if axis == 1:
        for i in range(Kx):
            cols = np.where(valid[i])[0]
            for j in cols:
                for k in cols:
                    if j != k:
                        a_list.append((i, j))
                        b_list.append((i, k))
    else:
        for j in range(Ky):
            rows = np.where(valid[:, j])[0]
            for i in rows:
                for l in rows:
                    if i != l:
                        a_list.append((i, j))
                        b_list.append((l, j))
    if not a_list:
        empty = np.zeros((0, 2), dtype=np.int64)
        return empty, empty
    return np.asarray(a_list, dtype=np.int64), np.asarray(b_list, dtype=np.int64)


def patching_probe(
    student: Student,
    latents: LatentArrays,
    probes_report: LinearProbeReport,
) -> PatchingReport:
    """Subspace activation patching using the linear-probe directions.

    We use the rank-1 ``y``-direction from the linear probe onto ``y`` for
    row swaps, and the rank-1 ``x``-direction for column swaps.  See
    :class:`PatchingReport` for the metric definition.
    """
    H_grid = latents.H_lat_grid
    valid = np.isfinite(H_grid).all(axis=-1)

    # Teacher V (normalised, in head_a's prediction frame), gridded.
    V_grid = np.full((latents.Kx, latents.Ky), np.nan, dtype=np.float64)
    V_grid[latents.i_idx, latents.j_idx] = latents.V_lat_norm

    head_apply = jax.jit(jax.vmap(student.head_a))

    # Pre-compute student head_a at every lattice point (no-swap baseline).
    pred_lat = np.asarray(head_apply(jnp.asarray(latents.H_lat)))
    pred_grid = np.full((latents.Kx, latents.Ky), np.nan, dtype=np.float64)
    pred_grid[latents.i_idx, latents.j_idx] = pred_lat

    coef_y = probes_report.coefficients["y"][:-1].astype(np.float64)
    norm_y = float(np.linalg.norm(coef_y))
    u_y = np.zeros_like(coef_y) if norm_y < 1e-12 else (coef_y / norm_y)
    coef_x = probes_report.coefficients["x"][:-1].astype(np.float64)
    norm_x = float(np.linalg.norm(coef_x))
    u_x = np.zeros_like(coef_x) if norm_x < 1e-12 else (coef_x / norm_x)

    def _patch_along(
        u: np.ndarray, axis: int
    ) -> tuple[float, float, float, float, int]:
        a_idx, b_idx = _enumerate_axis_pairs(valid, axis)
        if a_idx.shape[0] == 0:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        h_a = H_grid[a_idx[:, 0], a_idx[:, 1]]                          # (P, D)
        h_b = H_grid[b_idx[:, 0], b_idx[:, 1]]
        u_dt = u.astype(h_a.dtype)
        delta_proj = (h_b - h_a) @ u_dt                                 # (P,)
        h_swap = h_a + delta_proj[:, None] * u_dt[None, :]
        v_target = V_grid[b_idx[:, 0], b_idx[:, 1]]
        v_no_swap = pred_grid[a_idx[:, 0], a_idx[:, 1]]
        v_full_swap = pred_grid[b_idx[:, 0], b_idx[:, 1]]
        v_swap = np.asarray(head_apply(jnp.asarray(h_swap)))
        no_mse = float(np.mean((v_no_swap - v_target) ** 2))
        sw_mse = float(np.mean((v_swap - v_target) ** 2))
        full_mse = float(np.mean((v_full_swap - v_target) ** 2))
        return no_mse, sw_mse, full_mse, _factorisation_index(sw_mse, no_mse, full_mse), int(a_idx.shape[0])

    no_row, sw_row, full_row, fi_row, n_row = _patch_along(u_y, axis=1)
    no_col, sw_col, full_col, fi_col, n_col = _patch_along(u_x, axis=0)

    return PatchingReport(
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
# Hessian alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HessianAlignmentReport:
    """Per-lattice-node alignment between student and teacher Hessians.

    All eigenvalue arrays are in ascending order (``[bot, top]``).  Both
    Hessians are evaluated in normalised input coordinates (the student's
    natural frame; teacher Hessian targets in :class:`SobolevDataset` have
    been transformed by the same input/output normalisation).
    """

    student_min_eig: np.ndarray      # (K_lat,) student smaller eigvalue
    teacher_min_eig: np.ndarray      # (K_lat,) teacher smaller eigvalue
    student_max_eig: np.ndarray      # (K_lat,) student larger eigvalue
    teacher_max_eig: np.ndarray      # (K_lat,) teacher larger eigvalue
    both_pd_fraction: float
    eigvec_cosine_top: np.ndarray    # (K_lat,) abs cosine of largest eigvecs
    eigvec_cosine_bot: np.ndarray    # (K_lat,) abs cosine of smallest eigvecs
    eigval_ratio_top: np.ndarray     # (K_lat,) student_max / teacher_max
    eigval_ratio_bot: np.ndarray     # (K_lat,) student_min / teacher_min


def hessian_alignment(
    student: Student,
    dataset: SobolevDataset,
) -> HessianAlignmentReport:
    """Eigen-alignment of student vs teacher Hessian at lattice nodes."""
    is_node = np.asarray(dataset.is_node, dtype=bool)
    if not is_node.any():
        empty = np.zeros((0,), dtype=np.float64)
        return HessianAlignmentReport(
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

    xy_lat = jnp.asarray(np.asarray(dataset.xy)[is_node])
    Hxx = np.asarray(dataset.Hxx)[is_node]
    Hxy = np.asarray(dataset.Hxy)[is_node]
    Hyy = np.asarray(dataset.Hyy)[is_node]
    teacher_H = np.stack(
        [
            np.stack([Hxx, Hxy], axis=-1),
            np.stack([Hxy, Hyy], axis=-1),
        ],
        axis=-2,
    )                                           # (K_lat, 2, 2)

    hess_fn = jax.vmap(jax.hessian(lambda inp: f_arith(student, inp)))
    student_H = np.asarray(hess_fn(xy_lat))     # (K_lat, 2, 2)

    # Symmetrise (cancel numerical noise from the autodiff pass).
    student_H_sym = 0.5 * (student_H + np.swapaxes(student_H, -1, -2))
    teacher_H_sym = 0.5 * (teacher_H + np.swapaxes(teacher_H, -1, -2))
    s_evals, s_evecs = np.linalg.eigh(student_H_sym)
    t_evals, t_evecs = np.linalg.eigh(teacher_H_sym)

    s_min = s_evals[:, 0]
    s_max = s_evals[:, 1]
    t_min = t_evals[:, 0]
    t_max = t_evals[:, 1]

    both_pd = (s_min > 0.0) & (t_min > 0.0)
    both_pd_fraction = float(np.mean(both_pd.astype(np.float64))) if both_pd.size else float("nan")

    # Eigvecs are sign-ambiguous, so use the absolute cosine.
    cos_bot = np.abs(np.einsum("kd,kd->k", s_evecs[:, :, 0], t_evecs[:, :, 0]))
    cos_top = np.abs(np.einsum("kd,kd->k", s_evecs[:, :, 1], t_evecs[:, :, 1]))

    safe_top = np.where(np.abs(t_max) > 1e-12, t_max, np.nan)
    safe_bot = np.where(np.abs(t_min) > 1e-12, t_min, np.nan)
    ratio_top = s_max / safe_top
    ratio_bot = s_min / safe_bot

    return HessianAlignmentReport(
        student_min_eig=s_min,
        teacher_min_eig=t_min,
        student_max_eig=s_max,
        teacher_max_eig=t_max,
        both_pd_fraction=both_pd_fraction,
        eigvec_cosine_top=cos_top,
        eigvec_cosine_bot=cos_bot,
        eigval_ratio_top=ratio_top,
        eigval_ratio_bot=ratio_bot,
    )


# ---------------------------------------------------------------------------
# Bundle / driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeBundle:
    """Aggregate of every probe report plus the diagnostics report.

    Use :meth:`verdict` to compute the three-criterion structural-resolution
    test described at the top of this module.
    """

    latents: LatentArrays
    pca: PCAReport
    probes: LinearProbeReport
    patching: PatchingReport
    hessian: HessianAlignmentReport
    diagnostics: DiagnosticsReport | None = None

    def verdict(
        self,
        *,
        max_grad_angle_deg: float = 10.0,
        min_pd_auroc: float = 0.95,
        min_arith_r2: float = 0.9,
    ) -> dict[str, bool]:
        """Three-criterion test: gradient-following, PD certificate, arithmetic axes."""
        if self.diagnostics is None:
            grad_ok = False
            pd_ok = False
        else:
            grad_ok = bool(
                np.isfinite(self.diagnostics.node_grad_angle_deg)
                and self.diagnostics.node_grad_angle_deg < max_grad_angle_deg
            )
            pd_ok = bool(
                np.isfinite(self.diagnostics.energy_pd_auroc)
                and self.diagnostics.energy_pd_auroc > min_pd_auroc
            )
        r2_ipj = self.probes.r2.get("i+j", float("nan"))
        r2_T = self.probes.r2.get("T[i,j]", float("nan"))
        arith_r2 = max(
            r2_ipj if np.isfinite(r2_ipj) else -np.inf,
            r2_T if np.isfinite(r2_T) else -np.inf,
        )
        arith_ok = bool(arith_r2 > min_arith_r2)
        return {
            "grad_following": grad_ok,
            "pd_certificate": pd_ok,
            "arithmetic_axes_linear": arith_ok,
            "all_three": bool(grad_ok and pd_ok and arith_ok),
        }


def run_all_probes(
    student: Student,
    dataset: SobolevDataset,
    teacher: MeshTeacher,
    *,
    diagnostics: DiagnosticsReport | None = None,
    ridge: float = 1e-3,
    seed: int = 0,
) -> ProbeBundle:
    """Run all five probes in dependency order.

    Order matters because :func:`patching_probe` depends on the linear-probe
    coefficients to define the patching subspace.
    """
    latents = compute_latents(student, dataset, teacher)
    pca = latent_pca(latents)
    probes_rep = linear_probes(latents, ridge=ridge, seed=seed)
    patching = patching_probe(student, latents, probes_rep)
    hessian = hessian_alignment(student, dataset)
    return ProbeBundle(
        latents=latents,
        pca=pca,
        probes=probes_rep,
        patching=patching,
        hessian=hessian,
        diagnostics=diagnostics,
    )


__all__ = [
    "HessianAlignmentReport",
    "LatentArrays",
    "LinearProbeReport",
    "PCAReport",
    "PatchingReport",
    "ProbeBundle",
    "compute_latents",
    "hessian_alignment",
    "latent_pca",
    "linear_probes",
    "patching_probe",
    "run_all_probes",
]
