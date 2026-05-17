"""Flatten the :class:`~teacher.MeshTeacher` into a JAX-ready training set.

Responsibilities of this module:

- Flatten the 2D mesh into ``(N, 2)`` input + per-point teacher targets.
- Apply input/output normalization (inputs to ``[-1, 1]``, output mean/std
  centring) so the student sees a well-conditioned problem.  Derivative targets
  are rescaled with the same constants so Sobolev terms stay self-consistent.
- Provide a sampling helper :func:`sample_minibatch` mixing **lattice**,
  **uniform-mesh**, and **Chebyshev-clustered** samples in user-controlled
  proportions.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .teacher import MeshTeacher


@dataclass(frozen=True)
class Normalisation:
    """Affine input/output transform constants.

    Inputs: ``x_norm = (x - x_center) / x_scale`` (target ``[-1, 1]`` range).
    Output: ``v_norm = (V - v_mean) / v_std``.

    Derivatives transform as ``dV/dx_norm = (dV/dx) * x_scale / v_std`` and
    ``d2V/dx_norm^2 = (d2V/dx^2) * x_scale^2 / v_std`` so the Sobolev loss stays
    consistent in normalised coordinates.
    """

    x_center: float
    x_scale: float
    y_center: float
    y_scale: float
    v_mean: float
    v_std: float
    vm_mean: float
    vm_std: float

    def normalise_inputs(self, xy: jnp.ndarray) -> jnp.ndarray:
        cx = jnp.asarray([self.x_center, self.y_center])
        sc = jnp.asarray([self.x_scale, self.y_scale])
        return (xy - cx) / sc

    def denormalise_value(self, v: jnp.ndarray) -> jnp.ndarray:
        return v * self.v_std + self.v_mean

    def denormalise_inputs(self, xy: jnp.ndarray) -> jnp.ndarray:
        cx = jnp.asarray([self.x_center, self.y_center])
        sc = jnp.asarray([self.x_scale, self.y_scale])
        return xy * sc + cx


@dataclass(frozen=True)
class SobolevDataset:
    """Flat, JAX-friendly view of the teacher mesh.

    Every per-point array has shape ``(N,)`` (or ``(N, 2)`` for the input
    coordinates) where ``N = Nx * Ny``.  Arrays are :class:`jax.numpy.ndarray`
    so they can be sliced and passed straight into ``jit``-compiled losses.

    Use :func:`sample_minibatch` (or the built-in ``minibatch_indices``) to
    select a subset that mixes lattice / uniform / Chebyshev draws.
    """

    xy: jnp.ndarray         # (N, 2) coordinates (normalised)
    xy_raw: jnp.ndarray     # (N, 2) original coordinates (for plotting)
    V: jnp.ndarray          # (N,) f_H value, normalised
    GX: jnp.ndarray         # (N,) d/dx_norm f_H, normalised
    GY: jnp.ndarray         # (N,) d/dy_norm f_H, normalised
    Hxx: jnp.ndarray        # (N,) d^2/dx_norm^2 f_H, normalised
    Hxy: jnp.ndarray        # (N,) d^2/dx_norm dy_norm f_H, normalised
    Hyy: jnp.ndarray        # (N,) d^2/dy_norm^2 f_H, normalised
    V_M: jnp.ndarray        # (N,) f_M value, normalised by (vm_mean, vm_std)
    is_node: jnp.ndarray    # (N,) bool: lattice node sample
    is_pd: jnp.ndarray      # (N,) bool: lattice node with PD f_M Hessian
    cheb_weight: jnp.ndarray  # (N,) sample weights emphasising Chebyshev nodes
    norm: Normalisation

    @property
    def n(self) -> int:
        return int(self.xy.shape[0])


def _chebyshev_weights(xs: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Per-point density weights peaked near the Chebyshev abscissae.

    Cheap analytic surrogate: weight inversely proportional to ``sqrt(1 - t^2) +
    eps`` where ``t`` is ``x`` mapped to ``[-1, 1]``.  This produces the same
    O(1/sqrt(1-t^2)) clustering at the endpoints that Chebyshev nodes have,
    without requiring a discrete resampling step.
    """
    t = 2.0 * (xs - x_min) / max(x_max - x_min, 1e-12) - 1.0
    return 1.0 / np.sqrt(np.clip(1.0 - t * t, 1e-3, None))


def build_dataset(
    teacher: MeshTeacher,
    *,
    target_norm_range: float = 1.0,
    residual_band: float | None = None,
    grad_clip: float | None = None,
    hess_clip: float | None = None,
    value_clip_quantile: float | None = None,
) -> SobolevDataset:
    """Flatten ``teacher`` into a normalised :class:`SobolevDataset`.

    ``target_norm_range`` controls the half-width of the input range after
    normalisation; ``1.0`` maps node extremes to ``[-1, +1]``.

    ``residual_band`` (preferred clipping mode): keep a mesh point only if the
    teacher value satisfies ``|V - (x + y)| <= residual_band``, i.e. it is
    within a band around the addition-table reference plane.  This mirrors the
    teacher notebook's ``Z_DISPLAY_RES_CAP`` and is the right defence against
    the high-degree barycentric Hermite Runge-style oscillation off the
    lattice.  When ``None`` (default), the band is auto-set to
    ``2 * (max(nodes_x) + max(nodes_y))`` (the same heuristic the teacher
    notebook uses for visualisation).  Lattice nodes are always kept.

    ``grad_clip`` / ``hess_clip``: optional absolute caps on the gradient L2
    norm and Hessian Frobenius norm of teacher targets.  Mesh points exceeding
    either cap are dropped.  When ``None`` (default), they are auto-set from
    the per-axis maximum slope magnitude (``max|Dx| + max|Dy|``) and a 10x
    multiplier.  This is essential at high K where the polynomial Hessian
    blows up between lattice cells even when ``V`` is in band.

    ``value_clip_quantile`` (legacy option): if set, drop mesh points whose
    teacher value/gradient/Hessian magnitudes fall outside the central
    ``[q, 1 - q]`` quantile band.  Less robust than ``residual_band``; kept for
    edge cases where the addition-table reference is not meaningful.
    """
    xs = np.asarray(teacher.xs)
    ys = np.asarray(teacher.ys)
    nodes_x = np.asarray(teacher.nodes_x)
    nodes_y = np.asarray(teacher.nodes_y)

    x_center = 0.5 * (float(nodes_x.min()) + float(nodes_x.max()))
    x_half = max(0.5 * (float(nodes_x.max()) - float(nodes_x.min())), 1e-12)
    x_scale = x_half / target_norm_range
    y_center = 0.5 * (float(nodes_y.min()) + float(nodes_y.max()))
    y_half = max(0.5 * (float(nodes_y.max()) - float(nodes_y.min())), 1e-12)
    y_scale = y_half / target_norm_range

    V = np.asarray(teacher.V).ravel()
    GX = np.asarray(teacher.GX).ravel()
    GY = np.asarray(teacher.GY).ravel()
    Hxx = np.asarray(teacher.Hxx).ravel()
    Hxy = np.asarray(teacher.Hxy).ravel()
    Hyy = np.asarray(teacher.Hyy).ravel()
    V_M = np.asarray(teacher.V_M).ravel()
    is_node = np.asarray(teacher.is_node).ravel()
    is_pd = np.asarray(teacher.is_pd).ravel()

    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    xy_raw_full = np.stack([XX.ravel(), YY.ravel()], axis=-1)        # (N_full, 2)
    n_full = xy_raw_full.shape[0]
    reference_plane = (xy_raw_full[:, 0] + xy_raw_full[:, 1])

    finite_mask = (
        np.isfinite(V) & np.isfinite(GX) & np.isfinite(GY)
        & np.isfinite(Hxx) & np.isfinite(Hxy) & np.isfinite(Hyy)
        & np.isfinite(V_M)
    )
    keep = finite_mask.copy()

    if residual_band is None:
        residual_band = 2.0 * (
            float(np.abs(nodes_x).max()) + float(np.abs(nodes_y).max())
        )
    if residual_band is not None and residual_band > 0.0:
        with np.errstate(invalid="ignore", over="ignore"):
            within_band = np.where(
                finite_mask, np.abs(V - reference_plane) <= float(residual_band), False
            )
        keep &= within_band

    Dx = np.asarray(teacher.Dx)
    Dy = np.asarray(teacher.Dy)
    if grad_clip is None:
        grad_clip = 10.0 * (float(np.abs(Dx).max()) + float(np.abs(Dy).max()) + 1.0)
    if hess_clip is None:
        hess_clip = 10.0 * residual_band

    if grad_clip is not None and grad_clip > 0.0:
        with np.errstate(invalid="ignore", over="ignore"):
            g_norm = np.sqrt(np.where(finite_mask, GX * GX + GY * GY, np.inf))
        keep &= g_norm <= float(grad_clip)
    if hess_clip is not None and hess_clip > 0.0:
        with np.errstate(invalid="ignore", over="ignore"):
            h_norm = np.sqrt(
                np.where(
                    finite_mask, Hxx * Hxx + 2.0 * Hxy * Hxy + Hyy * Hyy, np.inf
                )
            )
        keep &= h_norm <= float(hess_clip)

    if value_clip_quantile is not None and value_clip_quantile > 0.0:
        q = float(value_clip_quantile)
        with np.errstate(invalid="ignore", over="ignore"):
            v_abs = np.abs(np.where(finite_mask, V, 0.0))
            g_norm = np.sqrt(np.where(finite_mask, GX * GX + GY * GY, 0.0))
            h_norm = np.sqrt(
                np.where(finite_mask, Hxx * Hxx + 2.0 * Hxy * Hxy + Hyy * Hyy, 0.0)
            )
        finite_idx = np.flatnonzero(finite_mask)
        if finite_idx.size:
            v_hi = float(np.quantile(v_abs[finite_idx], 1.0 - q))
            g_hi = float(np.quantile(g_norm[finite_idx], 1.0 - q))
            h_hi = float(np.quantile(h_norm[finite_idx], 1.0 - q))
            outliers = (v_abs > v_hi) | (g_norm > g_hi) | (h_norm > h_hi)
            keep &= ~outliers

    # Always retain lattice nodes regardless of magnitude (they are the
    # Birkhoff supervision anchors).
    keep |= is_node & finite_mask

    xy_raw = xy_raw_full[keep]
    V = V[keep]
    GX = GX[keep]
    GY = GY[keep]
    Hxx = Hxx[keep]
    Hxy = Hxy[keep]
    Hyy = Hyy[keep]
    V_M = V_M[keep]
    is_node = is_node[keep]
    is_pd = is_pd[keep]

    # Robust normalisation constants from the kept points.
    v_mean = float(np.median(V))
    v_std = float(np.median(np.abs(V - v_mean)) * 1.4826) or float(np.std(V)) or 1.0
    vm_mean = float(np.median(V_M))
    vm_std = float(np.median(np.abs(V_M - vm_mean)) * 1.4826) or float(np.std(V_M)) or 1.0

    xy = (xy_raw - np.array([x_center, y_center])) / np.array([x_scale, y_scale])

    V_norm = (V - v_mean) / v_std
    GX_norm = GX * (x_scale / v_std)
    GY_norm = GY * (y_scale / v_std)
    Hxx_norm = Hxx * (x_scale * x_scale / v_std)
    Hxy_norm = Hxy * (x_scale * y_scale / v_std)
    Hyy_norm = Hyy * (y_scale * y_scale / v_std)
    VM_norm = (V_M - vm_mean) / vm_std

    cheb_w_x = _chebyshev_weights(xs, float(nodes_x.min()), float(nodes_x.max()))
    cheb_w_y = _chebyshev_weights(ys, float(nodes_y.min()), float(nodes_y.max()))
    cheb_w_full = (cheb_w_x[:, None] * cheb_w_y[None, :]).ravel()
    cheb_w = cheb_w_full[keep]

    norm = Normalisation(
        x_center=x_center,
        x_scale=x_scale,
        y_center=y_center,
        y_scale=y_scale,
        v_mean=v_mean,
        v_std=v_std,
        vm_mean=vm_mean,
        vm_std=vm_std,
    )

    return SobolevDataset(
        xy=jnp.asarray(xy),
        xy_raw=jnp.asarray(xy_raw),
        V=jnp.asarray(V_norm),
        GX=jnp.asarray(GX_norm),
        GY=jnp.asarray(GY_norm),
        Hxx=jnp.asarray(Hxx_norm),
        Hxy=jnp.asarray(Hxy_norm),
        Hyy=jnp.asarray(Hyy_norm),
        V_M=jnp.asarray(VM_norm),
        is_node=jnp.asarray(is_node),
        is_pd=jnp.asarray(is_pd),
        cheb_weight=jnp.asarray(cheb_w),
        norm=norm,
    )


def sample_minibatch(
    key: jax.Array,
    dataset: SobolevDataset,
    batch_size: int,
    *,
    lattice_frac: float = 0.25,
    chebyshev_frac: float = 0.5,
) -> jnp.ndarray:
    """Sample mixed indices: ``lattice`` + ``Chebyshev-weighted`` + ``uniform``.

    Returns an int array of shape ``(batch_size,)`` with the chosen flat
    indices into ``dataset``.

    The fractions ``lattice_frac`` and ``chebyshev_frac`` must sum to <= 1; the
    remainder is drawn uniformly from the full mesh.  When ``lattice_frac > 0``
    but no lattice nodes exist in the dataset, the lattice quota silently falls
    back to uniform sampling.
    """
    if lattice_frac < 0.0 or chebyshev_frac < 0.0:
        raise ValueError("Sampling fractions must be non-negative")
    if lattice_frac + chebyshev_frac > 1.0 + 1e-9:
        raise ValueError(
            f"lattice_frac + chebyshev_frac = {lattice_frac + chebyshev_frac} > 1.0"
        )

    n_lat = int(round(batch_size * lattice_frac))
    n_cheb = int(round(batch_size * chebyshev_frac))
    n_unif = batch_size - n_lat - n_cheb

    keys = jax.random.split(key, 3)

    is_node = dataset.is_node
    n_total = dataset.n
    lat_idx_pool = jnp.where(is_node, size=n_total, fill_value=-1)[0]
    n_lat_real = int(jnp.sum(is_node).item())
    if n_lat_real == 0:
        n_unif += n_lat
        n_lat = 0

    pieces = []
    if n_lat > 0:
        # Sample with replacement from the first n_lat_real entries of lat_idx_pool.
        lat_idx = jax.random.randint(
            keys[0], shape=(n_lat,), minval=0, maxval=n_lat_real
        )
        pieces.append(lat_idx_pool[lat_idx])
    if n_cheb > 0:
        weights = dataset.cheb_weight / jnp.sum(dataset.cheb_weight)
        pieces.append(
            jax.random.choice(keys[1], n_total, shape=(n_cheb,), replace=True, p=weights)
        )
    if n_unif > 0:
        pieces.append(
            jax.random.randint(keys[2], shape=(n_unif,), minval=0, maxval=n_total)
        )

    return jnp.concatenate(pieces, axis=0)


def select(dataset: SobolevDataset, indices: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Gather every per-point field at ``indices``; returns a plain ``dict``."""
    return {
        "xy": dataset.xy[indices],
        "V": dataset.V[indices],
        "GX": dataset.GX[indices],
        "GY": dataset.GY[indices],
        "Hxx": dataset.Hxx[indices],
        "Hxy": dataset.Hxy[indices],
        "Hyy": dataset.Hyy[indices],
        "V_M": dataset.V_M[indices],
        "is_node": dataset.is_node[indices],
        "is_pd": dataset.is_pd[indices],
    }


__all__ = [
    "Normalisation",
    "SobolevDataset",
    "build_dataset",
    "sample_minibatch",
    "select",
]
