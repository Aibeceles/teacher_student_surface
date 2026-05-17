"""Flatten the :class:`~teacher.CharacterMeshTeacher` into a JAX training set.

Re/Im split: every per-point complex teacher field becomes two real arrays.
Output normalisation runs independently on the Re and Im channels because
they have different scales near the unit circle (Im has zero mean by
symmetry; Re does not).  Inputs are normalised exactly as in the real
:mod:`sobolev_distill.dataset`.

``sample_minibatch`` is reused from :mod:`sobolev_distill.dataset` because
the sampling logic only depends on ``is_node`` / ``cheb_weight``, which we
provide identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from sobolev_distill.dataset import _chebyshev_weights, sample_minibatch  # noqa: F401

from .teacher import CharacterMeshTeacher


@dataclass(frozen=True)
class CharacterNormalisation:
    """Affine input + per-channel output transform constants.

    Inputs: ``x_norm = (x - x_center) / x_scale`` (target ``[-1, 1]``).
    Outputs (per channel ``c in {re, im}``):
    ``v_norm = (V - v_mean[c]) / v_std[c]``.
    Derivatives transform per channel by ``x_scale / v_std[c]`` (grad)
    and ``x_scale**2 / v_std[c]`` (Hessian).
    """

    x_center: float
    x_scale: float
    y_center: float
    y_scale: float
    v_re_mean: float
    v_re_std: float
    v_im_mean: float
    v_im_std: float
    vm_mean: float
    vm_std: float

    def normalise_inputs(self, xy: jnp.ndarray) -> jnp.ndarray:
        cx = jnp.asarray([self.x_center, self.y_center])
        sc = jnp.asarray([self.x_scale, self.y_scale])
        return (xy - cx) / sc

    def denormalise_value(self, v: jnp.ndarray) -> jnp.ndarray:
        """Per-channel denormalisation; ``v`` shape ``(..., 2)`` (re, im)."""
        means = jnp.asarray([self.v_re_mean, self.v_im_mean])
        stds = jnp.asarray([self.v_re_std, self.v_im_std])
        return v * stds + means

    def denormalise_inputs(self, xy: jnp.ndarray) -> jnp.ndarray:
        cx = jnp.asarray([self.x_center, self.y_center])
        sc = jnp.asarray([self.x_scale, self.y_scale])
        return xy * sc + cx


@dataclass(frozen=True)
class CharacterSobolevDataset:
    """Flat (Re, Im) view of the character teacher mesh.

    Per-point arrays have shape ``(N,)``.  Outputs are stored as separate
    Re/Im channels so the loss can MSE them independently without ever
    constructing complex tensors inside the JAX student.
    """

    xy: jnp.ndarray         # (N, 2) coordinates (normalised)
    xy_raw: jnp.ndarray     # (N, 2) original coordinates
    V_re: jnp.ndarray
    V_im: jnp.ndarray
    GX_re: jnp.ndarray
    GX_im: jnp.ndarray
    GY_re: jnp.ndarray
    GY_im: jnp.ndarray
    Hxx_re: jnp.ndarray
    Hxx_im: jnp.ndarray
    Hxy_re: jnp.ndarray
    Hxy_im: jnp.ndarray
    Hyy_re: jnp.ndarray
    Hyy_im: jnp.ndarray
    V_M: jnp.ndarray
    is_node: jnp.ndarray
    is_pd: jnp.ndarray
    cheb_weight: jnp.ndarray
    axis_target_cos_x: jnp.ndarray  # cos(2*pi*x_raw/p) per row
    axis_target_sin_x: jnp.ndarray  # sin(2*pi*x_raw/p) per row
    axis_target_cos_y: jnp.ndarray  # cos(2*pi*y_raw/p) per row
    axis_target_sin_y: jnp.ndarray  # sin(2*pi*y_raw/p) per row
    norm: CharacterNormalisation
    modulus: int

    @property
    def n(self) -> int:
        return int(self.xy.shape[0])


def build_character_dataset(
    teacher: CharacterMeshTeacher,
    *,
    target_norm_range: float = 1.0,
    residual_cap: float | None = None,
) -> CharacterSobolevDataset:
    """Flatten ``teacher`` into a normalised :class:`CharacterSobolevDataset`.

    ``residual_cap`` (defaults to ``2.0``) drops mesh points where
    ``|V| > residual_cap`` to defend against Runge-style oscillation off
    the lattice.  The character target lies on the unit circle so any
    ``|V|`` substantially above 1 is interpolation noise.  Lattice nodes
    are always retained.
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

    V_re = np.asarray(teacher.V_re).ravel()
    V_im = np.asarray(teacher.V_im).ravel()
    GX_re = np.asarray(teacher.GX_re).ravel()
    GX_im = np.asarray(teacher.GX_im).ravel()
    GY_re = np.asarray(teacher.GY_re).ravel()
    GY_im = np.asarray(teacher.GY_im).ravel()
    Hxx_re = np.asarray(teacher.Hxx_re).ravel()
    Hxx_im = np.asarray(teacher.Hxx_im).ravel()
    Hxy_re = np.asarray(teacher.Hxy_re).ravel()
    Hxy_im = np.asarray(teacher.Hxy_im).ravel()
    Hyy_re = np.asarray(teacher.Hyy_re).ravel()
    Hyy_im = np.asarray(teacher.Hyy_im).ravel()
    V_M = np.asarray(teacher.V_M).ravel()
    is_node = np.asarray(teacher.is_node).ravel()
    is_pd = np.asarray(teacher.is_pd).ravel()

    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    xy_raw_full = np.stack([XX.ravel(), YY.ravel()], axis=-1)

    finite_fields = (
        V_re, V_im, GX_re, GX_im, GY_re, GY_im,
        Hxx_re, Hxx_im, Hxy_re, Hxy_im, Hyy_re, Hyy_im, V_M,
    )
    finite_mask = np.ones(xy_raw_full.shape[0], dtype=bool)
    for arr in finite_fields:
        finite_mask &= np.isfinite(arr)

    keep = finite_mask.copy()
    if residual_cap is None:
        residual_cap = 2.0
    if residual_cap is not None and residual_cap > 0.0:
        with np.errstate(invalid="ignore", over="ignore"):
            v_abs = np.sqrt(np.where(finite_mask, V_re * V_re + V_im * V_im, np.inf))
        keep &= v_abs <= float(residual_cap)
    keep |= is_node & finite_mask  # always retain lattice nodes

    xy_raw = xy_raw_full[keep]
    V_re = V_re[keep]
    V_im = V_im[keep]
    GX_re = GX_re[keep]
    GX_im = GX_im[keep]
    GY_re = GY_re[keep]
    GY_im = GY_im[keep]
    Hxx_re = Hxx_re[keep]
    Hxx_im = Hxx_im[keep]
    Hxy_re = Hxy_re[keep]
    Hxy_im = Hxy_im[keep]
    Hyy_re = Hyy_re[keep]
    Hyy_im = Hyy_im[keep]
    V_M = V_M[keep]
    is_node = is_node[keep]
    is_pd = is_pd[keep]

    def _robust_stats(arr: np.ndarray) -> tuple[float, float]:
        mean = float(np.median(arr))
        std = float(np.median(np.abs(arr - mean)) * 1.4826) or float(np.std(arr)) or 1.0
        return mean, std

    v_re_mean, v_re_std = _robust_stats(V_re)
    v_im_mean, v_im_std = _robust_stats(V_im)
    vm_mean, vm_std = _robust_stats(V_M)

    xy = (xy_raw - np.array([x_center, y_center])) / np.array([x_scale, y_scale])

    V_re_n = (V_re - v_re_mean) / v_re_std
    V_im_n = (V_im - v_im_mean) / v_im_std
    GX_re_n = GX_re * (x_scale / v_re_std)
    GX_im_n = GX_im * (x_scale / v_im_std)
    GY_re_n = GY_re * (y_scale / v_re_std)
    GY_im_n = GY_im * (y_scale / v_im_std)
    Hxx_re_n = Hxx_re * (x_scale * x_scale / v_re_std)
    Hxx_im_n = Hxx_im * (x_scale * x_scale / v_im_std)
    Hxy_re_n = Hxy_re * (x_scale * y_scale / v_re_std)
    Hxy_im_n = Hxy_im * (x_scale * y_scale / v_im_std)
    Hyy_re_n = Hyy_re * (y_scale * y_scale / v_re_std)
    Hyy_im_n = Hyy_im * (y_scale * y_scale / v_im_std)
    VM_n = (V_M - vm_mean) / vm_std

    cheb_w_x = _chebyshev_weights(xs, float(nodes_x.min()), float(nodes_x.max()))
    cheb_w_y = _chebyshev_weights(ys, float(nodes_y.min()), float(nodes_y.max()))
    cheb_w_full = (cheb_w_x[:, None] * cheb_w_y[None, :]).ravel()
    cheb_w = cheb_w_full[keep]

    # Axis targets in raw coords: at lattice node (i, j) these equal the
    # canonical p-th-root-of-unity decomposition (cos 2 pi i/p, sin 2 pi i/p,
    # cos 2 pi j/p, sin 2 pi j/p).  Off-lattice they extend smoothly via the
    # same trigonometric formula, providing pressure for the trunk to expose
    # linearly decodable axis features.
    omega = 2.0 * math.pi / float(teacher.modulus)
    axis_target_cos_x = np.cos(omega * xy_raw[:, 0]).astype(np.float32)
    axis_target_sin_x = np.sin(omega * xy_raw[:, 0]).astype(np.float32)
    axis_target_cos_y = np.cos(omega * xy_raw[:, 1]).astype(np.float32)
    axis_target_sin_y = np.sin(omega * xy_raw[:, 1]).astype(np.float32)

    norm = CharacterNormalisation(
        x_center=x_center,
        x_scale=x_scale,
        y_center=y_center,
        y_scale=y_scale,
        v_re_mean=v_re_mean,
        v_re_std=v_re_std,
        v_im_mean=v_im_mean,
        v_im_std=v_im_std,
        vm_mean=vm_mean,
        vm_std=vm_std,
    )

    return CharacterSobolevDataset(
        xy=jnp.asarray(xy),
        xy_raw=jnp.asarray(xy_raw),
        V_re=jnp.asarray(V_re_n),
        V_im=jnp.asarray(V_im_n),
        GX_re=jnp.asarray(GX_re_n),
        GX_im=jnp.asarray(GX_im_n),
        GY_re=jnp.asarray(GY_re_n),
        GY_im=jnp.asarray(GY_im_n),
        Hxx_re=jnp.asarray(Hxx_re_n),
        Hxx_im=jnp.asarray(Hxx_im_n),
        Hxy_re=jnp.asarray(Hxy_re_n),
        Hxy_im=jnp.asarray(Hxy_im_n),
        Hyy_re=jnp.asarray(Hyy_re_n),
        Hyy_im=jnp.asarray(Hyy_im_n),
        V_M=jnp.asarray(VM_n),
        is_node=jnp.asarray(is_node),
        is_pd=jnp.asarray(is_pd),
        cheb_weight=jnp.asarray(cheb_w),
        axis_target_cos_x=jnp.asarray(axis_target_cos_x),
        axis_target_sin_x=jnp.asarray(axis_target_sin_x),
        axis_target_cos_y=jnp.asarray(axis_target_cos_y),
        axis_target_sin_y=jnp.asarray(axis_target_sin_y),
        norm=norm,
        modulus=int(teacher.modulus),
    )


def select_character(
    dataset: CharacterSobolevDataset,
    indices: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Gather every per-point field at ``indices``; returns a plain ``dict``.

    The returned ``axis_target`` is a ``(B, 4)`` array stacking
    ``[cos(2 pi x_raw/p), sin(2 pi x_raw/p), cos(2 pi y_raw/p),
    sin(2 pi y_raw/p)]`` -- consumed by the auxiliary axis loss when
    ``LossWeights.axis > 0`` and the student carries an ``axis_probe``
    head.  Always present so the loss code path is shape-stable; the
    field is unused when ``weights.axis == 0`` or ``axis_probe is None``.
    """
    axis_target = jnp.stack(
        [
            dataset.axis_target_cos_x[indices],
            dataset.axis_target_sin_x[indices],
            dataset.axis_target_cos_y[indices],
            dataset.axis_target_sin_y[indices],
        ],
        axis=-1,
    )
    return {
        "xy": dataset.xy[indices],
        "V_re": dataset.V_re[indices],
        "V_im": dataset.V_im[indices],
        "GX_re": dataset.GX_re[indices],
        "GX_im": dataset.GX_im[indices],
        "GY_re": dataset.GY_re[indices],
        "GY_im": dataset.GY_im[indices],
        "Hxx_re": dataset.Hxx_re[indices],
        "Hxx_im": dataset.Hxx_im[indices],
        "Hxy_re": dataset.Hxy_re[indices],
        "Hxy_im": dataset.Hxy_im[indices],
        "Hyy_re": dataset.Hyy_re[indices],
        "Hyy_im": dataset.Hyy_im[indices],
        "V_M": dataset.V_M[indices],
        "is_node": dataset.is_node[indices],
        "is_pd": dataset.is_pd[indices],
        "axis_target": axis_target,
    }


__all__ = [
    "CharacterNormalisation",
    "CharacterSobolevDataset",
    "build_character_dataset",
    "sample_minibatch",
    "select_character",
]
