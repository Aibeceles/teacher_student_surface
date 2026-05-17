"""Materialize the character Birkhoff-Hermite teacher fields on a 2D mesh.

The complex value/grad/Hessian fields of ``f_H : R^2 -> C`` are split into
real and imaginary companions so the downstream JAX student stays purely
real-valued.  The energy field ``V_M = (|f_H|**2 - 1)**2 + lam * Z_W`` is
real by construction (zero at every lattice node since ``|f_H| = 1``
there).  The PD label ``is_pd`` and the per-node minimum eigenvalue are
inherited from the modulus-energy Hessian
:func:`graphic_zero_character.character_birkhoff.modulus_energy_diagnostics_on_device`.

This module reuses :func:`sobolev_distill.teacher._vectorised_axis_basis`
verbatim (the per-axis Hermite envelopes are basis-agnostic) to avoid a
second copy of the SciPy interpolator vectorisation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from graphic_zero.surfaces_barycentric import (
    build_barycentric_lagrange_basis,
    periodic_cardinal,
    periodic_cardinal_prime,
    periodic_cardinal_second,
)
from graphic_zero_character.character_birkhoff import (
    character_addition_table_with_slopes,
)
from sobolev_distill.teacher import _vectorised_axis_basis


@dataclass(frozen=True)
class CharacterMeshTeacher:
    """Frozen container of character teacher fields on a 2D mesh.

    All array fields are :class:`jax.numpy.ndarray` so they flow into
    ``jit`` / ``vmap`` without an extra device transfer.  Shapes:

    - ``Kx`` / ``Ky`` are the lattice node counts.
    - ``Nx`` / ``Ny`` are the mesh sizes.
    """

    nodes_x: jnp.ndarray            # (Kx,)
    nodes_y: jnp.ndarray            # (Ky,)
    xs: jnp.ndarray                 # (Nx,)
    ys: jnp.ndarray                 # (Ny,)

    T_re: jnp.ndarray               # (Kx, Ky) Re T
    T_im: jnp.ndarray               # (Kx, Ky) Im T
    Dx_ij_re: jnp.ndarray           # (Kx, Ky)
    Dx_ij_im: jnp.ndarray           # (Kx, Ky)
    Dy_ij_re: jnp.ndarray           # (Kx, Ky)
    Dy_ij_im: jnp.ndarray           # (Kx, Ky)

    V_re: jnp.ndarray               # (Nx, Ny)
    V_im: jnp.ndarray               # (Nx, Ny)
    GX_re: jnp.ndarray              # (Nx, Ny)
    GX_im: jnp.ndarray              # (Nx, Ny)
    GY_re: jnp.ndarray              # (Nx, Ny)
    GY_im: jnp.ndarray              # (Nx, Ny)
    Hxx_re: jnp.ndarray             # (Nx, Ny)
    Hxx_im: jnp.ndarray             # (Nx, Ny)
    Hxy_re: jnp.ndarray             # (Nx, Ny)
    Hxy_im: jnp.ndarray             # (Nx, Ny)
    Hyy_re: jnp.ndarray             # (Nx, Ny)
    Hyy_im: jnp.ndarray             # (Nx, Ny)

    V_M: jnp.ndarray                # (Nx, Ny) modulus energy
    Z_W: jnp.ndarray                # (Nx, Ny) bowl regulariser

    is_node: jnp.ndarray            # (Nx, Ny) bool: mesh point coincides with a node
    is_pd: jnp.ndarray              # (Nx, Ny) bool: per-node modulus-energy Hessian PD
    node_min_eig: jnp.ndarray       # (Kx, Ky) min eigvalsh of per-node H_E

    lam: float
    modulus: int


def _node_indices_on_mesh(
    eval_points: np.ndarray,
    nodes: np.ndarray,
    atol: float = 1e-9,
) -> np.ndarray:
    """Mirror of :func:`sobolev_distill.teacher._node_indices_on_mesh`.

    Returns shape ``(N,)`` int array: index in ``nodes`` matching each mesh
    point, or ``-1`` if no node within ``atol``.
    """
    eval_arr = np.asarray(eval_points, dtype=np.float64)
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    diff = np.abs(eval_arr[:, None] - nodes_arr[None, :])
    nearest = np.argmin(diff, axis=1)
    nearest_dist = diff[np.arange(eval_arr.size), nearest]
    out = np.where(nearest_dist <= atol, nearest, -1)
    return out.astype(np.int64)


def _einsum_2d_slope_complex(
    AX: np.ndarray,
    AY: np.ndarray,
    BX: np.ndarray,
    BY: np.ndarray,
    T: np.ndarray,
    Dx_ij: np.ndarray,
    Dy_ij: np.ndarray,
) -> np.ndarray:
    """Helper: ``sum AX[a,i] AY[b,j] T[i,j] + ...`` with 2D slope tensors.

    Each input ``AX`` / ``AY`` is a real envelope tensor; ``T`` / ``Dx_ij``
    / ``Dy_ij`` are complex.  The result is complex of shape
    ``(AX.shape[0], AY.shape[0])``.
    """
    return (
        np.einsum("ai,bj,ij->ab", AX, AY, T)
        + np.einsum("ai,bj,ij->ab", BX, AY, Dx_ij)
        + np.einsum("ai,bj,ij->ab", AX, BY, Dy_ij)
    )


def build_character_teacher_mesh(
    nodes_x: Sequence[float],
    nodes_y: Sequence[float],
    p: int,
    *,
    Lx: Sequence[Any] | None = None,
    Lpx: Sequence[Callable[..., Any]] | None = None,
    Ly: Sequence[Any] | None = None,
    Lpy: Sequence[Callable[..., Any]] | None = None,
    xs: np.ndarray | None = None,
    ys: np.ndarray | None = None,
    mesh_n: int = 256,
    lam: float = 1.0,
) -> CharacterMeshTeacher:
    """Build a :class:`CharacterMeshTeacher` for the character target.

    ``nodes_x`` / ``nodes_y`` are real lattice coordinates (typically the
    integer Welch lattice ``{0, ..., p-1}``).  The character addition
    table ``T_ij = zeta**((nodes_x[i] + nodes_y[j]) % p)`` is built via
    :func:`character_addition_table_with_slopes`.

    The mesh inserts every lattice node so ``is_node`` / ``is_pd`` labels
    are exact.  The per-node modulus-energy Hessian is computed analytically
    (no finite differences) and its eigvalsh PD test gives ``is_pd``.
    """
    nodes_x_arr = np.asarray(nodes_x, dtype=np.float64)
    nodes_y_arr = np.asarray(nodes_y, dtype=np.float64)

    T, Dx_ij, Dy_ij = character_addition_table_with_slopes(nodes_x_arr, nodes_y_arr, p)

    if Lx is None or Lpx is None:
        Lx, Lpx = build_barycentric_lagrange_basis(nodes_x_arr.tolist())
    if Ly is None or Lpy is None:
        Ly, Lpy = build_barycentric_lagrange_basis(nodes_y_arr.tolist())

    if xs is None:
        xs = np.linspace(float(nodes_x_arr.min()), float(nodes_x_arr.max()), mesh_n)
    if ys is None:
        ys = np.linspace(float(nodes_y_arr.min()), float(nodes_y_arr.max()), mesh_n)
    xs = np.unique(np.concatenate([np.asarray(xs, dtype=np.float64), nodes_x_arr]))
    ys = np.unique(np.concatenate([np.asarray(ys, dtype=np.float64), nodes_y_arr]))

    bx = _vectorised_axis_basis(xs, nodes_x_arr, Lx, Lpx)
    by = _vectorised_axis_basis(ys, nodes_y_arr, Ly, Lpy)
    AX_mesh, AY_mesh = bx["alpha"], by["alpha"]
    BX_mesh, BY_mesh = bx["beta"], by["beta"]
    AXP_mesh, AYP_mesh = bx["alpha_p"], by["alpha_p"]
    BXP_mesh, BYP_mesh = bx["beta_p"], by["beta_p"]
    A2X_mesh, A2Y_mesh = bx["alpha_pp"], by["alpha_pp"]
    B2X_mesh, B2Y_mesh = bx["beta_pp"], by["beta_pp"]
    LX_mesh, LY_mesh = bx["L"], by["L"]

    # ---- value, gradient, Hessian on the mesh (all complex) -----------
    V = _einsum_2d_slope_complex(AX_mesh, AY_mesh, BX_mesh, BY_mesh, T, Dx_ij, Dy_ij)
    GX = _einsum_2d_slope_complex(AXP_mesh, AY_mesh, BXP_mesh, BY_mesh, T, Dx_ij, Dy_ij)
    GY = _einsum_2d_slope_complex(AX_mesh, AYP_mesh, BX_mesh, BYP_mesh, T, Dx_ij, Dy_ij)
    Hxx = _einsum_2d_slope_complex(A2X_mesh, AY_mesh, B2X_mesh, BY_mesh, T, Dx_ij, Dy_ij)
    Hyy = _einsum_2d_slope_complex(AX_mesh, A2Y_mesh, BX_mesh, B2Y_mesh, T, Dx_ij, Dy_ij)
    Hxy = _einsum_2d_slope_complex(AXP_mesh, AYP_mesh, BXP_mesh, BYP_mesh, T, Dx_ij, Dy_ij)

    # ---- Z_W bowl (real) ------------------------------------------------
    LX_sq = LX_mesh * LX_mesh
    LY_sq = LY_mesh * LY_mesh
    dx2 = (xs[:, None] - nodes_x_arr[None, :]) ** 2
    dy2 = (ys[:, None] - nodes_y_arr[None, :]) ** 2
    Z_W = (
        np.einsum("xi,yj,xi->xy", LX_sq, LY_sq, dx2)
        + np.einsum("xi,yj,yj->xy", LX_sq, LY_sq, dy2)
    )

    # ---- modulus energy V_M = (|f_H|^2 - 1)^2 + lam * Z_W ---------------
    abs2 = V.real * V.real + V.imag * V.imag
    V_M = (abs2 - 1.0) ** 2 + float(lam) * Z_W

    # ---- per-node modulus-energy Hessian via Gauss-Newton + bowl --------
    bx_node = _vectorised_axis_basis(nodes_x_arr, nodes_x_arr, Lx, Lpx)
    by_node = _vectorised_axis_basis(nodes_y_arr, nodes_y_arr, Ly, Lpy)
    AX_node = bx_node["alpha"]
    AY_node = by_node["alpha"]
    BX_node = bx_node["beta"]
    BY_node = by_node["beta"]
    AXP_node = bx_node["alpha_p"]
    AYP_node = by_node["alpha_p"]
    BXP_node = bx_node["beta_p"]
    BYP_node = by_node["beta_p"]

    GX_node = _einsum_2d_slope_complex(
        AXP_node, AY_node, BXP_node, BY_node, T, Dx_ij, Dy_ij
    )
    GY_node = _einsum_2d_slope_complex(
        AX_node, AYP_node, BX_node, BYP_node, T, Dx_ij, Dy_ij
    )

    # Analytic bowl Hessian diagonal at nodes:
    #   wxx[a] = 2 + sum_{i != a} 2 (x_a - x_i)^2 ell'_i(x_a)^2
    Lp_node_x = np.zeros((nodes_x_arr.size, nodes_x_arr.size), dtype=np.float64)
    for i in range(nodes_x_arr.size):
        Lp_node_x[:, i] = np.asarray(Lpx[i](nodes_x_arr), dtype=np.float64)
    Lp_node_y = np.zeros((nodes_y_arr.size, nodes_y_arr.size), dtype=np.float64)
    for j in range(nodes_y_arr.size):
        Lp_node_y[:, j] = np.asarray(Lpy[j](nodes_y_arr), dtype=np.float64)
    diff_x = nodes_x_arr[:, None] - nodes_x_arr[None, :]
    diff_y = nodes_y_arr[:, None] - nodes_y_arr[None, :]
    wxx = 2.0 + (2.0 * (diff_x ** 2) * (Lp_node_x ** 2)).sum(axis=1)  # (Kx,)
    wyy = 2.0 + (2.0 * (diff_y ** 2) * (Lp_node_y ** 2)).sum(axis=1)  # (Ky,)

    # H_E = 2 * Re[conj(g) g^T] + lam * diag(wxx, wyy) at each node.
    # Note: V_M = (|f_H|^2 - 1)^2 has H = 4 * Re[conj(g) g^T] at f_H on the
    # unit circle (|f_H|=1).  We use coefficient 4 (not 2) because the
    # quartic outer term is |f_H|^2 - 1, and its Hessian via chain rule
    # gives factor 2 * 2 = 4 (no other terms survive at the node).
    gx_re = GX_node.real
    gx_im = GX_node.imag
    gy_re = GY_node.real
    gy_im = GY_node.imag
    H00 = 4.0 * (gx_re * gx_re + gx_im * gx_im) + float(lam) * wxx[:, None]
    H11 = 4.0 * (gy_re * gy_re + gy_im * gy_im) + float(lam) * wyy[None, :]
    H01 = 4.0 * (gx_re * gy_re + gx_im * gy_im)
    h_stack = np.stack(
        [
            np.stack([H00, H01], axis=-1),
            np.stack([H01, H11], axis=-1),
        ],
        axis=-2,
    )
    eigs = np.linalg.eigvalsh(h_stack)  # (Kx, Ky, 2)
    node_min_eig = eigs[..., 0]

    # ---- mesh-level node mask & PD label -------------------------------
    x_node_idx = _node_indices_on_mesh(xs, nodes_x_arr)
    y_node_idx = _node_indices_on_mesh(ys, nodes_y_arr)
    is_node_x = x_node_idx >= 0
    is_node_y = y_node_idx >= 0
    is_node = is_node_x[:, None] & is_node_y[None, :]

    is_pd = np.zeros_like(is_node, dtype=bool)
    pd_grid = node_min_eig > 0.0
    x_at_nodes = np.where(is_node_x)[0]
    y_at_nodes = np.where(is_node_y)[0]
    if x_at_nodes.size and y_at_nodes.size:
        for mi in x_at_nodes:
            ni = int(x_node_idx[mi])
            for mj in y_at_nodes:
                nj = int(y_node_idx[mj])
                is_pd[mi, mj] = bool(pd_grid[ni, nj])

    return CharacterMeshTeacher(
        nodes_x=jnp.asarray(nodes_x_arr),
        nodes_y=jnp.asarray(nodes_y_arr),
        xs=jnp.asarray(xs),
        ys=jnp.asarray(ys),
        T_re=jnp.asarray(T.real),
        T_im=jnp.asarray(T.imag),
        Dx_ij_re=jnp.asarray(Dx_ij.real),
        Dx_ij_im=jnp.asarray(Dx_ij.imag),
        Dy_ij_re=jnp.asarray(Dy_ij.real),
        Dy_ij_im=jnp.asarray(Dy_ij.imag),
        V_re=jnp.asarray(V.real),
        V_im=jnp.asarray(V.imag),
        GX_re=jnp.asarray(GX.real),
        GX_im=jnp.asarray(GX.imag),
        GY_re=jnp.asarray(GY.real),
        GY_im=jnp.asarray(GY.imag),
        Hxx_re=jnp.asarray(Hxx.real),
        Hxx_im=jnp.asarray(Hxx.imag),
        Hxy_re=jnp.asarray(Hxy.real),
        Hxy_im=jnp.asarray(Hxy.imag),
        Hyy_re=jnp.asarray(Hyy.real),
        Hyy_im=jnp.asarray(Hyy.imag),
        V_M=jnp.asarray(V_M),
        Z_W=jnp.asarray(Z_W),
        is_node=jnp.asarray(is_node),
        is_pd=jnp.asarray(is_pd),
        node_min_eig=jnp.asarray(node_min_eig),
        lam=float(lam),
        modulus=int(p),
    )


# Reduce noisy F841 if unused
_ = math


def _vectorised_periodic_axis_basis(
    eval_points: np.ndarray,
    nodes: np.ndarray,
    K: int,
) -> dict[str, np.ndarray]:
    """Periodic (Dirichlet) analogue of :func:`_vectorised_axis_basis`.

    Returns shape ``(N, K)`` tensors keyed
    ``{"L", "alpha", "beta", "alpha_p", "beta_p", "alpha_pp", "beta_pp"}``
    where ``L = alpha = alpha_i^per`` and the ``beta*`` channels are zero
    (the periodic Lagrange interpolant of a single-Fourier-mode signal is
    exact, so no Birkhoff slope envelope is needed).  Plugged into the same
    ``_einsum_2d_slope_complex`` pipeline as the polynomial teacher with
    ``Dx_ij = Dy_ij = zeros`` so the B-channel einsums vanish.
    """
    eval_arr = np.asarray(eval_points, dtype=np.float64)
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    n = eval_arr.shape[0]
    k = nodes_arr.shape[0]
    alpha = np.zeros((n, k), dtype=np.float64)
    alpha_p = np.zeros((n, k), dtype=np.float64)
    alpha_pp = np.zeros((n, k), dtype=np.float64)
    for i in range(k):
        alpha[:, i] = np.asarray(periodic_cardinal(i, nodes_arr, eval_arr, int(K)), dtype=np.float64)
        alpha_p[:, i] = np.asarray(periodic_cardinal_prime(i, nodes_arr, eval_arr, int(K)), dtype=np.float64)
        alpha_pp[:, i] = np.asarray(periodic_cardinal_second(i, nodes_arr, eval_arr, int(K)), dtype=np.float64)
    zeros_nk = np.zeros((n, k), dtype=np.float64)
    return {
        "L": alpha,
        "alpha": alpha,
        "beta": zeros_nk,
        "alpha_p": alpha_p,
        "beta_p": zeros_nk,
        "alpha_pp": alpha_pp,
        "beta_pp": zeros_nk,
    }


def _nearest_node_distance(coords: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """``min_i |coords[a] - nodes[i]|`` per coord (no periodic wrap; nodes assumed dense)."""
    diff = np.abs(coords[:, None] - nodes[None, :])
    return diff.min(axis=1)


def build_character_teacher_mesh_periodic(
    nodes_x: Sequence[float],
    nodes_y: Sequence[float],
    p: int,
    *,
    xs: np.ndarray | None = None,
    ys: np.ndarray | None = None,
    mesh_n: int = 256,
    lam: float = 1.0,
) -> CharacterMeshTeacher:
    """Build a periodic-cardinal :class:`CharacterMeshTeacher` for the character target.

    Same return type and semantics as :func:`build_character_teacher_mesh`,
    but the per-axis cardinal envelopes are the trig-polynomial Dirichlet
    kernels from :mod:`graphic_zero.surfaces_barycentric`.  Because
    ``zeta**((x + y) mod p)`` is a single-Fourier-mode signal in the basis'
    band, the periodic interpolant matches the analytic value AND its
    gradients exactly on the entire mesh; the off-lattice "wrap" is smooth
    and ``|f_H| = 1`` everywhere (not just at lattice nodes).

    The bowl regulariser is the torus-natural ``Z_W = sin^2(pi (x - x_nearest) / K)
    + sin^2(pi (y - y_nearest) / K)`` which vanishes at every lattice node.
    The per-node bowl Hessian diagonal is the constant ``2 (pi / K)^2``.

    ``nodes_x`` / ``nodes_y`` are assumed integer Welch coordinates
    ``{0, 1, ..., p - 1}``; ``K_x = nodes_x.size`` and ``K_y = nodes_y.size``
    set the trig-polynomial band on each axis.
    """
    nodes_x_arr = np.asarray(nodes_x, dtype=np.float64)
    nodes_y_arr = np.asarray(nodes_y, dtype=np.float64)
    K_x = int(nodes_x_arr.size)
    K_y = int(nodes_y_arr.size)

    T, _, _ = character_addition_table_with_slopes(nodes_x_arr, nodes_y_arr, p)
    Dx_ij = np.zeros_like(T)
    Dy_ij = np.zeros_like(T)

    if xs is None:
        xs = np.linspace(float(nodes_x_arr.min()), float(nodes_x_arr.max()), mesh_n)
    if ys is None:
        ys = np.linspace(float(nodes_y_arr.min()), float(nodes_y_arr.max()), mesh_n)
    xs = np.unique(np.concatenate([np.asarray(xs, dtype=np.float64), nodes_x_arr]))
    ys = np.unique(np.concatenate([np.asarray(ys, dtype=np.float64), nodes_y_arr]))

    bx = _vectorised_periodic_axis_basis(xs, nodes_x_arr, K_x)
    by = _vectorised_periodic_axis_basis(ys, nodes_y_arr, K_y)
    AX_mesh, AY_mesh = bx["alpha"], by["alpha"]
    BX_mesh, BY_mesh = bx["beta"], by["beta"]
    AXP_mesh, AYP_mesh = bx["alpha_p"], by["alpha_p"]
    BXP_mesh, BYP_mesh = bx["beta_p"], by["beta_p"]
    A2X_mesh, A2Y_mesh = bx["alpha_pp"], by["alpha_pp"]
    B2X_mesh, B2Y_mesh = bx["beta_pp"], by["beta_pp"]

    V = _einsum_2d_slope_complex(AX_mesh, AY_mesh, BX_mesh, BY_mesh, T, Dx_ij, Dy_ij)
    GX = _einsum_2d_slope_complex(AXP_mesh, AY_mesh, BXP_mesh, BY_mesh, T, Dx_ij, Dy_ij)
    GY = _einsum_2d_slope_complex(AX_mesh, AYP_mesh, BX_mesh, BYP_mesh, T, Dx_ij, Dy_ij)
    Hxx = _einsum_2d_slope_complex(A2X_mesh, AY_mesh, B2X_mesh, BY_mesh, T, Dx_ij, Dy_ij)
    Hyy = _einsum_2d_slope_complex(AX_mesh, A2Y_mesh, BX_mesh, B2Y_mesh, T, Dx_ij, Dy_ij)
    Hxy = _einsum_2d_slope_complex(AXP_mesh, AYP_mesh, BXP_mesh, BYP_mesh, T, Dx_ij, Dy_ij)

    # Torus bowl: vanishes at every lattice node, periodic per axis.
    dist_x = _nearest_node_distance(xs, nodes_x_arr)
    dist_y = _nearest_node_distance(ys, nodes_y_arr)
    bowl_x = np.sin(np.pi * dist_x / float(K_x)) ** 2
    bowl_y = np.sin(np.pi * dist_y / float(K_y)) ** 2
    Z_W = bowl_x[:, None] + bowl_y[None, :]

    abs2 = V.real * V.real + V.imag * V.imag
    V_M = (abs2 - 1.0) ** 2 + float(lam) * Z_W

    bx_node = _vectorised_periodic_axis_basis(nodes_x_arr, nodes_x_arr, K_x)
    by_node = _vectorised_periodic_axis_basis(nodes_y_arr, nodes_y_arr, K_y)
    AX_node = bx_node["alpha"]
    AY_node = by_node["alpha"]
    BX_node = bx_node["beta"]
    BY_node = by_node["beta"]
    AXP_node = bx_node["alpha_p"]
    AYP_node = by_node["alpha_p"]
    BXP_node = bx_node["beta_p"]
    BYP_node = by_node["beta_p"]

    GX_node = _einsum_2d_slope_complex(
        AXP_node, AY_node, BXP_node, BY_node, T, Dx_ij, Dy_ij
    )
    GY_node = _einsum_2d_slope_complex(
        AX_node, AYP_node, BX_node, BYP_node, T, Dx_ij, Dy_ij
    )

    # Constant per-node bowl Hessian diagonal: d^2/dx^2 sin^2(pi (x - v_a) / K) at x=v_a.
    bowl_diag_x = 2.0 * (np.pi / float(K_x)) ** 2
    bowl_diag_y = 2.0 * (np.pi / float(K_y)) ** 2
    wxx = np.full(K_x, bowl_diag_x, dtype=np.float64)
    wyy = np.full(K_y, bowl_diag_y, dtype=np.float64)

    gx_re = GX_node.real
    gx_im = GX_node.imag
    gy_re = GY_node.real
    gy_im = GY_node.imag
    H00 = 4.0 * (gx_re * gx_re + gx_im * gx_im) + float(lam) * wxx[:, None]
    H11 = 4.0 * (gy_re * gy_re + gy_im * gy_im) + float(lam) * wyy[None, :]
    H01 = 4.0 * (gx_re * gy_re + gx_im * gy_im)
    h_stack = np.stack(
        [
            np.stack([H00, H01], axis=-1),
            np.stack([H01, H11], axis=-1),
        ],
        axis=-2,
    )
    eigs = np.linalg.eigvalsh(h_stack)
    node_min_eig = eigs[..., 0]

    x_node_idx = _node_indices_on_mesh(xs, nodes_x_arr)
    y_node_idx = _node_indices_on_mesh(ys, nodes_y_arr)
    is_node_x = x_node_idx >= 0
    is_node_y = y_node_idx >= 0
    is_node = is_node_x[:, None] & is_node_y[None, :]

    is_pd = np.zeros_like(is_node, dtype=bool)
    pd_grid = node_min_eig > 0.0
    x_at_nodes = np.where(is_node_x)[0]
    y_at_nodes = np.where(is_node_y)[0]
    if x_at_nodes.size and y_at_nodes.size:
        for mi in x_at_nodes:
            ni = int(x_node_idx[mi])
            for mj in y_at_nodes:
                nj = int(y_node_idx[mj])
                is_pd[mi, mj] = bool(pd_grid[ni, nj])

    # Use analytic complex slope tables for the on-lattice ``Dx_ij_*`` /
    # ``Dy_ij_*`` fields exposed on the dataclass: they are the analytic
    # gradient at lattice nodes (``(2*pi*1j / p) * T_ij``), regardless of
    # whether they were used in the einsums above.
    _, Dx_ij_analytic, Dy_ij_analytic = character_addition_table_with_slopes(
        nodes_x_arr, nodes_y_arr, p
    )

    return CharacterMeshTeacher(
        nodes_x=jnp.asarray(nodes_x_arr),
        nodes_y=jnp.asarray(nodes_y_arr),
        xs=jnp.asarray(xs),
        ys=jnp.asarray(ys),
        T_re=jnp.asarray(T.real),
        T_im=jnp.asarray(T.imag),
        Dx_ij_re=jnp.asarray(Dx_ij_analytic.real),
        Dx_ij_im=jnp.asarray(Dx_ij_analytic.imag),
        Dy_ij_re=jnp.asarray(Dy_ij_analytic.real),
        Dy_ij_im=jnp.asarray(Dy_ij_analytic.imag),
        V_re=jnp.asarray(V.real),
        V_im=jnp.asarray(V.imag),
        GX_re=jnp.asarray(GX.real),
        GX_im=jnp.asarray(GX.imag),
        GY_re=jnp.asarray(GY.real),
        GY_im=jnp.asarray(GY.imag),
        Hxx_re=jnp.asarray(Hxx.real),
        Hxx_im=jnp.asarray(Hxx.imag),
        Hxy_re=jnp.asarray(Hxy.real),
        Hxy_im=jnp.asarray(Hxy.imag),
        Hyy_re=jnp.asarray(Hyy.real),
        Hyy_im=jnp.asarray(Hyy.imag),
        V_M=jnp.asarray(V_M),
        Z_W=jnp.asarray(Z_W),
        is_node=jnp.asarray(is_node),
        is_pd=jnp.asarray(is_pd),
        node_min_eig=jnp.asarray(node_min_eig),
        lam=float(lam),
        modulus=int(p),
    )


__all__ = [
    "CharacterMeshTeacher",
    "build_character_teacher_mesh",
    "build_character_teacher_mesh_periodic",
]
