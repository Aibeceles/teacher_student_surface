"""Materialize the Birkhoff-Hermite teacher fields on a dense JAX-friendly mesh.

The CuPy/SciPy stack in :mod:`graphic_zero.hermite_barycentric_gpu` already gives
us the value field ``f_H`` and the regularised energy ``f_M`` on a 2D grid.  This
module extends the same recipe with **mesh-evaluated** Hermite tensors for the
gradient (``alpha'``, ``beta'``) and Hessian (``alpha''``, ``beta''``) directions,
runs the einsums in NumPy, and returns the result as ``jnp.ndarray``s for the
downstream Sobolev student.

Design notes:

- Teacher is materialised **once** on a chosen mesh; the student trains against
  cached arrays.  This keeps the inner loop pure JAX and ``jit``-friendly.
- The per-basis SciPy ``BarycentricInterpolator`` calls are vectorised over the
  whole evaluation grid (one ``__call__`` per basis function, not one per
  point), so building a 256x256 mesh with K=64 nodes is sub-second.
- The PD certificate from :func:`f_m_diagnostics_on_device` is reproduced here
  for **every** lattice node, not only the worst one, so the energy head can be
  trained against a per-node ``is_pd`` label.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from graphic_zero.surfaces_barycentric import build_barycentric_lagrange_basis


@dataclass(frozen=True)
class MeshTeacher:
    """Frozen container of teacher fields evaluated on a 2D mesh.

    All array fields are :class:`jax.numpy.ndarray` so they flow into ``jit`` /
    ``vmap`` without an extra device transfer.  Shapes are documented per
    attribute; ``Nx`` / ``Ny`` are the mesh sizes and ``Kx`` / ``Ky`` are the
    interpolation node counts.
    """

    nodes_x: jnp.ndarray  # (Kx,) interpolation nodes on x-axis
    nodes_y: jnp.ndarray  # (Ky,) interpolation nodes on y-axis
    xs: jnp.ndarray       # (Nx,) mesh evaluation coordinates on x-axis
    ys: jnp.ndarray       # (Ny,) mesh evaluation coordinates on y-axis
    T: jnp.ndarray        # (Kx, Ky) value table at nodes (addition table)
    Dx: jnp.ndarray       # (Kx,) per-node x-derivative target (Birkhoff data)
    Dy: jnp.ndarray       # (Ky,) per-node y-derivative target
    lam: float            # bowl regularisation coefficient used for f_M

    V: jnp.ndarray        # (Nx, Ny) f_H values on the mesh
    GX: jnp.ndarray       # (Nx, Ny) d/dx f_H
    GY: jnp.ndarray       # (Nx, Ny) d/dy f_H
    Hxx: jnp.ndarray      # (Nx, Ny) d^2/dx^2 f_H
    Hxy: jnp.ndarray      # (Nx, Ny) d^2/dx dy f_H
    Hyy: jnp.ndarray      # (Nx, Ny) d^2/dy^2 f_H
    V_M: jnp.ndarray      # (Nx, Ny) f_M = f_H0 + lam * z_w on the mesh
    Z_W: jnp.ndarray      # (Nx, Ny) bowl term z_w used inside V_M

    is_node: jnp.ndarray   # (Nx, Ny) bool: mesh point coincides with a lattice node
    is_pd: jnp.ndarray     # (Nx, Ny) bool: is_node and per-node f_M Hessian PD
    node_min_eig: jnp.ndarray  # (Kx, Ky) min eigvalsh of f_M Hessian per lattice node


def _vectorised_axis_basis(
    eval_points: np.ndarray,
    nodes: Sequence[float],
    L_list: Sequence[Any],
    Lp_list: Sequence[Callable[..., Any]],
) -> dict[str, np.ndarray]:
    """Compute Hermite alpha/beta tensors and their first/second derivatives.

    All outputs have shape ``(N, K)`` where ``N = len(eval_points)`` and
    ``K = len(nodes)``.  The math mirrors
    :func:`graphic_zero.surfaces_barycentric.hermite_alpha_beta`,
    :func:`hermite_alpha_beta_prime`, :func:`hermite_alpha_beta_second`, but
    vectorised over ``eval_points`` so the per-basis SciPy interpolator is
    evaluated only once per ``i``.
    """
    eval_arr = np.asarray(eval_points, dtype=np.float64)
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    n = eval_arr.shape[0]
    k = nodes_arr.shape[0]

    L = np.zeros((n, k), dtype=np.float64)   # ell_i(x_a)
    Lp = np.zeros((n, k), dtype=np.float64)  # ell'_i(x_a)
    L2 = np.zeros((n, k), dtype=np.float64)  # ell''_i(x_a)
    c = np.zeros(k, dtype=np.float64)        # ell'_i(node_i)
    for i in range(k):
        interp = L_list[i]
        L[:, i] = np.asarray(interp(eval_arr), dtype=np.float64)
        Lp[:, i] = np.asarray(Lp_list[i](eval_arr), dtype=np.float64)
        L2[:, i] = np.asarray(interp.derivative(eval_arr, der=2), dtype=np.float64)
        c[i] = float(np.asarray(Lp_list[i](float(nodes_arr[i]))).reshape(-1)[0])

    u = eval_arr[:, None] - nodes_arr[None, :]            # (N, K)  x - v_i
    one_minus = 1.0 - 2.0 * u * c[None, :]                # (N, K)  w(x)
    L_sq = L * L
    two_LLp = 2.0 * L * Lp
    LLcurv = Lp * Lp + L * L2

    alpha = one_minus * L_sq
    beta = u * L_sq
    alpha_p = (-2.0 * c[None, :]) * L_sq + one_minus * two_LLp
    beta_p = L_sq + u * two_LLp
    # alpha'' = -8 c L L' + 2 (1 - 2 c u) (L'^2 + L L'')
    alpha_pp = -8.0 * c[None, :] * L * Lp + 2.0 * one_minus * LLcurv
    # beta''  =  4 L L' + 2 u (L'^2 + L L'')
    beta_pp = 4.0 * L * Lp + 2.0 * u * LLcurv

    return {
        "L": L,
        "alpha": alpha,
        "beta": beta,
        "alpha_p": alpha_p,
        "beta_p": beta_p,
        "alpha_pp": alpha_pp,
        "beta_pp": beta_pp,
    }


def _node_indices_on_mesh(
    eval_points: np.ndarray,
    nodes: np.ndarray,
    atol: float = 1e-9,
) -> np.ndarray:
    """Return shape ``(N,)`` int array: index in ``nodes`` matching each mesh
    point, or ``-1`` if no node within ``atol``.  Used to build ``is_node`` and
    to look up per-node certificates."""
    eval_arr = np.asarray(eval_points, dtype=np.float64)
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    diff = np.abs(eval_arr[:, None] - nodes_arr[None, :])  # (N, K)
    nearest = np.argmin(diff, axis=1)
    nearest_dist = diff[np.arange(eval_arr.size), nearest]
    out = np.where(nearest_dist <= atol, nearest, -1)
    return out.astype(np.int64)


def build_teacher_mesh(
    nodes_x: Sequence[float],
    nodes_y: Sequence[float],
    T: np.ndarray,
    Dx: Sequence[float],
    Dy: Sequence[float],
    *,
    Lx: Sequence[Any] | None = None,
    Lpx: Sequence[Callable[..., Any]] | None = None,
    Ly: Sequence[Any] | None = None,
    Lpy: Sequence[Callable[..., Any]] | None = None,
    xs: np.ndarray | None = None,
    ys: np.ndarray | None = None,
    mesh_n: int = 256,
    lam: float = 1.0,
) -> MeshTeacher:
    """Build a :class:`MeshTeacher` on a chosen evaluation grid.

    If ``Lx``/``Lpx``/``Ly``/``Lpy`` are not supplied, fresh barycentric
    Lagrange bases are constructed via
    :func:`graphic_zero.surfaces_barycentric.build_barycentric_lagrange_basis`.

    If ``xs``/``ys`` are not supplied, a uniform ``mesh_n``-by-``mesh_n`` grid
    spanning the node range is used; the union of node coordinates is **inserted
    into the grid** so every lattice node has an exact mesh sample (this is what
    enables ``is_node`` / ``is_pd`` labels and lattice-only diagnostics).
    """
    nodes_x_arr = np.asarray(nodes_x, dtype=np.float64)
    nodes_y_arr = np.asarray(nodes_y, dtype=np.float64)
    T_arr = np.asarray(T, dtype=np.float64)
    Dx_arr = np.asarray(Dx, dtype=np.float64)
    Dy_arr = np.asarray(Dy, dtype=np.float64)

    if T_arr.shape != (nodes_x_arr.size, nodes_y_arr.size):
        raise ValueError(
            f"T shape {T_arr.shape} incompatible with node sizes "
            f"({nodes_x_arr.size}, {nodes_y_arr.size})"
        )
    if Dx_arr.shape != (nodes_x_arr.size,):
        raise ValueError(f"Dx shape {Dx_arr.shape} != ({nodes_x_arr.size},)")
    if Dy_arr.shape != (nodes_y_arr.size,):
        raise ValueError(f"Dy shape {Dy_arr.shape} != ({nodes_y_arr.size},)")

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

    # ---- f_H value ------------------------------------------------------
    V = (
        np.einsum("xi,yj,ij->xy", AX_mesh, AY_mesh, T_arr)
        + np.einsum("xi,yj,i->xy", BX_mesh, AY_mesh, Dx_arr)
        + np.einsum("xi,yj,j->xy", AX_mesh, BY_mesh, Dy_arr)
    )
    # ---- f_H gradient ---------------------------------------------------
    GX = (
        np.einsum("xi,yj,ij->xy", AXP_mesh, AY_mesh, T_arr)
        + np.einsum("xi,yj,i->xy", BXP_mesh, AY_mesh, Dx_arr)
        + np.einsum("xi,yj,j->xy", AXP_mesh, BY_mesh, Dy_arr)
    )
    GY = (
        np.einsum("xi,yj,ij->xy", AX_mesh, AYP_mesh, T_arr)
        + np.einsum("xi,yj,i->xy", BX_mesh, AYP_mesh, Dx_arr)
        + np.einsum("xi,yj,j->xy", AX_mesh, BYP_mesh, Dy_arr)
    )
    # ---- f_H Hessian ----------------------------------------------------
    Hxx = (
        np.einsum("xi,yj,ij->xy", A2X_mesh, AY_mesh, T_arr)
        + np.einsum("xi,yj,i->xy", B2X_mesh, AY_mesh, Dx_arr)
        + np.einsum("xi,yj,j->xy", A2X_mesh, BY_mesh, Dy_arr)
    )
    Hyy = (
        np.einsum("xi,yj,ij->xy", AX_mesh, A2Y_mesh, T_arr)
        + np.einsum("xi,yj,i->xy", BX_mesh, A2Y_mesh, Dx_arr)
        + np.einsum("xi,yj,j->xy", AX_mesh, B2Y_mesh, Dy_arr)
    )
    Hxy = (
        np.einsum("xi,yj,ij->xy", AXP_mesh, AYP_mesh, T_arr)
        + np.einsum("xi,yj,i->xy", BXP_mesh, AYP_mesh, Dx_arr)
        + np.einsum("xi,yj,j->xy", AXP_mesh, BYP_mesh, Dy_arr)
    )

    # ---- f_M = f_H0 + lam * z_w ----------------------------------------
    V_H0 = np.einsum("xi,yj,ij->xy", AX_mesh, AY_mesh, T_arr)
    LX_sq = LX_mesh * LX_mesh
    LY_sq = LY_mesh * LY_mesh
    dx2 = (xs[:, None] - nodes_x_arr[None, :]) ** 2  # (Nx, Kx)
    dy2 = (ys[:, None] - nodes_y_arr[None, :]) ** 2  # (Ny, Ky)
    Z_W = (
        np.einsum("xi,yj,xi->xy", LX_sq, LY_sq, dx2)
        + np.einsum("xi,yj,yj->xy", LX_sq, LY_sq, dy2)
    )
    V_M = V_H0 + float(lam) * Z_W

    # ---- per-node certificate -----------------------------------------
    # Node-evaluated tensors are exactly the rows of the mesh tensors at the
    # lattice indices.
    x_node_idx = _node_indices_on_mesh(xs, nodes_x_arr)
    y_node_idx = _node_indices_on_mesh(ys, nodes_y_arr)
    if np.any(x_node_idx[x_node_idx >= 0] < 0):  # pragma: no cover - defensive
        raise RuntimeError("Failed to align lattice nodes onto x mesh")

    x_at_nodes = np.where(x_node_idx >= 0)[0]
    y_at_nodes = np.where(y_node_idx >= 0)[0]
    if x_at_nodes.size != nodes_x_arr.size or y_at_nodes.size != nodes_y_arr.size:
        raise RuntimeError(
            "After mesh-node merge, expected every lattice node to appear; "
            f"got {x_at_nodes.size}/{nodes_x_arr.size} on x, "
            f"{y_at_nodes.size}/{nodes_y_arr.size} on y."
        )

    # Order x_at_nodes / y_at_nodes by their node index so AX_node[i, :] is the
    # alpha row at lattice node i (matches precompute_*_at_nodes layout).
    x_node_order = np.argsort(x_node_idx[x_at_nodes])
    y_node_order = np.argsort(y_node_idx[y_at_nodes])
    x_at_nodes = x_at_nodes[x_node_order]
    y_at_nodes = y_at_nodes[y_node_order]

    AX_node = AX_mesh[x_at_nodes]
    AY_node = AY_mesh[y_at_nodes]
    AXP_node = AXP_mesh[x_at_nodes]
    AYP_node = AYP_mesh[y_at_nodes]
    A2X_node = A2X_mesh[x_at_nodes]
    A2Y_node = A2Y_mesh[y_at_nodes]

    # Beta tensors at nodes are required for the second-derivative
    # contributions; the vectorised helper computes alpha and beta together.
    bx_node = _vectorised_axis_basis(nodes_x_arr, nodes_x_arr, Lx, Lpx)
    by_node = _vectorised_axis_basis(nodes_y_arr, nodes_y_arr, Ly, Lpy)
    BX_node = bx_node["beta"]
    BY_node = by_node["beta"]
    B2X_node = bx_node["beta_pp"]
    B2Y_node = by_node["beta_pp"]
    BXP_node = bx_node["beta_p"]
    BYP_node = by_node["beta_p"]

    Hxx_node = (
        np.einsum("ai,bj,ij->ab", A2X_node, AY_node, T_arr)
        + np.einsum("ai,bj,i->ab", B2X_node, AY_node, Dx_arr)
        + np.einsum("ai,bj,j->ab", A2X_node, BY_node, Dy_arr)
    )
    Hyy_node = (
        np.einsum("ai,bj,ij->ab", AX_node, A2Y_node, T_arr)
        + np.einsum("ai,bj,i->ab", BX_node, A2Y_node, Dx_arr)
        + np.einsum("ai,bj,j->ab", AX_node, B2Y_node, Dy_arr)
    )
    Hxy_node = (
        np.einsum("ai,bj,ij->ab", AXP_node, AYP_node, T_arr)
        + np.einsum("ai,bj,i->ab", BXP_node, AYP_node, Dx_arr)
        + np.einsum("ai,bj,j->ab", AXP_node, BYP_node, Dy_arr)
    )

    # Bowl Hessian per node: at node a, b the bowl contributes
    #   wxx[a] = 2 + sum_{i != a} 2 (x_a - x_i)^2 ell'_i(x_a)^2
    # We can compute this directly:
    # At node a, ell_i(x_a) = delta_{ia}, so LX_node_sq[a, i] = delta_{ia}.
    # The bowl gradient/Hessian formulas in the analytic notebook use this.
    # Closed form (matches bowl_hess_diagonal_terms in the existing module):
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

    # f_M Hessian per node: H_node + lam * diag(wxx[i], wyy[j])
    H00 = Hxx_node + float(lam) * wxx[:, None]
    H11 = Hyy_node + float(lam) * wyy[None, :]
    H01 = Hxy_node
    # (Kx, Ky, 2, 2) symmetric; eigvalsh per node:
    h_stack = np.stack(
        [
            np.stack([H00, H01], axis=-1),
            np.stack([H01, H11], axis=-1),
        ],
        axis=-2,
    )
    eigs = np.linalg.eigvalsh(h_stack)  # (Kx, Ky, 2)
    node_min_eig = eigs[..., 0]         # (Kx, Ky)

    # ---- mesh-level node mask & PD label -------------------------------
    is_node_x = (x_node_idx >= 0)  # (Nx,)
    is_node_y = (y_node_idx >= 0)  # (Ny,)
    is_node = is_node_x[:, None] & is_node_y[None, :]  # (Nx, Ny)

    is_pd = np.zeros_like(is_node, dtype=bool)
    if x_at_nodes.size and y_at_nodes.size:
        pd_grid = node_min_eig > 0.0  # (Kx, Ky)
        idx_x = x_node_idx[x_at_nodes]
        idx_y = y_node_idx[y_at_nodes]
        for ai, mi in enumerate(x_at_nodes):
            ni = idx_x[ai]
            for bj, mj in enumerate(y_at_nodes):
                nj = idx_y[bj]
                is_pd[mi, mj] = bool(pd_grid[ni, nj])

    return MeshTeacher(
        nodes_x=jnp.asarray(nodes_x_arr),
        nodes_y=jnp.asarray(nodes_y_arr),
        xs=jnp.asarray(xs),
        ys=jnp.asarray(ys),
        T=jnp.asarray(T_arr),
        Dx=jnp.asarray(Dx_arr),
        Dy=jnp.asarray(Dy_arr),
        lam=float(lam),
        V=jnp.asarray(V),
        GX=jnp.asarray(GX),
        GY=jnp.asarray(GY),
        Hxx=jnp.asarray(Hxx),
        Hxy=jnp.asarray(Hxy),
        Hyy=jnp.asarray(Hyy),
        V_M=jnp.asarray(V_M),
        Z_W=jnp.asarray(Z_W),
        is_node=jnp.asarray(is_node),
        is_pd=jnp.asarray(is_pd),
        node_min_eig=jnp.asarray(node_min_eig),
    )


__all__ = [
    "MeshTeacher",
    "build_teacher_mesh",
]
