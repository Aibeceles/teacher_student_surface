"""GPU helpers for barycentric Hermite / Birkhoff verification (CuPy).

SciPy ``BarycentricInterpolator`` stays on CPU; this module precomputes Hermite
tensors at nodes and runs ``cp.einsum`` / batched ``eigvalsh`` on device.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .surfaces_barycentric import (
    hermite_alpha_beta,
    hermite_alpha_beta_prime,
    hermite_alpha_beta_second,
    lagrange_basis_at_node,
    lagrange_prime_at_node,
    periodic_cardinal,
    periodic_cardinal_prime,
    periodic_cardinal_second,
)

if TYPE_CHECKING:
    import cupy as cp


def require_cupy() -> Any:
    try:
        import cupy as cp  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "CuPy is required for GPU Hermite helpers. Install a wheel matching your CUDA, "
            "e.g. pip install cupy-cuda12x"
        ) from exc
    return cp


def precompute_hermite_tensors_at_nodes(
    nodes: Sequence[float],
    L_list: Sequence[Any],
    Lp_list: Sequence[Callable[..., Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shape (K, K): index [a, i] = basis i evaluated at node coordinate nodes[a]."""
    nodes_list = [float(v) for v in nodes]
    k = len(nodes_list)
    ax = np.zeros((k, k), dtype=np.float64)
    bx = np.zeros((k, k), dtype=np.float64)
    axp = np.zeros((k, k), dtype=np.float64)
    bxp = np.zeros((k, k), dtype=np.float64)
    for a, x_a in enumerate(nodes_list):
        for i in range(k):
            a0, b0 = hermite_alpha_beta(i, nodes_list, L_list, Lp_list, x_a)
            a1, b1, ap, bp = hermite_alpha_beta_prime(i, nodes_list, L_list, Lp_list, x_a)
            ax[a, i] = a0
            bx[a, i] = b0
            axp[a, i] = ap
            bxp[a, i] = bp
    return ax, bx, axp, bxp


def precompute_alpha_second_at_nodes(
    nodes: Sequence[float],
    L_list: Sequence[Any],
    Lp_list: Sequence[Callable[..., Any]],
) -> np.ndarray:
    """Second derivative of alpha_i at nodes[a]; shape (K, K)."""
    nodes_list = [float(v) for v in nodes]
    k = len(nodes_list)
    a2 = np.zeros((k, k), dtype=np.float64)
    for a, x_a in enumerate(nodes_list):
        for i in range(k):
            a2[a, i], _ = hermite_alpha_beta_second(i, nodes_list, L_list, Lp_list, x_a)
    return a2


def precompute_lagrange_prime_grid(
    Lp_list: Sequence[Callable[..., Any]],
    nodes: Sequence[float],
) -> np.ndarray:
    """Lp_grid[a, i] = ell_i'(nodes[a]); shape (K, K)."""
    nodes_list = [float(v) for v in nodes]
    k = len(nodes_list)
    g = np.zeros((k, k), dtype=np.float64)
    for a, x_a in enumerate(nodes_list):
        for i in range(k):
            g[a, i] = lagrange_prime_at_node(Lp_list, i, x_a)
    return g


def bowl_hess_diagonal_terms(
    nodes: np.ndarray,
    lp_grid: np.ndarray,
) -> np.ndarray:
    """w[a] = 2 + sum_i 2 * (nodes[a]-nodes[i])^2 * lp_grid[a,i]^2 (i==a term is zero)."""
    nodes = np.asarray(nodes, dtype=np.float64)
    diff = nodes[:, None] - nodes[None, :]
    contrib = 2.0 * (diff**2) * (lp_grid**2)
    return 2.0 + contrib.sum(axis=1)


def mesh_hermite_lagrange_tensors(
    xs: np.ndarray,
    ys: np.ndarray,
    nodes_x: Sequence[float],
    nodes_y: Sequence[float],
    Lx: Sequence[Any],
    Lpx: Sequence[Callable[..., Any]],
    Ly: Sequence[Any],
    Lpy: Sequence[Callable[..., Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CPU tensors for mesh evaluation (same as notebook mesh_matrices list comps)."""
    kx = len(nodes_x)
    ky = len(nodes_y)
    ax_x = np.zeros((xs.size, kx), dtype=np.float64)
    bx_x = np.zeros((xs.size, kx), dtype=np.float64)
    lx_x = np.zeros((xs.size, kx), dtype=np.float64)
    ay_y = np.zeros((ys.size, ky), dtype=np.float64)
    by_y = np.zeros((ys.size, ky), dtype=np.float64)
    ly_y = np.zeros((ys.size, ky), dtype=np.float64)
    nodes_xl = [float(v) for v in nodes_x]
    nodes_yl = [float(v) for v in nodes_y]
    for xi, x in enumerate(xs):
        for i in range(kx):
            ax_x[xi, i], bx_x[xi, i] = hermite_alpha_beta(i, nodes_xl, Lx, Lpx, float(x))
            lx_x[xi, i] = lagrange_basis_at_node(Lx, i, float(x))
    for yi, y in enumerate(ys):
        for j in range(ky):
            ay_y[yi, j], by_y[yi, j] = hermite_alpha_beta(j, nodes_yl, Ly, Lpy, float(y))
            ly_y[yi, j] = lagrange_basis_at_node(Ly, j, float(y))
    return ax_x, bx_x, lx_x, ay_y, by_y, ly_y


def addition_table(nodes_x: np.ndarray, nodes_y: np.ndarray) -> np.ndarray:
    return np.asarray(nodes_x, dtype=np.float64)[:, None] + np.asarray(nodes_y, dtype=np.float64)[None, :]


# ---------------------------------------------------------------------------
# Periodic-cardinal (Dirichlet) drop-in tensors
# ---------------------------------------------------------------------------
#
# These mirror ``precompute_hermite_tensors_at_nodes`` and
# ``mesh_hermite_lagrange_tensors`` but use the trig-polynomial cardinal
# envelopes from :mod:`graphic_zero.surfaces_barycentric` instead of the
# polynomial Hermite envelopes.  Because the periodic Lagrange interpolant
# of a single-Fourier-mode signal is itself, the Birkhoff slope channels
# (``BX``, ``BY``, ``Dx``, ``Dy``) can be set to zero and the existing
# einsum kernels (``verify_fh_*``, ``f_m_diagnostics_*``,
# ``mesh_surfaces_on_device``) stay verbatim.  The helpers therefore
# return ``BX_zero`` / ``BXP_zero`` / ``B2X_zero`` arrays alongside the
# periodic ``alpha`` tensors.


def precompute_periodic_tensors_at_nodes(
    nodes: Sequence[float],
    K: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(AX_per, BX_zero, AXP_per, BXP_zero)`` of shape ``(K, K)``.

    ``AX_per[a, i] = alpha_i^per(nodes[a])``; cardinal property gives
    ``AX_per == eye(K)`` exactly.  ``AXP_per[a, i] = alpha_i^per'(nodes[a])``.
    The B-channel tensors are zero arrays of matching shape so callers that
    feed the existing hermite einsum kernels with ``Dx_ij = 0`` see the
    B-channel einsums vanish.  ``K`` defaults to ``len(nodes)``.
    """
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    n = nodes_arr.size
    if K is None:
        K = n
    AX = np.zeros((n, n), dtype=np.float64)
    AXP = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        AX[:, i] = np.asarray(periodic_cardinal(i, nodes_arr, nodes_arr, int(K)), dtype=np.float64)
        AXP[:, i] = np.asarray(periodic_cardinal_prime(i, nodes_arr, nodes_arr, int(K)), dtype=np.float64)
    BX_zero = np.zeros((n, n), dtype=np.float64)
    BXP_zero = np.zeros((n, n), dtype=np.float64)
    return AX, BX_zero, AXP, BXP_zero


def precompute_periodic_alpha_second_at_nodes(
    nodes: Sequence[float],
    K: int | None = None,
) -> np.ndarray:
    """Return ``A2X_per`` of shape ``(K, K)`` with ``A2X_per[a, i] = alpha_i^per''(nodes[a])``."""
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    n = nodes_arr.size
    if K is None:
        K = n
    A2 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A2[:, i] = np.asarray(
            periodic_cardinal_second(i, nodes_arr, nodes_arr, int(K)), dtype=np.float64
        )
    return A2


def mesh_periodic_tensors(
    xs: np.ndarray,
    ys: np.ndarray,
    nodes_x: Sequence[float],
    nodes_y: Sequence[float],
    K_x: int | None = None,
    K_y: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CPU tensors for periodic-cardinal mesh evaluation.

    Returns ``(ax_x, bx_x_zero, lx_x, ay_y, by_y_zero, ly_y)`` matching the
    signature of :func:`mesh_hermite_lagrange_tensors`.  Here
    ``lx_x == ax_x == alpha^per_i(xs[a])`` because the trig-polynomial
    cardinal IS its own Lagrange basis on the lattice.  ``bx_x_zero`` and
    ``by_y_zero`` are zero arrays of shape ``(N, K)``.
    """
    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    nodes_xl = np.asarray(nodes_x, dtype=np.float64)
    nodes_yl = np.asarray(nodes_y, dtype=np.float64)
    if K_x is None:
        K_x = nodes_xl.size
    if K_y is None:
        K_y = nodes_yl.size
    nx = xs_arr.size
    ny = ys_arr.size
    kx = nodes_xl.size
    ky = nodes_yl.size
    ax_x = np.zeros((nx, kx), dtype=np.float64)
    ay_y = np.zeros((ny, ky), dtype=np.float64)
    for i in range(kx):
        ax_x[:, i] = np.asarray(
            periodic_cardinal(i, nodes_xl, xs_arr, int(K_x)), dtype=np.float64
        )
    for j in range(ky):
        ay_y[:, j] = np.asarray(
            periodic_cardinal(j, nodes_yl, ys_arr, int(K_y)), dtype=np.float64
        )
    bx_x_zero = np.zeros((nx, kx), dtype=np.float64)
    by_y_zero = np.zeros((ny, ky), dtype=np.float64)
    lx_x = ax_x
    ly_y = ay_y
    return ax_x, bx_x_zero, lx_x, ay_y, by_y_zero, ly_y


def verify_fh_on_device(
    cp: Any,
    T: np.ndarray,
    D_x: np.ndarray,
    D_y: np.ndarray,
    AX: np.ndarray,
    BX: np.ndarray,
    AXP: np.ndarray,
    BXP: np.ndarray,
    AY: np.ndarray,
    BY: np.ndarray,
    AYP: np.ndarray,
    BYP: np.ndarray,
) -> tuple[float, float, float, Any, Any, Any]:
    """Return (rel_val_err, rel_gx_err, rel_gy_err, V, GX, GY) on device."""
    T_d = cp.asarray(T, dtype=cp.float64)
    Dx = cp.asarray(D_x, dtype=cp.float64)
    Dy = cp.asarray(D_y, dtype=cp.float64)
    AXd = cp.asarray(AX, dtype=cp.float64)
    BXd = cp.asarray(BX, dtype=cp.float64)
    AXPd = cp.asarray(AXP, dtype=cp.float64)
    BXPd = cp.asarray(BXP, dtype=cp.float64)
    AYd = cp.asarray(AY, dtype=cp.float64)
    BYd = cp.asarray(BY, dtype=cp.float64)
    AYPd = cp.asarray(AYP, dtype=cp.float64)
    BYPd = cp.asarray(BYP, dtype=cp.float64)

    V = (
        cp.einsum("ai,bj,ij->ab", AXd, AYd, T_d)
        + cp.einsum("ai,bj,i->ab", BXd, AYd, Dx)
        + cp.einsum("ai,bj,j->ab", AXd, BYd, Dy)
    )
    GX = (
        cp.einsum("ai,bj,ij->ab", AXPd, AYd, T_d)
        + cp.einsum("ai,bj,i->ab", BXPd, AYd, Dx)
        + cp.einsum("ai,bj,j->ab", AXPd, BYd, Dy)
    )
    GY = (
        cp.einsum("ai,bj,ij->ab", AXd, AYPd, T_d)
        + cp.einsum("ai,bj,i->ab", BXd, AYPd, Dx)
        + cp.einsum("ai,bj,j->ab", AXd, BYPd, Dy)
    )

    val_err = float(cp.max(cp.abs(V - T_d)))
    gx_err = float(cp.max(cp.abs(GX - Dx[:, None])))
    gy_err = float(cp.max(cp.abs(GY - Dy[None, :])))

    scale_value = max(float(cp.max(cp.abs(T_d))), 1.0)
    scale_grad = max(float(cp.max(cp.abs(Dx))), float(cp.max(cp.abs(Dy))), 1.0)
    return val_err / scale_value, gx_err / scale_grad, gy_err / scale_grad, V, GX, GY


def f_m_diagnostics_on_device(
    cp: Any,
    T: np.ndarray,
    lam: float,
    AX: np.ndarray,
    AY: np.ndarray,
    AXP: np.ndarray,
    AYP: np.ndarray,
    A2X: np.ndarray,
    A2Y: np.ndarray,
    wxx: np.ndarray,
    wyy: np.ndarray,
    scale_value: float,
    scale_grad: float,
) -> tuple[float, float, float]:
    """Return (rel_val_err, rel_grad_err, min_eig) for f_M at given lambda.

    Tensors agree with ``precompute_*_at_nodes``: ``AX``.shape ``== (Nx, Kx)``, ``AY``.shape
    ``== (Ny, Ky)``, ``T``.shape ``== (Kx, Ky)``. Bowl weights broadcast as ``wxx``: (Nx,),
    ``wyy``: (Ny,); the Hessian grid is ``(Nx, Ny)``.
    """
    nx, kx_ax = AX.shape
    ny, ky_ay = AY.shape
    kt0, kt1 = T.shape
    if kx_ax != kt0 or ky_ay != kt1:
        raise ValueError(
            f"T shape {T.shape} incompatible with AX {AX.shape} and AY {AY.shape}: "
            "need T.shape == (AX.shape[1], AY.shape[1])."
        )
    wx = np.asarray(wxx, dtype=np.float64).ravel()
    wy = np.asarray(wyy, dtype=np.float64).ravel()
    if wx.shape[0] != nx:
        raise ValueError(f"wxx length {wx.shape[0]} must match AX.rows {nx}")
    if wy.shape[0] != ny:
        raise ValueError(f"wyy length {wy.shape[0]} must match AY.rows {ny}")

    T_d = cp.asarray(T, dtype=cp.float64)
    AXd = cp.asarray(AX, dtype=cp.float64)
    AYd = cp.asarray(AY, dtype=cp.float64)
    AXPd = cp.asarray(AXP, dtype=cp.float64)
    AYPd = cp.asarray(AYP, dtype=cp.float64)
    A2Xd = cp.asarray(A2X, dtype=cp.float64)
    A2Yd = cp.asarray(A2Y, dtype=cp.float64)
    wxx_d = cp.asarray(wxx, dtype=cp.float64)
    wyy_d = cp.asarray(wyy, dtype=cp.float64)

    V0 = cp.einsum("ai,bj,ij->ab", AXd, AYd, T_d)
    val_err = float(cp.max(cp.abs(V0 - T_d)))

    GX0 = cp.einsum("ai,bj,ij->ab", AXPd, AYd, T_d)
    GY0 = cp.einsum("ai,bj,ij->ab", AXd, AYPd, T_d)
    grad_err = float(cp.max(cp.maximum(cp.abs(GX0), cp.abs(GY0))))

    Hxx = cp.einsum("ai,bj,ij->ab", A2Xd, AYd, T_d)
    Hyy = cp.einsum("ai,bj,ij->ab", AXd, A2Yd, T_d)
    Hxy = cp.einsum("ai,bj,ij->ab", AXPd, AYPd, T_d)

    # Broadcast wxx[a], wyy[b] to (a,b)
    wxx_ab = wxx_d[:, None]
    wyy_ab = wyy_d[None, :]
    H00 = Hxx + float(lam) * wxx_ab
    H11 = Hyy + float(lam) * wyy_ab
    H01 = Hxy

    # Stack 2x2 Hermitian blocks: (a, b, 2, 2)
    h_stack = cp.stack(
        [
            cp.stack([H00, H01], axis=-1),
            cp.stack([H01, H11], axis=-1),
        ],
        axis=-2,
    )
    eigs = cp.linalg.eigvalsh(h_stack)
    min_eig = float(cp.min(eigs[..., 0]))

    rel_val = val_err / scale_value
    rel_grad = grad_err / scale_grad
    return rel_val, rel_grad, min_eig


def mesh_surfaces_on_device(
    cp: Any,
    T: np.ndarray,
    D_x: np.ndarray,
    D_y: np.ndarray,
    lam: float,
    ax_x: np.ndarray,
    bx_x: np.ndarray,
    lx_x: np.ndarray,
    ay_y: np.ndarray,
    by_y: np.ndarray,
    ly_y: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    nodes_x: np.ndarray,
    nodes_y: np.ndarray,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Return (XX, YY, Z_H, Z_M, Z_lin, Z_H0, Z_W) as CuPy arrays."""
    T_d = cp.asarray(T, dtype=cp.float64)
    Dx = cp.asarray(D_x, dtype=cp.float64)
    Dy = cp.asarray(D_y, dtype=cp.float64)
    axd = cp.asarray(ax_x, dtype=cp.float64)
    bxd = cp.asarray(bx_x, dtype=cp.float64)
    lxd = cp.asarray(lx_x, dtype=cp.float64)
    ayd = cp.asarray(ay_y, dtype=cp.float64)
    byd = cp.asarray(by_y, dtype=cp.float64)
    lyd = cp.asarray(ly_y, dtype=cp.float64)

    z_h = (
        cp.einsum("xi,yj,ij->xy", axd, ayd, T_d)
        + cp.einsum("xi,yj,i->xy", bxd, ayd, Dx)
        + cp.einsum("xi,yj,j->xy", axd, byd, Dy)
    )
    z_h0 = cp.einsum("xi,yj,ij->xy", axd, ayd, T_d)
    lx_sq = lxd**2
    ly_sq = lyd**2
    nodes_x = cp.asarray(nodes_x, dtype=cp.float64)
    nodes_y = cp.asarray(nodes_y, dtype=cp.float64)
    xs_d = cp.asarray(xs, dtype=cp.float64)
    ys_d = cp.asarray(ys, dtype=cp.float64)
    dx = (xs_d[:, None] - nodes_x[None, :]) ** 2
    dy = (ys_d[:, None] - nodes_y[None, :]) ** 2
    z_w = cp.einsum("xi,yj,xi->xy", lx_sq, ly_sq, dx) + cp.einsum("xi,yj,yj->xy", lx_sq, ly_sq, dy)
    z_m = z_h0 + float(lam) * z_w

    xx, yy = cp.meshgrid(xs_d, ys_d, indexing="ij")
    z_lin = xx + yy
    return xx, yy, z_h, z_m, z_lin, z_h0, z_w
