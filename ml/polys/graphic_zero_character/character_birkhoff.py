"""Birkhoff-Hermite assembly for the character target ``T_ij = zeta**(i+j)``.

The character addition table is **not separable** per axis: its analytic
gradient ``d/dx zeta**(x+y) = (2*pi*i / p) * zeta**(x+y)`` depends on both
indices.  This breaks the 1D-slope shape ``Dx[i]``, ``Dy[j]`` used by
:func:`graphic_zero.hermite_barycentric_gpu.verify_fh_on_device`.  Here we
provide the 2D-slope analogue with einsum signatures
``"ai,bj,ij->ab"`` for every term.

The accompanying real-valued **modulus energy**

    E(x, y) = | f_H(x, y) - T_nearest |^2  +  lambda * Z_W(x, y)

inherits the bowl regulariser ``Z_W`` from the real path
(:func:`graphic_zero.hermite_barycentric_gpu.bowl_hess_diagonal_terms`).
At every lattice node ``f_H = T`` exactly (Birkhoff exactness), so the
``F * partial^2 F-bar`` second-order term in ``nabla^2 |F|^2`` vanishes
and the per-node Hessian collapses to

    nabla^2 E |_node  =  2 * Re[ conj(grad f_H) (grad f_H)^T ]
                       +  lambda * diag(w_xx, w_yy).

This is real symmetric, PSD always, and PD as soon as either ``lambda > 0``
or the real and imaginary parts of ``grad f_H`` are linearly independent.
The ``eigvalsh`` PD test from
:func:`graphic_zero.hermite_barycentric_gpu.f_m_diagnostics_on_device`
applies verbatim.

Notation: lowercase ``i`` / ``j`` are lattice indices; ``1j`` is the
imaginary unit (kept consistent with NumPy/Python convention).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def character_addition_table_with_slopes(
    nodes_x: np.ndarray,
    nodes_y: np.ndarray,
    p: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(T, Dx_ij, Dy_ij)`` for the character target.

    ``T[i, j] = zeta ** ((nodes_x[i] + nodes_y[j]) % p)`` and the analytic
    partial derivatives at the lattice node:

        Dx_ij[i, j] = (2*pi*1j / p) * T[i, j]
        Dy_ij[i, j] = (2*pi*1j / p) * T[i, j]

    Both slopes coincide because ``T`` depends on ``x + y``; we still return
    them separately so the einsum API matches the real path.

    ``nodes_x`` and ``nodes_y`` are interpreted as real numbers; their sum
    is reduced modulo ``p`` for the character lookup.  When the nodes are
    integers (the canonical Welch lattice ``{0, 1, ..., p-1}``) this gives
    the exact integer-indexed character table.
    """
    if p < 1:
        raise ValueError(f"modulus p must be >= 1, got {p}")
    nx = np.asarray(nodes_x, dtype=np.float64)
    ny = np.asarray(nodes_y, dtype=np.float64)
    sums = (nx[:, None] + ny[None, :]) % float(p)
    omega = 2.0 * math.pi / float(p)
    phase = omega * sums
    T = np.exp(1j * phase)
    deriv_factor = 1j * omega
    Dx_ij = deriv_factor * T
    Dy_ij = deriv_factor * T
    return T, Dx_ij, Dy_ij


def verify_fh_character_on_device(
    cp: Any,
    T: np.ndarray,
    Dx_ij: np.ndarray,
    Dy_ij: np.ndarray,
    AX: np.ndarray,
    BX: np.ndarray,
    AXP: np.ndarray,
    BXP: np.ndarray,
    AY: np.ndarray,
    BY: np.ndarray,
    AYP: np.ndarray,
    BYP: np.ndarray,
) -> tuple[float, float, float, Any, Any, Any]:
    """2D-slope einsum analogue of :func:`verify_fh_on_device` for complex T.

    Returns ``(rel_val_err, rel_grad_err_re, rel_grad_err_im, V, GX, GY)``.

    Shapes:
    - ``T``, ``Dx_ij``, ``Dy_ij``: ``(Kx, Ky)`` complex.
    - ``AX``, ``BX``, ``AXP``, ``BXP``: ``(Nx, Kx)`` real (Hermite envelopes).
    - ``AY``, ``BY``, ``AYP``, ``BYP``: ``(Ny, Ky)`` real.

    The output ``V``, ``GX``, ``GY`` are device tensors of shape ``(Nx, Ny)``
    complex.  Errors are reported as relative max-abs deviations against the
    expected lattice values.
    """
    T_d = cp.asarray(T, dtype=cp.complex128)
    Dxd = cp.asarray(Dx_ij, dtype=cp.complex128)
    Dyd = cp.asarray(Dy_ij, dtype=cp.complex128)
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
        + cp.einsum("ai,bj,ij->ab", BXd, AYd, Dxd)
        + cp.einsum("ai,bj,ij->ab", AXd, BYd, Dyd)
    )
    GX = (
        cp.einsum("ai,bj,ij->ab", AXPd, AYd, T_d)
        + cp.einsum("ai,bj,ij->ab", BXPd, AYd, Dxd)
        + cp.einsum("ai,bj,ij->ab", AXPd, BYd, Dyd)
    )
    GY = (
        cp.einsum("ai,bj,ij->ab", AXd, AYPd, T_d)
        + cp.einsum("ai,bj,ij->ab", BXd, AYPd, Dxd)
        + cp.einsum("ai,bj,ij->ab", AXd, BYPd, Dyd)
    )

    val_err = float(cp.max(cp.abs(V - T_d)))
    grad_re_err = float(cp.max(cp.abs(cp.real(GX - Dxd))))
    grad_im_err = float(cp.max(cp.abs(cp.imag(GX - Dxd))))
    grad_re_err = max(grad_re_err, float(cp.max(cp.abs(cp.real(GY - Dyd)))))
    grad_im_err = max(grad_im_err, float(cp.max(cp.abs(cp.imag(GY - Dyd)))))

    scale_value = max(float(cp.max(cp.abs(T_d))), 1.0)
    scale_grad = max(float(cp.max(cp.abs(Dxd))), float(cp.max(cp.abs(Dyd))), 1.0)
    return val_err / scale_value, grad_re_err / scale_grad, grad_im_err / scale_grad, V, GX, GY


def modulus_energy_diagnostics_on_device(
    cp: Any,
    T: np.ndarray,
    Dx_ij: np.ndarray,
    Dy_ij: np.ndarray,
    AX: np.ndarray,
    BX: np.ndarray,
    AXP: np.ndarray,
    BXP: np.ndarray,
    AY: np.ndarray,
    BY: np.ndarray,
    AYP: np.ndarray,
    BYP: np.ndarray,
    lam: float,
    wxx: np.ndarray,
    wyy: np.ndarray,
    scale_value: float = 1.0,
    scale_grad: float = 1.0,
) -> tuple[float, float, float]:
    """Return ``(rel_val_err, rel_grad_err, min_eig)`` for the modulus energy at lattice nodes.

    Builds the per-node Hessian

        H_E[a, b] = 2 Re[ conj(grad f_H) grad f_H^T ]  +  lam * diag(w_xx[a], w_yy[b])

    via complex einsums on the Birkhoff-Hermite basis tensors evaluated at
    lattice nodes (``AX`` etc. should be the node-restricted tensors of
    shape ``(Kx, Kx)`` and ``(Ky, Ky)``).  Reports the value/gradient
    residuals against the analytic ``T`` / ``Dx_ij`` / ``Dy_ij`` and the
    minimum eigenvalue of ``H_E`` across all nodes.
    """
    T_d = cp.asarray(T, dtype=cp.complex128)
    Dxd = cp.asarray(Dx_ij, dtype=cp.complex128)
    Dyd = cp.asarray(Dy_ij, dtype=cp.complex128)
    AXd = cp.asarray(AX, dtype=cp.float64)
    BXd = cp.asarray(BX, dtype=cp.float64)
    AXPd = cp.asarray(AXP, dtype=cp.float64)
    BXPd = cp.asarray(BXP, dtype=cp.float64)
    AYd = cp.asarray(AY, dtype=cp.float64)
    BYd = cp.asarray(BY, dtype=cp.float64)
    AYPd = cp.asarray(AYP, dtype=cp.float64)
    BYPd = cp.asarray(BYP, dtype=cp.float64)
    wxx_d = cp.asarray(wxx, dtype=cp.float64)
    wyy_d = cp.asarray(wyy, dtype=cp.float64)

    V0 = (
        cp.einsum("ai,bj,ij->ab", AXd, AYd, T_d)
        + cp.einsum("ai,bj,ij->ab", BXd, AYd, Dxd)
        + cp.einsum("ai,bj,ij->ab", AXd, BYd, Dyd)
    )
    val_err = float(cp.max(cp.abs(V0 - T_d)))

    GX0 = (
        cp.einsum("ai,bj,ij->ab", AXPd, AYd, T_d)
        + cp.einsum("ai,bj,ij->ab", BXPd, AYd, Dxd)
        + cp.einsum("ai,bj,ij->ab", AXPd, BYd, Dyd)
    )
    GY0 = (
        cp.einsum("ai,bj,ij->ab", AXd, AYPd, T_d)
        + cp.einsum("ai,bj,ij->ab", BXd, AYPd, Dxd)
        + cp.einsum("ai,bj,ij->ab", AXd, BYPd, Dyd)
    )
    grad_err = float(cp.max(cp.maximum(cp.abs(GX0 - Dxd), cp.abs(GY0 - Dyd))))

    # Per-node Gauss-Newton outer product: 2 * Re[ conj(g) g^T ]
    gx_re = cp.real(GX0)
    gx_im = cp.imag(GX0)
    gy_re = cp.real(GY0)
    gy_im = cp.imag(GY0)

    H00 = 2.0 * (gx_re * gx_re + gx_im * gx_im) + float(lam) * wxx_d[:, None]
    H11 = 2.0 * (gy_re * gy_re + gy_im * gy_im) + float(lam) * wyy_d[None, :]
    H01 = 2.0 * (gx_re * gy_re + gx_im * gy_im)

    h_stack = cp.stack(
        [
            cp.stack([H00, H01], axis=-1),
            cp.stack([H01, H11], axis=-1),
        ],
        axis=-2,
    )
    eigs = cp.linalg.eigvalsh(h_stack)
    min_eig = float(cp.min(eigs[..., 0]))

    rel_val = val_err / max(scale_value, 1.0)
    rel_grad = grad_err / max(scale_grad, 1.0)
    return rel_val, rel_grad, min_eig


__all__ = [
    "character_addition_table_with_slopes",
    "modulus_energy_diagnostics_on_device",
    "verify_fh_character_on_device",
]
