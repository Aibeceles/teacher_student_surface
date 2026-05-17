"""Barycentric Lagrange basis (SciPy) for stable Hermite/Birkhoff evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

try:
    from scipy.interpolate import BarycentricInterpolator
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install scipy for barycentric Lagrange bases: pip install scipy") from exc


def _derivative_callable(interp: BarycentricInterpolator) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """SciPy >= 1.14: ``interp.derivative(x, der=1)`` — no separate derivative interpolator object."""

    def lp(x: float | np.ndarray) -> float | np.ndarray:
        xa = np.asarray(x, dtype=float)
        out = interp.derivative(xa, der=1)
        if xa.ndim == 0:
            return float(out)
        return np.asarray(out, dtype=float)

    return lp


def build_barycentric_lagrange_basis(
    nodes: Sequence[float],
) -> tuple[list[BarycentricInterpolator], list[Callable[[float | np.ndarray], float | np.ndarray]]]:
    """Return per-node Lagrange basis interpolators L_j and first-derivative callables L'_j.

    L_j interpolates the Kronecker vector e_j on ``nodes`` (unique ascending recommended).
    """
    nodes_arr = np.asarray(nodes, dtype=float)
    k = nodes_arr.size
    l_basis: list[BarycentricInterpolator] = []
    lp_basis: list[Callable[[float | np.ndarray], float | np.ndarray]] = []
    for j in range(k):
        y = np.zeros(k, dtype=float)
        y[j] = 1.0
        interp = BarycentricInterpolator(nodes_arr, y)
        l_basis.append(interp)
        lp_basis.append(_derivative_callable(interp))
    return l_basis, lp_basis


def _as_float(val: float | np.ndarray) -> float:
    return float(np.asarray(val, dtype=float).reshape(-1)[0])


def lagrange_basis_at_node(L_list: Sequence[BarycentricInterpolator], i: int, x: float) -> float:
    return _as_float(L_list[i](x))


def lagrange_prime_at_node(
    Lp_list: Sequence[Callable[[float | np.ndarray], float | np.ndarray]],
    i: int,
    x: float,
) -> float:
    return _as_float(Lp_list[i](x))


def hermite_alpha_beta(
    i: int,
    nodes: Sequence[float],
    L_list: Sequence[BarycentricInterpolator],
    Lp_list: Sequence[Callable[[float | np.ndarray], float | np.ndarray]],
    x: float,
) -> tuple[float, float]:
    """alpha_i, beta_i at x (same closed form as expanded-Polynomial Hermite construction)."""
    v_i = float(nodes[i])
    ell_x = _as_float(L_list[i](x))
    ell_prime_vi = _as_float(Lp_list[i](v_i))
    alpha = (1.0 - 2.0 * (x - v_i) * ell_prime_vi) * (ell_x**2)
    beta = (x - v_i) * (ell_x**2)
    return alpha, beta


def hermite_alpha_beta_prime(
    i: int,
    nodes: Sequence[float],
    L_list: Sequence[BarycentricInterpolator],
    Lp_list: Sequence[Callable[[float | np.ndarray], float | np.ndarray]],
    x: float,
) -> tuple[float, float, float, float]:
    """Return (alpha, beta, alpha', beta') at x using L_i, L_i'."""
    v_i = float(nodes[i])
    L = _as_float(L_list[i](x))
    Lp = _as_float(Lp_list[i](x))
    c = _as_float(Lp_list[i](v_i))
    one_minus = 1.0 - 2.0 * (x - v_i) * c
    d_one_minus = -2.0 * c
    alpha = one_minus * (L**2)
    alpha_p = d_one_minus * (L**2) + one_minus * 2.0 * L * Lp
    beta = (x - v_i) * (L**2)
    beta_p = (L**2) + (x - v_i) * 2.0 * L * Lp
    return alpha, beta, alpha_p, beta_p


def hermite_alpha_beta_second(
    i: int,
    nodes: Sequence[float],
    L_list: Sequence[BarycentricInterpolator],
    Lp_list: Sequence[Callable[[float | np.ndarray], float | np.ndarray]],
    x: float,
    eps: float = 1e-5,
) -> tuple[float, float]:
    """Second derivatives $\\alpha_i''(x)$, $\\beta_i''(x)$ at ``x``.

    Implemented in closed form from ``hermite_alpha_beta`` / ``hermite_alpha_beta_prime`` using
    ``L_i''(x)`` from SciPy ``BarycentricInterpolator.derivative(..., der=2)``.
    Finite-difference stepping is brittle on Hermite/tensor node grids at scale $10^2$+
    (`eps`) and is not used here; the ``eps`` argument is retained for API compatibility.
    """

    _ = eps  # kept for callers that pass spacing; analytic path ignores it

    v_i = float(nodes[i])
    interp = L_list[i]
    xa = np.asarray(x, dtype=float)
    # L'' from the same interpolant that defines alpha (avoids mismatched derivative objects).
    ell_x = float(interp(xa))
    ell_p_x = float(Lp_list[i](xa))
    ell_pp_x = float(interp.derivative(xa, der=2))

    c = _as_float(Lp_list[i](v_i))
    one_minus = 1.0 - 2.0 * (x - v_i) * c
    d_one_minus = -2.0 * c

    # alpha'' from alpha' = d_one_minus * L^2 + one_minus * 2 L Lp
    alpha_pp = (
        d_one_minus * 2.0 * ell_x * ell_p_x
        + (-2.0 * c) * (2.0 * ell_x * ell_p_x)
        + one_minus * 2.0 * (ell_p_x * ell_p_x + ell_x * ell_pp_x)
    )

    # beta = (x-v) L^2, beta' per hermite_alpha_beta_prime
    u = x - v_i
    beta_pp = 4.0 * ell_x * ell_p_x + 2.0 * u * (ell_p_x * ell_p_x + ell_x * ell_pp_x)

    return float(alpha_pp), float(beta_pp)


# ---------------------------------------------------------------------------
# Periodic cardinal envelopes (trig-polynomial Dirichlet kernel)
# ---------------------------------------------------------------------------
#
# These are the closed-form analogues of the polynomial Hermite ``alpha`` /
# ``alpha'`` / ``alpha''`` envelopes for the *periodic* boundary condition
# ``alpha_i(v_0) == alpha_i(v_K)`` etc., on ``K`` equispaced nodes.  The
# unique real periodic cardinal is the (modified, for even ``K``) Dirichlet
# kernel
#
#     alpha_i^per(x) = (1/K) sum_k exp(2*pi*j*k*(x - v_i) / K),
#
# with ``k`` ranging over ``{-K/2, ..., K/2 - 1}`` for even ``K`` and
# ``{-(K-1)/2, ..., (K-1)/2}`` for odd ``K``.  Taking ``.real`` extracts the
# real symmetric kernel: at the Nyquist ``k = -K/2`` (even ``K`` only) the
# imaginary ``i sin(pi (x - v_i))`` cancels and the real ``cos(pi (x - v_i))``
# completes the symmetric ``cos`` series.  Cardinal property
# ``alpha_i^per(v_j) == delta_{ij}`` holds for any ``K``.
#
# The basis spans frequencies ``|k| < K/2`` (plus Nyquist for even ``K``);
# any single-Fourier-mode signal in that band is reproduced *exactly* by the
# periodic Lagrange interpolation einsum
# ``sum_i alpha_i^per(x) * f(v_i)``, which is the structural property used by
# the character teacher to interpolate ``zeta**((x + y) mod p)`` smoothly off
# the lattice.
#
# All three helpers accept a scalar or array ``x`` and return matching shape.


def _periodic_dft_freqs(K: int) -> np.ndarray:
    """Return the symmetric (or near-symmetric for even K) DFT frequency band.

    ``k = {-floor(K/2), ..., K - floor(K/2) - 1}``.  For odd K this is
    symmetric; for even K it includes the negative Nyquist ``-K/2`` and
    excludes the positive Nyquist (the missing ``+K/2`` term is supplied by
    the ``.real`` extraction in the cardinal sums below).
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    half = K // 2
    return np.arange(-half, K - half, dtype=np.float64)


def _periodic_cardinal_eval(
    i: int,
    nodes: Sequence[float],
    x: float | np.ndarray,
    K: int,
    multiplier: np.ndarray | None = None,
) -> np.ndarray | float:
    """Common machinery for ``alpha_i^per`` and its derivatives.

    ``multiplier`` broadcasts over the DFT frequency axis.  Pass ``None`` for
    the cardinal itself, ``2*pi*j*k / K`` for the first derivative, and
    ``-(2*pi*k / K)**2`` for the second derivative.
    """
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    if not (0 <= int(i) < nodes_arr.size):
        raise IndexError(f"node index {i} out of range for K={nodes_arr.size}")
    x_arr = np.asarray(x, dtype=np.float64)
    orig_shape = x_arr.shape
    x_flat = x_arr.reshape(-1)
    K_f = float(K)
    k = _periodic_dft_freqs(int(K))
    theta = (x_flat[:, None] - float(nodes_arr[int(i)])) / K_f
    phase = 2.0 * np.pi * theta * k[None, :]
    summand = np.exp(1j * phase)
    if multiplier is not None:
        summand = summand * multiplier[None, :]
    val = summand.sum(axis=-1).real / K_f
    out = val.reshape(orig_shape) if orig_shape else val.reshape(())
    if orig_shape == ():
        return float(out)
    return np.asarray(out, dtype=np.float64)


def periodic_cardinal(
    i: int,
    nodes: Sequence[float],
    x: float | np.ndarray,
    K: int,
) -> float | np.ndarray:
    """Real periodic cardinal ``alpha_i^per(x)`` on ``K`` equispaced nodes.

    Cardinal property ``alpha_i^per(v_j) == delta_{ij}`` holds exactly for
    any ``K``.  Off the lattice ``alpha_i^per`` is a smooth periodic
    trig-polynomial of frequencies ``|k| <= K/2``.
    """
    return _periodic_cardinal_eval(i, nodes, x, K, multiplier=None)


def periodic_cardinal_prime(
    i: int,
    nodes: Sequence[float],
    x: float | np.ndarray,
    K: int,
) -> float | np.ndarray:
    """First derivative ``d/dx alpha_i^per(x)``."""
    K_f = float(K)
    k = _periodic_dft_freqs(int(K))
    multiplier = 1j * 2.0 * np.pi * k / K_f
    return _periodic_cardinal_eval(i, nodes, x, K, multiplier=multiplier)


def periodic_cardinal_second(
    i: int,
    nodes: Sequence[float],
    x: float | np.ndarray,
    K: int,
) -> float | np.ndarray:
    """Second derivative ``d^2/dx^2 alpha_i^per(x)``."""
    K_f = float(K)
    k = _periodic_dft_freqs(int(K))
    multiplier = -((2.0 * np.pi * k / K_f) ** 2)
    return _periodic_cardinal_eval(i, nodes, x, K, multiplier=multiplier.astype(np.complex128))
