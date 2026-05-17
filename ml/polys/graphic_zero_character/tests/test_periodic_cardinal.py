"""Tests for the periodic-cardinal (Dirichlet-kernel) envelopes.

Covers:

- Cardinal property ``alpha_i^per(v_j) == delta_{ij}`` to machine precision.
- Partition of unity ``Sum_i alpha_i^per(x) == 1`` for any ``x``.
- Single-Fourier-mode reproduction: the periodic Lagrange einsum on the
  character lattice values reproduces the analytic ``zeta**((x + y) mod p)``
  exactly off the lattice -- the structural property that justifies dropping
  the Birkhoff slope channels in the periodic teacher.
- Derivative reproduction ``Sum_i alpha_i^per'(x) zeta**i == (2*pi*1j / K) zeta**(x/K)``.
- Drop-in tensor helpers from
  :mod:`graphic_zero.hermite_barycentric_gpu` produce ``AX_per == eye(K)``
  and zero ``B*`` channels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _ensure_polys_on_path() -> None:
    here = Path(__file__).resolve()
    polys_root = here.parent.parent.parent
    if str(polys_root) not in sys.path:
        sys.path.insert(0, str(polys_root))


_ensure_polys_on_path()


from graphic_zero.hermite_barycentric_gpu import (  # noqa: E402
    mesh_periodic_tensors,
    precompute_periodic_alpha_second_at_nodes,
    precompute_periodic_tensors_at_nodes,
)
from graphic_zero.surfaces_barycentric import (  # noqa: E402
    periodic_cardinal,
    periodic_cardinal_prime,
    periodic_cardinal_second,
)


_MODULI = [4, 5, 7, 8, 11]


@pytest.mark.parametrize("K", _MODULI)
def test_cardinal_property(K: int) -> None:
    nodes = np.arange(K, dtype=np.float64)
    A = np.array(
        [[periodic_cardinal(i, nodes, float(nodes[j]), K) for j in range(K)] for i in range(K)]
    )
    np.testing.assert_allclose(A, np.eye(K), atol=1e-12)


@pytest.mark.parametrize("K", _MODULI)
def test_vectorised_matches_scalar(K: int) -> None:
    nodes = np.arange(K, dtype=np.float64)
    xs = np.linspace(0.0, K, 7 * K + 1)[:-1]
    for i in range(K):
        vec = periodic_cardinal(i, nodes, xs, K)
        scal = np.array([periodic_cardinal(i, nodes, float(x), K) for x in xs])
        np.testing.assert_allclose(np.asarray(vec), scal, atol=1e-13)


@pytest.mark.parametrize("K", _MODULI)
def test_partition_of_unity(K: int) -> None:
    nodes = np.arange(K, dtype=np.float64)
    xs = np.linspace(0.0, K, 7 * K + 1)
    pu = sum(np.asarray(periodic_cardinal(i, nodes, xs, K)) for i in range(K))
    np.testing.assert_allclose(pu, np.ones_like(xs), atol=1e-12)


@pytest.mark.parametrize("K", _MODULI)
def test_single_fourier_mode_value_reproduction(K: int) -> None:
    """``Sum_i alpha_i^per(x) zeta**i = zeta**x`` with ``zeta = exp(2 pi 1j / K)``."""
    nodes = np.arange(K, dtype=np.float64)
    xs = np.linspace(0.0, K, 11 * K + 1)
    zeta_i = np.exp(2j * np.pi * nodes / K)
    z = sum(np.asarray(periodic_cardinal(i, nodes, xs, K)) * zeta_i[i] for i in range(K))
    z_truth = np.exp(2j * np.pi * xs / K)
    np.testing.assert_allclose(z, z_truth, atol=1e-12)


@pytest.mark.parametrize("K", _MODULI)
def test_single_fourier_mode_first_derivative(K: int) -> None:
    """``Sum_i alpha_i^per'(x) zeta**i = (2 pi 1j / K) zeta**x``."""
    nodes = np.arange(K, dtype=np.float64)
    xs = np.linspace(0.0, K, 11 * K + 1)
    zeta_i = np.exp(2j * np.pi * nodes / K)
    zp = sum(
        np.asarray(periodic_cardinal_prime(i, nodes, xs, K)) * zeta_i[i] for i in range(K)
    )
    zp_truth = (2j * np.pi / K) * np.exp(2j * np.pi * xs / K)
    np.testing.assert_allclose(zp, zp_truth, atol=1e-11)


@pytest.mark.parametrize("K", _MODULI)
def test_single_fourier_mode_second_derivative(K: int) -> None:
    """``Sum_i alpha_i^per''(x) zeta**i = -(2 pi / K)**2 zeta**x``."""
    nodes = np.arange(K, dtype=np.float64)
    xs = np.linspace(0.0, K, 11 * K + 1)
    zeta_i = np.exp(2j * np.pi * nodes / K)
    zpp = sum(
        np.asarray(periodic_cardinal_second(i, nodes, xs, K)) * zeta_i[i] for i in range(K)
    )
    zpp_truth = -((2.0 * np.pi / K) ** 2) * np.exp(2j * np.pi * xs / K)
    np.testing.assert_allclose(zpp, zpp_truth, atol=1e-11)


@pytest.mark.parametrize("K", _MODULI)
def test_bivariate_character_reproduction(K: int) -> None:
    """The 2D einsum on ``T_ij = zeta**((i + j) mod K)`` reproduces
    ``zeta**((x + y) mod K)`` exactly (continuously, not just at lattice nodes).

    This is the structural property the periodic teacher relies on: with
    periodic alpha alone (no Birkhoff slope channels), the value einsum is
    the analytic interpolant of the modular addition character signal.
    """
    nodes = np.arange(K, dtype=np.float64)
    xs = np.linspace(0.0, K, 5 * K + 1)
    ys = np.linspace(0.0, K, 5 * K + 1)
    T = np.exp(2j * np.pi * ((nodes[:, None] + nodes[None, :]) % K) / K)
    AX = np.array([periodic_cardinal(i, nodes, xs, K) for i in range(K)]).T  # (Nx, K)
    AY = np.array([periodic_cardinal(j, nodes, ys, K) for j in range(K)]).T  # (Ny, K)
    V = np.einsum("ai,bj,ij->ab", AX, AY, T)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    V_truth = np.exp(2j * np.pi * (XX + YY) / K)
    np.testing.assert_allclose(V, V_truth, atol=1e-11)


@pytest.mark.parametrize("K", _MODULI)
def test_precompute_helpers_match_eye(K: int) -> None:
    nodes = np.arange(K, dtype=np.float64)
    AX, BX, AXP, BXP = precompute_periodic_tensors_at_nodes(nodes)
    np.testing.assert_allclose(AX, np.eye(K), atol=1e-12)
    assert np.all(BX == 0.0)
    assert np.all(BXP == 0.0)
    A2 = precompute_periodic_alpha_second_at_nodes(nodes)
    # On the integer lattice, A2 @ zeta_i should match the analytic second
    # derivative of zeta**(x/K) at x = a, which is -(2pi/K)^2 zeta**(a/K).
    zeta = np.exp(2j * np.pi * nodes / K)
    z_pred = A2 @ zeta
    z_truth = -((2.0 * np.pi / K) ** 2) * zeta
    np.testing.assert_allclose(z_pred, z_truth, atol=1e-11)


@pytest.mark.parametrize("K", _MODULI)
def test_mesh_periodic_tensors_zero_b_channels(K: int) -> None:
    nodes = np.arange(K, dtype=np.float64)
    xs = np.linspace(0.0, K, 5 * K + 1)
    ys = np.linspace(0.0, K, 4 * K + 1)
    ax, bx, lx, ay, by, ly = mesh_periodic_tensors(xs, ys, nodes, nodes)
    assert ax.shape == (xs.size, K)
    assert ay.shape == (ys.size, K)
    np.testing.assert_array_equal(ax, lx)
    np.testing.assert_array_equal(ay, ly)
    assert np.all(bx == 0.0)
    assert np.all(by == 0.0)
    # Off-lattice single-mode reproduction via the einsum.
    zeta = np.exp(2j * np.pi * nodes / K)
    z_pred = ax @ zeta
    z_truth = np.exp(2j * np.pi * xs / K)
    np.testing.assert_allclose(z_pred, z_truth, atol=1e-11)
