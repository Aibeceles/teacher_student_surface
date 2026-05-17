"""Tests for :func:`build_character_teacher_mesh_periodic`.

Covers the structural property the periodic teacher exists to provide:

- ``V_re`` / ``V_im`` match analytic ``cos(2 pi (x + y) / p)`` /
  ``sin(2 pi (x + y) / p)`` on the *entire* mesh, not just at lattice
  nodes.
- ``GX_re`` / ``GX_im`` match the analytic gradient.
- ``unit_circle_residual`` (max over mesh of ``||V|^2 - 1|``) is at the
  numerical floor (vs ``~ 0.5+`` for the polynomial teacher at p=8).
- ``Z_W`` vanishes at every lattice node and is positive elsewhere.
- ``node_min_eig > 0`` at every lattice node (PD certificate works).
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


from sobolev_distill_character import (  # noqa: E402
    build_character_teacher_mesh,
    build_character_teacher_mesh_periodic,
)


_MODULI = [4, 5, 8]
_TOL = 1e-5  # JAX defaults to float32; the underlying NumPy is float64.


@pytest.mark.parametrize("p", _MODULI)
def test_value_matches_analytic_character_on_full_mesh(p: int) -> None:
    nodes = np.arange(p, dtype=np.float64)
    teacher = build_character_teacher_mesh_periodic(nodes, nodes, p, mesh_n=64, lam=1.0)
    xs = np.asarray(teacher.xs, dtype=np.float64)
    ys = np.asarray(teacher.ys, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    V_truth = np.exp(2j * np.pi * (XX + YY) / p)
    V = np.asarray(teacher.V_re) + 1j * np.asarray(teacher.V_im)
    np.testing.assert_allclose(V, V_truth, atol=_TOL)


@pytest.mark.parametrize("p", _MODULI)
def test_unit_circle_residual_is_at_numerical_floor(p: int) -> None:
    nodes = np.arange(p, dtype=np.float64)
    teacher = build_character_teacher_mesh_periodic(nodes, nodes, p, mesh_n=64, lam=1.0)
    V = np.asarray(teacher.V_re) + 1j * np.asarray(teacher.V_im)
    abs_dev = float(np.max(np.abs(np.abs(V) - 1.0)))
    assert abs_dev <= _TOL, f"|V| - 1 max = {abs_dev:.4e}"

    # The polynomial teacher at the same p has a much larger off-lattice
    # deviation; assert that the periodic version is materially better.
    poly = build_character_teacher_mesh(nodes, nodes, p, mesh_n=64, lam=1.0)
    Vp = np.asarray(poly.V_re) + 1j * np.asarray(poly.V_im)
    poly_dev = float(np.max(np.abs(np.abs(Vp) - 1.0)))
    assert abs_dev < poly_dev / 100.0, (abs_dev, poly_dev)


@pytest.mark.parametrize("p", _MODULI)
def test_gradient_matches_analytic(p: int) -> None:
    nodes = np.arange(p, dtype=np.float64)
    teacher = build_character_teacher_mesh_periodic(nodes, nodes, p, mesh_n=64, lam=1.0)
    xs = np.asarray(teacher.xs, dtype=np.float64)
    ys = np.asarray(teacher.ys, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    V_truth = np.exp(2j * np.pi * (XX + YY) / p)
    GX_truth = (2j * np.pi / p) * V_truth
    GY_truth = (2j * np.pi / p) * V_truth
    GX = np.asarray(teacher.GX_re) + 1j * np.asarray(teacher.GX_im)
    GY = np.asarray(teacher.GY_re) + 1j * np.asarray(teacher.GY_im)
    np.testing.assert_allclose(GX, GX_truth, atol=_TOL)
    np.testing.assert_allclose(GY, GY_truth, atol=_TOL)


@pytest.mark.parametrize("p", _MODULI)
def test_torus_bowl_vanishes_at_nodes(p: int) -> None:
    nodes = np.arange(p, dtype=np.float64)
    teacher = build_character_teacher_mesh_periodic(nodes, nodes, p, mesh_n=64, lam=1.0)
    Z_W = np.asarray(teacher.Z_W)
    is_node = np.asarray(teacher.is_node)
    assert is_node.any(), "expected at least one lattice node in the mesh"
    np.testing.assert_allclose(Z_W[is_node], 0.0, atol=_TOL)
    # Off-lattice strictly positive.
    off = ~is_node
    if off.any():
        assert float(Z_W[off].min()) > 0.0


@pytest.mark.parametrize("p", _MODULI)
def test_node_min_eig_positive(p: int) -> None:
    nodes = np.arange(p, dtype=np.float64)
    teacher = build_character_teacher_mesh_periodic(nodes, nodes, p, mesh_n=64, lam=1.0)
    nme = np.asarray(teacher.node_min_eig)
    assert float(nme.min()) > 0.0, f"min_eig = {float(nme.min())}"
