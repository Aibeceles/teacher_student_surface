"""Welch-style character variant of :mod:`graphic_zero`.

This sibling package introduces a complex-valued, character-targeted closed-
form quadratic family ``p_n`` with ``p_n(M) = zeta**n`` (``zeta`` a primitive
``p``-th root of unity).  Bivariate cross-join then realises modular
addition multiplicatively: ``p_{n1}(M) * p_{n2}(M) = zeta**(n1 + n2)``.

Reuses :mod:`graphic_zero` unchanged for the Hermite-Birkhoff machinery,
barycentric Lagrange bases, GPU einsums, and (optionally, with documented
semantic shift) the additive grid completion.
"""

from __future__ import annotations

from graphic_zero.evaluate_grid_completion import complete_evaluate_value_grid
from graphic_zero.hermite_barycentric_gpu import (
    addition_table,
    require_cupy,
    verify_fh_on_device,
)
from graphic_zero.surfaces_barycentric import (
    build_barycentric_lagrange_basis,
    hermite_alpha_beta,
    hermite_alpha_beta_prime,
    hermite_alpha_beta_second,
    lagrange_basis_at_node,
    lagrange_prime_at_node,
)

from .character_birkhoff import (
    character_addition_table_with_slopes,
    modulus_energy_diagnostics_on_device,
    verify_fh_character_on_device,
)
from .surfaces_quadratics_character import (
    CharacterMaxNQuadraticsTableBuilder,
    _character_coefficients_from_laws,
    _evaluate_quadratic,
)

__all__ = [
    "CharacterMaxNQuadraticsTableBuilder",
    "_character_coefficients_from_laws",
    "_evaluate_quadratic",
    "addition_table",
    "build_barycentric_lagrange_basis",
    "character_addition_table_with_slopes",
    "complete_evaluate_value_grid",
    "hermite_alpha_beta",
    "hermite_alpha_beta_prime",
    "hermite_alpha_beta_second",
    "lagrange_basis_at_node",
    "lagrange_prime_at_node",
    "modulus_energy_diagnostics_on_device",
    "require_cupy",
    "verify_fh_character_on_device",
    "verify_fh_on_device",
]
