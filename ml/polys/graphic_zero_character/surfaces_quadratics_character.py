"""Character-targeted MaxN quadratic family.

Per-axis variant of :mod:`graphic_zero.surfaces_quadratics_generated` whose
boundary condition at the horizon is a primitive ``p``-th root of unity
instead of ``2**n``.  Concretely, for each ``n`` in ``{0, ..., p-1}`` the
quadratic ``p_n`` is the unique Lagrange interpolant through the geometric
ramp

    p_n(M-2) = zeta**(n-2),
    p_n(M-1) = zeta**(n-1),
    p_n(M)   = zeta**n,                  zeta = exp(2*pi*i / p).

In shifted coordinate ``u = x - M`` the closed form (verified analytically
at ``u in {0, -1, -2}``) is

    p_n(x) = (zeta**(n-2) / 2) * [
        (zeta - 1)**2 * u**2
        + (3*zeta - 1) * (zeta - 1) * u
        + 2 * zeta**2
    ]

which expands back to ``c_0 + c_1 * x + c_2 * x**2`` via ``u -> x - M``.

Bivariate cross-join ``F(x, y) = p_{n1}(x) * p_{n2}(y)`` evaluates at the
canonical horizon to ``F(M, M) = zeta**(n1 + n2)``, so modular addition is
recovered by ``(n1 + n2) mod p == round(p / (2*pi) * arg F(M, M))``.

The DataFrame schema mirrors :class:`graphic_zero.GeneratedMaxNQuadraticsTableBuilder`
so workbooks can join on ``index_Result`` / ``evaluateValue`` exactly as they
do for the real (``2**n``) family.  Coefficients are stored as Python
``complex`` objects (pandas ``object`` dtype).
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import pandas as pd


def _character_coefficients_from_laws(
    n: int,
    max_n: int,
    p: int,
) -> tuple[complex, complex, complex]:
    """Return ``(coeff0, coeff1, coeff2)`` of the Welch-style character quadratic.

    The output is in standard (non-shifted) form, i.e.
    ``p_n(x) = coeff0 + coeff1 * x + coeff2 * x**2``.

    Raises ``ValueError`` if ``p < 1``.  For ``p == 1`` the family collapses
    to the constant-zero quadratic (``zeta = 1`` makes every shifted
    coefficient vanish except the constant ``2*zeta**2``, but the prefactor
    yields ``zeta**(n-2) * 2 / 2 = 1``; see ``test_real_limit_degeneracy``).
    """
    if p < 1:
        raise ValueError(f"modulus p must be >= 1, got {p}")

    zeta = cmath.exp(2j * math.pi / p) if p > 1 else complex(1.0, 0.0)
    zeta_n_minus_2 = zeta ** (n - 2)
    m = float(max_n)

    # Shifted form coefficients in u = x - M
    c2_shift = zeta_n_minus_2 * (zeta - 1) ** 2 / 2
    c1_shift = zeta_n_minus_2 * (3 * zeta - 1) * (zeta - 1) / 2
    c0_shift = zeta_n_minus_2 * zeta * zeta  # = zeta**n

    # Expand u = x - M back into c0 + c1*x + c2*x**2:
    #   c2_shift * (x - M)**2 + c1_shift * (x - M) + c0_shift
    coeff2 = c2_shift
    coeff1 = -2 * m * c2_shift + c1_shift
    coeff0 = (m * m) * c2_shift - m * c1_shift + c0_shift
    return coeff0, coeff1, coeff2


def _evaluate_quadratic(
    coeff0: complex,
    coeff1: complex,
    coeff2: complex,
    x: complex | float,
) -> complex:
    """Standard ``a*x**2 + b*x + c`` evaluation; complex-typed."""
    return coeff2 * (x * x) + coeff1 * x + coeff0


@dataclass(frozen=True)
class CharacterMaxNQuadraticsTableBuilder:
    """Build character-quadratic rows ``n = 0, ..., p-1`` with native ``P_n``.

    ``modulus`` defaults to ``int(max_n)`` so ``p == M`` is the natural pick
    (matches Welch's "modulus equals lattice size" setup); pass an explicit
    integer to decouple them.

    ``complete_grid`` is ``False`` by default (opposite of the real version):
    additive composition via :func:`graphic_zero.complete_evaluate_value_grid`
    does **not** preserve the character semantic
    (``sum_b zeta**b != zeta**T``), so densifying yields a valid family with
    a different meaning ("evaluate to ``sum_b zeta**b``"), not modular
    addition.  Users wanting full ``Z/p`` coverage already get it from the
    base ``n in {0, ..., p-1}`` rows.
    """

    max_n: str
    modulus: int | None = None
    complete_grid: bool = False
    target_max: int | None = None

    def build(self) -> pd.DataFrame:
        max_n_int = int(self.max_n)
        p = int(self.modulus) if self.modulus is not None else max_n_int
        if p < 1:
            raise ValueError(f"modulus must be >= 1, got {p}")

        rows: list[dict[str, complex | int | str]] = []
        zeta = cmath.exp(2j * math.pi / p) if p > 1 else complex(1.0, 0.0)
        for n in range(0, p):
            coeff0, coeff1, coeff2 = _character_coefficients_from_laws(
                n=n, max_n=max_n_int, p=p
            )
            evaluate_value = _evaluate_quadratic(coeff0, coeff1, coeff2, float(max_n_int))
            rows.append(
                {
                    "N": str(n),
                    "coeff0": complex(coeff0),
                    "coeff1": complex(coeff1),
                    "coeff2": complex(coeff2),
                    "MaxN": str(max_n_int),
                    "index_Result": complex(evaluate_value),
                    "evaluateValue": complex(zeta**n),
                }
            )

        out = pd.DataFrame(
            rows,
            columns=[
                "N",
                "coeff0",
                "coeff1",
                "coeff2",
                "MaxN",
                "index_Result",
                "evaluateValue",
            ],
        )
        if not self.complete_grid:
            return out

        # Mechanical reuse only: this returns a valid densified family but
        # with a non-modular evaluation semantic.  See module docstring.
        from graphic_zero.evaluate_grid_completion import complete_evaluate_value_grid

        target_max = self.target_max if self.target_max is not None else 2 ** (max_n_int - 1)
        # complete_evaluate_value_grid expects integer evaluateValue; cast for
        # compatibility (drops the imaginary part of the character target).
        bridge = out.copy()
        bridge["evaluateValue"] = bridge["evaluateValue"].map(lambda z: int(round(z.real)))
        return complete_evaluate_value_grid(bridge, target_max=target_max)
