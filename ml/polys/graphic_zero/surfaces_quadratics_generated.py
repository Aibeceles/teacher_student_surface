"""Generate full MaxN quadratic sets from closed-form laws."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .evaluate_grid_completion import complete_evaluate_value_grid


def _coefficients_from_laws(n: int, max_n: int) -> tuple[float, float, float]:
    k = 2 ** (n - 2)
    coeff2 = k
    coeff1 = -((2 * max_n - 3) * k + 1)
    coeff0 = (max_n**2 - 3 * max_n + 4) * k + max_n
    return coeff0, coeff1, coeff2


def _evaluate_quadratic(coeff0: float, coeff1: float, coeff2: float, x: float) -> float:
    return coeff2 * (x**2) + coeff1 * x + coeff0


@dataclass(frozen=True)
class GeneratedMaxNQuadraticsTableBuilder:
    """Build quadratic rows N=0..MaxN-1 with native P_N coefficients."""

    max_n: str
    complete_grid: bool = True
    target_max: int | None = None

    def build(self) -> pd.DataFrame:
        max_n_int = int(self.max_n)
        rows: list[dict[str, float | int | str]] = []

        for n in range(0, max_n_int):
            coeff0, coeff1, coeff2 = _coefficients_from_laws(n=n, max_n=max_n_int)
            evaluate_value = _evaluate_quadratic(coeff0, coeff1, coeff2, float(max_n_int))
            rows.append(
                {
                    "N": str(n),
                    "coeff0": float(coeff0),
                    "coeff1": float(coeff1),
                    "coeff2": float(coeff2),
                    "MaxN": str(max_n_int),
                    "index_Result": float(evaluate_value),
                    "evaluateValue": int(2**n),
                }
            )

        out = pd.DataFrame(
            rows,
            columns=["N", "coeff0", "coeff1", "coeff2", "MaxN", "index_Result", "evaluateValue"],
        )
        if not self.complete_grid:
            return out

        target_max = self.target_max if self.target_max is not None else 2 ** (max_n_int - 1)
        return complete_evaluate_value_grid(out, target_max=target_max)
