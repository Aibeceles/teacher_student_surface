"""Complete missing integer evaluate values by binary composition."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

REQUIRED_COLUMNS = ("N", "coeff0", "coeff1", "coeff2", "MaxN", "index_Result", "evaluateValue")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _as_decimal(value: object) -> Decimal:
    return Decimal(str(value))


def complete_evaluate_value_grid(base_df: pd.DataFrame, target_max: int) -> pd.DataFrame:
    """Return dense evaluateValue rows from 1..target_max.

    Missing targets are synthesized by summing base power-of-two quadratics that
    correspond to set bits in the target integer.
    """
    if target_max < 1:
        raise ValueError("target_max must be >= 1")

    missing = [column for column in REQUIRED_COLUMNS if column not in base_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = base_df.loc[:, REQUIRED_COLUMNS].copy()
    df["evaluate_int"] = df["evaluateValue"].astype(int)
    existing_values = set(df["evaluate_int"].tolist())

    power_df = df[df["evaluate_int"].map(_is_power_of_two)].copy()
    if power_df.empty:
        raise ValueError("No base power-of-two evaluate rows available for grid completion.")

    power_rows: dict[int, pd.Series] = {}
    for _, row in power_df.iterrows():
        value = int(row["evaluate_int"])
        if value in power_rows:
            raise ValueError(f"Duplicate power-of-two evaluateValue row: {value}")
        power_rows[value] = row

    max_n_values = {str(v) for v in df["MaxN"].astype(str).tolist()}
    if len(max_n_values) != 1:
        raise ValueError(f"Expected a single MaxN value, got: {sorted(max_n_values)}")
    max_n = next(iter(max_n_values))

    synthetic_rows: list[dict[str, object]] = []
    for target in range(1, target_max + 1):
        if target in existing_values:
            continue

        components = [bit for bit in power_rows if target & bit]
        missing_components = [bit for bit in components if bit not in power_rows]
        if missing_components:
            raise ValueError(
                f"Cannot synthesize evaluateValue={target}, missing base components: {missing_components}"
            )

        coeff0 = sum((_as_decimal(power_rows[bit]["coeff0"]) for bit in components), start=Decimal(0))
        coeff1 = sum((_as_decimal(power_rows[bit]["coeff1"]) for bit in components), start=Decimal(0))
        coeff2 = sum((_as_decimal(power_rows[bit]["coeff2"]) for bit in components), start=Decimal(0))
        index_result = sum(
            (_as_decimal(power_rows[bit]["index_Result"]) for bit in components), start=Decimal(0)
        )

        synthetic_rows.append(
            {
                "N": str(target),
                "coeff0": float(coeff0),
                "coeff1": float(coeff1),
                "coeff2": float(coeff2),
                "MaxN": max_n,
                "index_Result": float(index_result),
                "evaluateValue": int(target),
            }
        )

    if synthetic_rows:
        df = pd.concat([df.drop(columns=["evaluate_int"]), pd.DataFrame(synthetic_rows)], ignore_index=True)
    else:
        df = df.drop(columns=["evaluate_int"])

    df["evaluate_sort"] = df["evaluateValue"].astype(int)
    df = df.sort_values("evaluate_sort", kind="stable").drop(columns=["evaluate_sort"]).reset_index(drop=True)
    return df.loc[:, REQUIRED_COLUMNS]
