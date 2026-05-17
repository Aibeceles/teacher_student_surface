"""Workbook-facing class for surfaces quadratics table generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .reduced_graph_table_live import QuadraticReducedGraphTableLive


class QuadraticsSurfaceTableBuilder:
    """Thin class facade used by `workbook/surfaces/quadratics/quadratics.ipynb`."""

    def __init__(
        self,
        *,
        range_low: int,
        range_high: int,
        max_n: str,
        db_properties_path: Path | None = None,
    ) -> None:
        self.range_low = range_low
        self.range_high = range_high
        self.max_n = max_n
        self.db_properties_path = db_properties_path

    def build(self) -> pd.DataFrame:
        return QuadraticReducedGraphTableLive(
            self.range_low,
            self.range_high,
            self.max_n,
            db_properties_path=self.db_properties_path,
        ).build()
