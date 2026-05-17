"""Live Neo4j s12QuadQ slice + pandas pipeline for reduced-by-N quadratic coefficients."""

from __future__ import annotations

from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

import pandas as pd

getcontext().prec = 50

_POLYS_ROOT = Path(__file__).resolve().parent.parent


def _db_config_candidates(explicit: Path | None) -> list[Path]:
    if explicit is not None:
        return [explicit]
    return [
        _POLYS_ROOT / "db.properties",
        Path("ml/polys/db.properties"),
        Path("db.properties"),
        Path("../db.properties"),
        Path("../../db.properties"),
        Path("../../../db.properties"),
    ]


def parse_db_properties(db_properties_path: Path | None = None) -> dict[str, str]:
    cfg_path = next((p for p in _db_config_candidates(db_properties_path) if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError("Could not find db.properties. Copy ml/polys/db.properties.example first.")

    props: dict[str, str] = {}
    for raw_line in cfg_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        props[key.strip()] = value.strip()

    required = ["neo4j.url", "neo4j.user", "neo4j.password", "neo4j.database"]
    missing = [k for k in required if not props.get(k)]
    if missing:
        raise ValueError(f"Missing db.properties keys: {missing}")

    raw_url = props["neo4j.url"].replace("jdbc:neo4j:", "")
    host_part = raw_url.split("//", 1)[-1]
    if ":" not in host_part:
        raw_url = f"{raw_url}:7687"
    props["bolt_url"] = raw_url
    return props


def _run_query(cfg: dict[str, str], cypher: str, parameters: dict[str, Any]) -> pd.DataFrame:
    try:
        from ml.neo4j.neo4jClient import Neo4jClient
    except ImportError:  # pragma: no cover
        Neo4jClient = None  # type: ignore[misc, assignment]

    if Neo4jClient is not None:
        client = Neo4jClient(cfg["bolt_url"], cfg["neo4j.user"], cfg["neo4j.password"])
        try:
            return client.run_query(cypher, cfg["neo4j.database"], parameters=parameters)
        finally:
            client.close()

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install neo4j driver: pip install neo4j") from exc

    driver = GraphDatabase.driver(
        cfg["bolt_url"],
        auth=(cfg["neo4j.user"], cfg["neo4j.password"]),
    )
    try:
        with driver.session(database=cfg["neo4j.database"]) as session:
            rows = [dict(record) for record in session.run(cypher, parameters)]
    finally:
        driver.close()
    return pd.DataFrame(rows)


def load_s12_quadq_live(
    cfg: dict[str, str],
    range_low: int,
    range_high: int,
    max_n: str,
) -> pd.DataFrame:
    cypher = """
    UNWIND range(toInteger($rangeLow), toInteger($rangeHigh)) AS n
    WITH toString(n) AS N, $maxN AS nMax
    MATCH (v:VertexNode)<-[]-(i:IndexedBy)-[]->(:Evaluate),
          (t:TwoSeqFactor)<-[]-(i)
    WHERE i.N = N AND i.MaxN = nMax AND i.Dimension = '2'
    RETURN i.N AS index,
           i.MaxN AS maxN,
           t.twoSeq AS rowScalar,
           '2' AS divisor,
           toString(CASE WHEN toString(v.Degree)='-1' THEN toInteger(v.Scalar) * 2 ELSE toInteger(v.Scalar) END) AS scalar,
           toString(CASE WHEN toString(v.Degree)='-1' THEN 0 ELSE toInteger(v.Degree) END) AS degree
    """
    rows_df = _run_query(
        cfg,
        cypher,
        {"rangeLow": range_low, "rangeHigh": range_high, "maxN": max_n},
    )

    if rows_df.empty:
        raise ValueError("Live query returned no rows. Check Neo4j connection and input range.")

    df = rows_df.copy()
    expected_cols = ["index", "maxN", "rowScalar", "divisor", "scalar", "degree"]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Query result missing columns: {missing_cols}")

    for col in expected_cols:
        df[col] = df[col].astype(str)
    return df[expected_cols]


def load_evaluate_by_index_live(
    cfg: dict[str, str],
    range_low: int,
    range_high: int,
    max_n: str,
) -> pd.DataFrame:
    cypher = """
    UNWIND range(toInteger($rangeLow), toInteger($rangeHigh)) AS n
    WITH toString(n) AS N, $maxN AS nMax
    MATCH (i:IndexedBy)-[]->(e:Evaluate)
    WHERE i.N = N AND i.MaxN = nMax AND i.Dimension = '2'
    RETURN DISTINCT i.N AS index, i.MaxN AS maxN, e.Value AS evaluateValue
    """
    df = _run_query(
        cfg,
        cypher,
        {"rangeLow": range_low, "rangeHigh": range_high, "maxN": max_n},
    )

    if df.empty:
        raise ValueError("Evaluate query returned no rows. Check Neo4j and range.")

    for col in ("index", "maxN", "evaluateValue"):
        if col not in df.columns:
            raise ValueError(f"Evaluate query missing column: {col}")
        df[col] = df[col].astype(str)
    return df.drop_duplicates(subset=["index", "maxN"]).reset_index(drop=True)


def _extend_result_row(row: pd.Series) -> pd.Series:
    scalar = Decimal(str(row["scalar"]))
    row_scalar = Decimal(str(row["rowScalar"]))
    degree = int(str(row["degree"]))
    divisor = Decimal(str(row["divisor"]))
    max_n = Decimal(str(row["maxN"]))
    result = (scalar / divisor) * (max_n**degree) * row_scalar
    return pd.Series(
        {
            "dimension": "2",
            "degree": str(row["degree"]),
            "scalar": str(row["scalar"]),
            "index": str(row["index"]),
            "maxIndex": str(row["maxN"]),
            "divisor": str(row["divisor"]),
            "result": format(result, "f"),
            "rowScalar": str(row["rowScalar"]),
        }
    )


def _run_pandas_pipeline(s12: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (ReducedMaxN8RangePivot, ReducedMaxN8RangeFinalResult)."""
    r_udf = s12.apply(_extend_result_row, axis=1)

    ds2 = r_udf.copy()
    ds2["weighted"] = ds2.apply(
        lambda r: Decimal(str(r["scalar"])) * Decimal(str(r["rowScalar"])) / Decimal(str(r["divisor"])),
        axis=1,
    )
    ds2["maxIndex_i"] = ds2["maxIndex"].astype(int)
    ds2["index_i"] = ds2["index"].astype(int)
    ds2["degree_i"] = ds2["degree"].astype(int)
    gds = ds2.groupby(["maxIndex_i", "index_i", "degree_i"], as_index=False)["weighted"].sum()
    gds = gds.rename(
        columns={"maxIndex_i": "_1", "index_i": "_2", "degree_i": "_3", "weighted": "scalar_Result"}
    )
    gds["scalar_Result"] = gds["scalar_Result"].map(lambda v: format(v, "f"))

    df2 = gds.copy()
    reduced = pd.DataFrame(
        {
            "MaxN": df2["_1"].astype(str),
            "N": df2["_2"].astype(str),
            "Degree": df2["_3"].astype(str),
            "Scalar": df2["scalar_Result"].astype(str),
        }
    )

    agg = reduced.copy()
    agg["Scalar_d"] = agg["Scalar"].map(lambda v: Decimal(str(v)))
    pivot21 = agg.pivot_table(index="N", columns="Degree", values="Scalar_d", aggfunc="sum")
    for c in ("0", "1", "2"):
        if c not in pivot21.columns:
            pivot21[c] = Decimal(0)
    reduced_pivot = pivot21[["0", "1", "2"]].reset_index()

    rr = reduced.copy()
    rows: list[dict[str, Any]] = []
    for _, r in rr.iterrows():
        scalar = Decimal(str(r["Scalar"]))
        max_n = int(str(r["MaxN"]))
        degree = int(str(r["Degree"]))
        result = scalar * (Decimal(max_n) ** degree)
        rows.append(
            {
                "Scalar": str(r["Scalar"]),
                "N": str(r["N"]),
                "Degree": str(r["Degree"]),
                "result": format(result, "f"),
                "MaxN": str(r["MaxN"]),
            }
        )
    res_df = pd.DataFrame(rows)

    res_df["result_d"] = res_df["result"].map(lambda v: Decimal(str(v)))
    final = res_df.groupby(["MaxN", "N"], as_index=False)["result_d"].sum()
    final = final.rename(columns={"result_d": "index_Result"})
    final["index_Result"] = final["index_Result"].map(lambda v: format(v, "f"))

    return reduced_pivot, final


def _merge_quadratics_reduced_by_n(
    reduced_pivot: pd.DataFrame,
    final_part: pd.DataFrame,
    eval_part: pd.DataFrame,
    max_n: str,
) -> pd.DataFrame:
    red = reduced_pivot.copy()
    red["N"] = red["N"].astype(str)
    red = red.rename(columns={"0": "coeff0", "1": "coeff1", "2": "coeff2"})
    red["MaxN"] = str(max_n)

    fp = final_part.copy()
    fp["N"] = fp["N"].astype(str)
    fp["MaxN"] = fp["MaxN"].astype(str)

    out = red.merge(fp[["N", "MaxN", "index_Result"]], on=["N", "MaxN"], how="left")
    out = out.merge(
        eval_part.rename(columns={"index": "N_eval", "maxN": "MaxN_eval"}),
        left_on=["N", "MaxN"],
        right_on=["N_eval", "MaxN_eval"],
        how="left",
    )
    return out.drop(columns=["N_eval", "MaxN_eval"], errors="ignore")


class QuadraticReducedGraphTableLive:
    """Live Neo4j query + in-memory pipeline producing ``quadratics_reduced_by_n`` (no root columns)."""

    def __init__(
        self,
        range_low: int,
        range_high: int,
        max_n: str,
        *,
        db_properties_path: Path | None = None,
    ) -> None:
        self.range_low = range_low
        self.range_high = range_high
        self.max_n = max_n
        self._db_properties_path = db_properties_path

    def build(self) -> pd.DataFrame:
        cfg = parse_db_properties(self._db_properties_path)
        s12 = load_s12_quadq_live(cfg, self.range_low, self.range_high, self.max_n)
        eval_part = load_evaluate_by_index_live(cfg, self.range_low, self.range_high, self.max_n)
        reduced_pivot, final_part = _run_pandas_pipeline(s12)
        return _merge_quadratics_reduced_by_n(reduced_pivot, final_part, eval_part, self.max_n)
