"""Snapshot paired daily close prices for the supplemental expiry universe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import polars as pl
from sqlalchemy import URL, create_engine, text


VERSION = "expiry_daily_close_facts_v1"
DEFAULT_UPSTREAM = Path(
    "maker/data/walkforward/"
    "prequential_challenger_ab12_supplemental_carry_20260821_v2_trade_fallback"
)
DEFAULT_OUTPUT = Path(
    "maker/data/walkforward/expiry_daily_close_facts_20260821_v1"
)
FUTURE_TABLE = "MarketInfo.taifex_futures_trades_daily"
SPOT_TABLE = "MarketInfo.twse_security_trades_daily"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def load_verified_expiry_marks(root: Path) -> tuple[pl.DataFrame, dict[str, object]]:
    marker_path = root / "complete.json"
    marker = _read_json(marker_path)
    declaration = marker.get("artifacts", {}).get("expiry_marks.parquet")
    path = root / "expiry_marks.parquet"
    if (
        marker.get("complete") is not True
        or not isinstance(declaration, dict)
        or not path.is_file()
        or _sha256(path) != declaration.get("sha256")
    ):
        raise ValueError("supplemental expiry mark source is incomplete or changed")
    marks = pl.read_parquet(path).select("Date", "ValueCode", "QuoteCode").sort(
        ["Date", "ValueCode", "QuoteCode"]
    )
    if marks.is_empty() or marks.select("Date", "ValueCode", "QuoteCode").n_unique() != marks.height:
        raise ValueError("supplemental expiry mark population is invalid")
    return marks, marker


def query_daily_close_facts(
    marks: pl.DataFrame,
    *,
    mysql_host: str,
    mysql_user: str,
    mysql_password: str,
) -> pl.DataFrame:
    dates = sorted(set(str(value) for value in marks["Date"].to_list()))
    placeholders = ", ".join(f":date_{index}" for index in range(len(dates)))
    params = {
        f"date_{index}": datetime.strptime(value, "%Y%m%d").date()
        for index, value in enumerate(dates)
    }
    engine = create_engine(
        URL.create(
            "mysql+pymysql",
            username=mysql_user,
            password=mysql_password,
            host=mysql_host,
            port=3306,
        ),
        pool_pre_ping=True,
    )
    future_sql = f"""
        SELECT DATE_FORMAT(date, '%Y%m%d') AS Date,
               quote_code AS QuoteCode,
               CAST(close_price AS DOUBLE) AS future_close_price,
               CAST(settlement_price AS DOUBLE) AS future_settlement_price,
               CAST(update_datetime AS CHAR) AS future_record_update_datetime
        FROM {FUTURE_TABLE}
        WHERE date IN ({placeholders}) AND trading_session = 'day'
    """
    spot_sql = f"""
        SELECT DATE_FORMAT(date, '%Y%m%d') AS Date,
               quote_code AS ValueCode,
               CAST(close_price AS DOUBLE) AS spot_close_price,
               CAST(update_datetime AS CHAR) AS spot_record_update_datetime
        FROM {SPOT_TABLE}
        WHERE date IN ({placeholders}) AND CHAR_LENGTH(quote_code) = 4
    """
    with engine.connect() as connection:
        future_rows = [dict(row) for row in connection.execute(text(future_sql), params).mappings()]
        spot_rows = [dict(row) for row in connection.execute(text(spot_sql), params).mappings()]
    future = pl.from_dicts(future_rows, infer_schema_length=None).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("future_close_price").cast(pl.Float64),
        pl.col("future_settlement_price").cast(pl.Float64),
        pl.col("future_record_update_datetime").cast(pl.String),
    )
    spot = pl.from_dicts(spot_rows, infer_schema_length=None).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("spot_close_price").cast(pl.Float64),
        pl.col("spot_record_update_datetime").cast(pl.String),
    )
    if future.select("Date", "QuoteCode").n_unique() != future.height:
        raise ValueError("futures daily close query returned duplicate date/contracts")
    if spot.select("Date", "ValueCode").n_unique() != spot.height:
        raise ValueError("spot daily close query returned duplicate date/products")
    joined = (
        marks.join(future, on=["Date", "QuoteCode"], how="left", validate="1:1")
        .join(spot, on=["Date", "ValueCode"], how="left", validate="1:1")
        .sort(["Date", "ValueCode", "QuoteCode"])
    )
    missing = joined.filter(
        pl.any_horizontal(
            pl.col("spot_close_price").is_null(),
            pl.col("future_close_price").is_null(),
            pl.col("spot_close_price") <= 0,
            pl.col("future_close_price") <= 0,
        )
    )
    if missing.height:
        raise ValueError(f"paired close coverage is incomplete: {missing.to_dicts()}")

    rows: list[dict[str, object]] = []
    for row in joined.iter_rows(named=True):
        identity_payload = {
            "Date": row["Date"],
            "ValueCode": row["ValueCode"],
            "QuoteCode": row["QuoteCode"],
            "spot_close_price": row["spot_close_price"],
            "future_close_price": row["future_close_price"],
            "spot_record_update_datetime": row["spot_record_update_datetime"],
            "future_record_update_datetime": row["future_record_update_datetime"],
            "spot_source_table": SPOT_TABLE,
            "future_source_table": FUTURE_TABLE,
            "spot_source_column": "close_price",
            "future_source_column": "close_price",
            "future_trading_session": "day",
        }
        rows.append(
            {
                **row,
                "spot_source_table": SPOT_TABLE,
                "future_source_table": FUTURE_TABLE,
                "spot_source_column": "close_price",
                "future_source_column": "close_price",
                "future_trading_session": "day",
                "future_settlement_price_used": False,
                "source_identity_sha256": _canonical_sha256(identity_payload),
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort(
        ["Date", "ValueCode", "QuoteCode"]
    )


def _artifact(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
        result.update(
            {
                "rows": frame.height,
                "columns": frame.width,
                "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
            }
        )
    return result


def publish(
    facts: pl.DataFrame,
    output: Path,
    *,
    upstream_root: Path,
    upstream_marker: Mapping[str, object],
) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        facts.write_parquet(stage / "daily_close_facts.parquet")
        query_spec = {
            "spot_source_table": SPOT_TABLE,
            "spot_value_column": "close_price",
            "future_source_table": FUTURE_TABLE,
            "future_value_column": "close_price",
            "future_filter": "trading_session = 'day'",
            "future_settlement_price_used": False,
            "identity_includes_database_update_datetime": True,
            "credentials_persisted": False,
        }
        (stage / "query_spec.json").write_text(
            json.dumps(query_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        dates = facts.group_by("Date").len().sort("Date")
        readme = [
            "# Expiry paired daily close facts",
            "",
            f"- {facts.height}/{facts.height} expiry product-contract pairs have both closes.",
            f"- Spot source: `{SPOT_TABLE}.close_price`.",
            f"- Futures source: `{FUTURE_TABLE}.close_price`, day session.",
            "- `settlement_price` is retained only as an audit field and is never used.",
            "- Each row identity binds the pair, prices, source columns, and both database update timestamps.",
            "",
            "| Date | Pairs |",
            "|---|---:|",
        ]
        for row in dates.iter_rows(named=True):
            readme.append(f"| {row['Date']} | {row['len']} |")
        (stage / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        inventory = {
            path.name: _artifact(path)
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        marker = {
            "complete": True,
            "schema_version": VERSION,
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
            "upstream_root": str(upstream_root.resolve()),
            "upstream_complete_sha256": _sha256(upstream_root / "complete.json"),
            "upstream_marker_payload_sha256": upstream_marker.get("marker_payload_sha256"),
            "pair_count": facts.height,
            "date_count": facts["Date"].n_unique(),
            "paired_close_coverage": facts.height,
            "future_settlement_price_used": False,
            "artifacts": inventory,
        }
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def run(upstream_root: Path, output: Path, *, host: str, user: str, password: str) -> pl.DataFrame:
    marks, marker = load_verified_expiry_marks(upstream_root)
    facts = query_daily_close_facts(
        marks, mysql_host=host, mysql_user=user, mysql_password=password
    )
    publish(facts, output, upstream_root=upstream_root, upstream_marker=marker)
    return facts


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mysql-host", default=os.getenv("MYSQL_HOST", "192.168.1.187"))
    parser.add_argument("--mysql-user", default=os.getenv("MYSQL_USER", "data.admin"))
    parser.add_argument("--mysql-password", default=os.getenv("MYSQL_PASSWORD"))
    args = parser.parse_args(argv)
    if not args.mysql_password:
        raise ValueError("MYSQL_PASSWORD or --mysql-password is required")
    facts = run(
        args.upstream_root,
        args.output,
        host=args.mysql_host,
        user=args.mysql_user,
        password=args.mysql_password,
    )
    print(facts.select("Date", "ValueCode", "QuoteCode", "spot_close_price", "future_close_price"))


if __name__ == "__main__":
    main()
