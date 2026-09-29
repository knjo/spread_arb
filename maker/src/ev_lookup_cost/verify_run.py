"""Independent checks of saved v19 decisions, fills, ledger and lookup snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from ..common.paths import futures_raw_path, spot_tick_path
from .causal_lookup import SECOND, open_ns
from .ev_rules import cell_of


def verify(output: Path, raw_days: set[str]) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise AssertionError("run is not complete")
    # Source hashes bind the saved run to its actual implementation, even if
    # the working tree changes later. Added verification files are not inputs.
    for name, digest in manifest["sources"].items():
        if hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() != digest:
            raise AssertionError("source snapshot mismatch")
    snapshots = json.loads((output / "snapshots.json").read_text())
    shadow_closed = []
    checks = dict(days=len(manifest["days"]), checked_portfolio_days=0, fills=0,
                  adverse_pairs=0, raw_prints_checked=0, expiry_positions=0,
                  final_unhedged=manifest["unhedged_final"])
    for day in manifest["days"]:
        for folder in sorted((output / f"Date={day}").iterdir()):
            if not folder.is_dir():
                continue
            actor = folder.name
            checks["checked_portfolio_days"] += 1
            pos_path = folder / "positions.parquet"
            positions = pl.read_parquet(pos_path) if pos_path.exists() else pl.DataFrame()
            trace_path = folder / "execution.parquet"
            trace = pl.read_parquet(trace_path) if trace_path.exists() else pl.DataFrame()
            dec_path = folder / "decisions.parquet"
            decisions = pl.read_parquet(dec_path) if dec_path.exists() else pl.DataFrame()
            ledger_path = folder / "ledger.parquet"
            if ledger_path.exists():
                ledger = pl.read_parquet(ledger_path)
                if not ledger["ns"].is_sorted():
                    raise AssertionError("ledger clock moved backwards")
                deltas = ledger["committed_cents"].diff() - ledger["delta_cents"]
                if deltas.drop_nulls().abs().max() not in (None, 0):
                    raise AssertionError("ledger deltas do not reconcile")
                if actor != "shadow":
                    cap = int(actor.rsplit("_", 1)[1][:-1]) * 1_000_000 * 100
                    if ledger["committed_cents"].max() > cap or ledger["committed_cents"].min() < 0:
                        raise AssertionError("hard cap breached")
            if decisions.height:
                admitted = decisions.filter(pl.col("admit"))
                if admitted.filter(pl.col("quote_ab") <= 0).height:
                    raise AssertionError("nonpositive basis admitted before quote")
                if decisions.filter((pl.col("ns") < open_ns(day)) |
                                     (pl.col("ns") >= open_ns(day) + 15600 * SECOND)).height:
                    raise AssertionError("decision used wrong session")
            if not positions.height:
                continue
            for p in positions.iter_rows(named=True):
                if p["actual_ab"] is not None and p["actual_ab"] <= 0 and p["entry_day"] == day:
                    checks["adverse_pairs"] += 1
                if p["hedged_ns"] is not None and p["hedged_ns"] < p["entry_fill_ns"]:
                    raise AssertionError("hedge precedes maker execution")
                if p["state"] == "closed":
                    if p["spot_buy_qty"] != p["spot_sell_qty"] or p["future_sell_qty"] != p["future_buy_qty"]:
                        raise AssertionError("closed position retains exposure")
                    gross = (p["spot_sell_cash"] - p["spot_buy_cash"] +
                             p["future_sell_cash"] - p["future_buy_cash"]) / 10_000
                    fees = p["spot_buy_cash"] / 1e8 * (20 if p["entry_day"] == day else 34)
                    if abs(gross - fees - p["pnl_twd"]) > 1e-6:
                        raise AssertionError("PnL does not match four actual legs")
                    releases = ledger.filter((pl.col("id") == p["id"]) & (pl.col("kind") == "release"))
                    if releases.height != 1 or releases.item(0, "ns") != p["close_ns"]:
                        raise AssertionError("capacity released before paired close")
                    if actor == "shadow":
                        shadow_closed.append(p)
            if not trace.height:
                continue
            checks["fills"] += trace.filter(pl.col("kind") == "maker_fill").height
            taker = trace.filter(pl.col("kind") == "taker_fill")
            if taker.height and taker.filter(pl.col("book_ns") > pl.col("ns")).height:
                raise AssertionError("hedge used future book")
            expiry = trace.filter(pl.col("kind") == "expiry_basis_zero_accounting")
            checks["expiry_positions"] += expiry.height
            for r in expiry.iter_rows(named=True):
                p = positions.filter(pl.col("id") == r["position_id"]).row(0, named=True)
                if not p["contract"]["expiry"] < day or r["ns"] != open_ns(day):
                    raise AssertionError("incorrect exact-contract settlement date")
            if day in raw_days:
                checks["raw_prints_checked"] += verify_raw_fills(day, trace)
        # Reconstruct each morning cell from shadow positions whose BOTH legs
        # closed before this day. Current-day resolutions must have no effect.
        for snap in [s for s in snapshots if s["day"] == day]:
            expected = {}
            for p in shadow_closed:
                if p["close_day"] not in snap["train_days"] or p["close_ns"] >= snap["cutoff_ns"]:
                    continue
                cell = cell_of(p["stream"], p["quote_ab"])
                s = expected.setdefault(cell, [0.0, 0.0, 0])
                s[0] += p["pnl_bp"]
                s[1] += max(manifest["days"].index(p["close_day"]) - p["entry_session"], .15)
                s[2] += 1
            if set(expected) != set(snap["cells"]):
                raise AssertionError("snapshot cell keys differ from past resolutions")
            for cell, values in expected.items():
                if any(abs(x - y) > 1e-6 for x, y in zip(values, snap["cells"][cell])):
                    raise AssertionError("snapshot includes unavailable outcomes")
    checks["passed"] = True
    (output / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    return checks


def verify_raw_fills(day: str, trace: pl.DataFrame) -> int:
    fills = trace.filter(pl.col("kind") == "maker_fill").with_columns(
        ((pl.col("stream") == "S2") & (pl.col("purpose") == "entry")).alias("future"))
    count = 0
    for future in (False, True):
        own = fills.filter(pl.col("future") == future).with_columns(
            pl.col("qc" if future else "vc").alias("instrument"))
        if own.is_empty():
            continue
        source = pl.scan_parquet(futures_raw_path(day) if future else spot_tick_path(day))
        scale = 10.0 ** -pl.col("DecimalLocator") if future else pl.lit(1.)
        raw = (source.filter(pl.col("QuoteCode").is_in(own["instrument"].unique().to_list()))
               .select(pl.col("RecvTime").dt.epoch("ns").alias("ns"),
                       pl.col("ChannelSeq").cast(pl.Int64).alias("trade_sequence"),
                       pl.col("QuoteCode").alias("instrument"),
                       (pl.col("FillLots") * (1 if future else 1000)).alias("raw_qty"),
                       (pl.col("FillPrice") * scale).alias("raw_price"), "TrialMatch")
               .filter(pl.col("ns").is_in(own["ns"].unique().to_list())).collect())
        keys = ["instrument", "ns", "trade_sequence"]
        used = own.group_by(keys).agg(pl.col("quantity").sum()).join(raw, on=keys, how="left", validate="1:1")
        if used.filter(pl.col("raw_qty").is_null() | (pl.col("quantity") > pl.col("raw_qty")) |
                       (pl.col("TrialMatch") != 0)).height:
            raise AssertionError("own fills exceed formal raw trade quantity")
        matched = own.join(raw, on=keys, how="left", validate="m:1")
        is_buy = (pl.col("stream") == "S1") & (pl.col("purpose") == "entry")
        if matched.filter(pl.when(is_buy).then(pl.col("raw_price") > pl.col("price") + 1e-8)
                           .otherwise(pl.col("raw_price") < pl.col("price") - 1e-8)).height:
            raise AssertionError("maker limit not reached by its own-contract print")
        count += used.height
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-days", nargs="*", default=[])
    args = parser.parse_args()
    print(json.dumps(verify(args.output, set(args.raw_days)), indent=2))


if __name__ == "__main__":
    main()
