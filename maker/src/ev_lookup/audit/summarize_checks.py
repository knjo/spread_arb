"""Save independent audit evidence and verify reference reproducibility."""
from __future__ import annotations

import hashlib
import json

import polars as pl

from audit_common import OUT, WF, SNAPSHOT, MAKER, EXT, days


def main() -> None:
    summary = json.loads((OUT / "allocation_summary.json").read_text())
    probes = json.loads((OUT / "execution_probes.json").read_text())
    source_manifest = json.loads((OUT / "source_manifest.json").read_text())
    for name, item in source_manifest.items():
        assert hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() == item["sha256"]
    for name in ("v18_reference_20", "v18_reference_50", "v18_reference_100", "v17_reference_20"):
        assert summary[name]["snapshot_comparison"]["matched"], name
        audit = summary[name]["capacity_audit"]
        assert audit["cap_violations"] == audit["negative_balance_events"] == audit["close_before_open"] == 0
    r = pl.read_parquet(OUT / "resolved_candidates.parquet")
    e = pl.read_parquet(OUT / "exit_details.parquet")
    trace = pl.read_parquet(OUT / "v18_reference_20_trace.parquet")
    accepted = trace.filter(pl.col("reason") == "ok").join(r, on="cid")
    exits = e.join(accepted.select("cid", "live_day", "ntl", "strm", "eb", "eu", "live_bp"), on="cid").filter(
        pl.col("day") == pl.col("live_day"))
    reuse = exits.group_by("day", "vc", "maker_sequence", "te").agg(pl.len()).filter(pl.col("len") > 1)
    reuse.sort("len", descending=True).write_csv(OUT / "reused_exit_labels.csv")
    tied = r.group_by("day0", "t0").agg(pl.len(), pl.col("strm").n_unique().alias("streams")).filter(pl.col("streams") > 1)
    sizes = pl.read_parquet(WF / "daily/Date=20260813/causal_fair.parquet", columns=["ValueCode", "contract_size"]).drop_nulls().unique().rename({"ValueCode": "vc"})
    prices = exits.join(sizes, on="vc", how="left").with_columns(pl.col("contract_size").fill_null(2000)).with_columns(
        (pl.col("ntl") / pl.col("contract_size")).alias("spot_entry"))
    prices = prices.with_columns(((pl.col("spot_exit_ask") - pl.col("future_ask_te")) / pl.col("spot_entry") * 1e4 +
        pl.col("eb") - (pl.col("eu") + 5)).alias("price_gap_bp"))
    prices.write_parquet(OUT / "exit_price_proxy_diagnostic.parquet")
    price_summary = prices.select(pl.len(), pl.col("price_gap_bp").mean().alias("mean_bp"),
        pl.col("price_gap_bp").median().alias("median_bp"),
        pl.col("price_gap_bp").quantile(0.05).alias("p05_bp"),
        pl.col("price_gap_bp").quantile(0.95).alias("p95_bp"),
        (pl.col("price_gap_bp") * pl.col("ntl") * 1e-4).sum().alias("total_gap_twd")).row(0, named=True)
    sample = pl.read_parquet(OUT / "raw_s2_20260520.parquet")
    mismatch = trace.filter((pl.col("day") == "20260520") & (pl.col("strm") == "S2") & (pl.col("reason") == "ok")).join(sample, on=["vc", "t"], suffix="_probe")
    wrong = mismatch.filter((pl.col("trigger") == "print") & (pl.col("model_contract") != pl.col("print_contract")))
    missing = wrong.filter(~pl.col("matching_contract_print"))
    missing.write_csv(OUT / "accepted_foreign_contract_prints_20260520.csv")
    expected_features = {
        "snapshot_reproduction": "v17 20M and v18 20M/50M/100M: 85 daily rows each, exact counts and money within 1e-5 TWD",
        "synthetic_execution_clock_advance_reproduced": probes["synthetic_ordering"],
        "accepted_S2_20260520": mismatch.height, "accepted_foreign_contract_prints_20260520": wrong.height,
        "accepted_without_quote_contract_print_20260520": missing.height,
        "same_second_cross_stream_groups": tied.height, "same_second_cross_stream_rows": tied["len"].sum(),
        "reused_exit_label_groups": reuse.height, "reused_exit_label_positions": reuse["len"].sum(),
        "max_positions_per_exit_label": reuse["len"].max(),
        "successful_exit_evaluations_with_skipped_future_nan_labels": e.filter(pl.col("te").is_not_null() & (pl.col("skipped_nan_touches") > 0)).height,
        "static_bpday_training_rows": r.filter(pl.col("day0") < "20260701").height,
        "static_bpday_training_outcomes_unavailable_at_Jul1": r.filter((pl.col("day0") < "20260701") & (pl.col("res_day") >= "20260701")).height,
        "candidate_dump_censored_rows_imputed_as_expiry": r.filter(pl.col("res_type") == "censored").height,
        "exit_price_diagnostic_not_corrected_backtest": price_summary,
        "source_files_changed_since_snapshot": [name for name, item in source_manifest.items()
            if hashlib.sha256((MAKER / "src/ev_lookup" / name).read_bytes()).hexdigest() != item["sha256"]],
    }
    assert missing.height == 68
    assert expected_features["static_bpday_training_outcomes_unavailable_at_Jul1"] == 484
    assert expected_features["candidate_dump_censored_rows_imputed_as_expiry"] == 74
    (OUT / "checks.json").write_text(json.dumps(expected_features, indent=2) + "\n")
    inventory = []
    for day in days():
        path = WF / f"daily/Date={day}/causal_fair.parquet"
        if not path.exists():
            path = EXT / f"{day}.parquet"
        for role, item in [("causal_grid", path), ("maker_fill", __import__("pathlib").Path(f"/media/kevin/SSD2/Data/makerFill/{day}_makerFill.parquet"))]:
            stat = item.stat()
            inventory.append({"day": day, "role": role, "path": str(item), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    pl.from_dicts(inventory).write_csv(OUT / "input_stat_inventory.csv")
    artifacts = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "audit_artifacts.json"}
    (OUT / "audit_artifacts.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    print(json.dumps(expected_features, indent=2))


if __name__ == "__main__":
    main()
