"""Reconcile spreadArb S2 points (reference walker) against maker's four-session S2 supply study.

    uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.points.compare_maker_s2 --day 20260706
"""
from __future__ import annotations

import argparse

import polars as pl

from ..common.paths import MAKER_ROOT, points_path
from .s2 import sequential_fills

MAKER_FILLS = MAKER_ROOT / "data/s2_entry_review_20260914/four_sessions/report/fills.parquet"
POLICY = "event0_depth5_r25"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", required=True)
    args = parser.parse_args()
    ours = pl.read_parquet(points_path(args.day, "s2_entries"))
    walk = sequential_fills(ours)
    maker = pl.read_parquet(MAKER_FILLS).filter((pl.col("policy") == POLICY) & (pl.col("day") == args.day))
    with pl.Config(tbl_rows=40, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(f"day {args.day}: candidate rows {ours.height} ({ours['vc'].n_unique()} products), rows with fill "
              f"{ours.filter(pl.col('t_fill_ns').is_not_null()).height}")
        print(f"walker fills {walk.height} ({walk['vc'].n_unique()} products) vs maker {maker.height} ({maker['vc'].n_unique()} products)")
        if walk.height:
            print(f"  ours : decay mean {float((walk['quote_ab'] - walk['actual_ab']).mean()):.2f} bp, "
                  f"actual_ab mean {float(walk['actual_ab'].mean()):.2f}, within 50 ms {float(walk['within_50ms'].mean()):.3f}, "
                  f"cancel race {int(walk['cancel_race'].sum())}, negative basis {int((walk['actual_ab'] <= 0).sum())}")
        print(f"  maker: decay mean {float(maker['entry_decay_bp'].mean()):.2f} bp, actual_ab mean {float(maker['actual_ab'].mean()):.2f}, "
              f"within 50 ms {float((maker['quote_to_fill_ms'] <= 50).mean()):.3f}, cancel race {int(maker['cancellation_race'].sum())}, "
              f"negative basis {int((maker['actual_ab'] <= 0).sum())}")
        o = walk.group_by("vc").len().rename({"len": "ours"}) if walk.height else pl.DataFrame({"vc": [], "ours": []})
        m = maker.group_by("vc").len().rename({"len": "maker"})
        both = m.join(o, on="vc", how="full", coalesce=True).fill_null(0).with_columns((pl.col("ours").cast(pl.Int64) - pl.col("maker").cast(pl.Int64)).alias("diff"))
        print("\nper-product fills (top by maker):")
        print(both.sort(["maker", "ours"], descending=True).head(25))
        print("\nproducts where we have fills but maker has none:", both.filter(pl.col("maker") == 0)["vc"].to_list()[:30])
        print("products where maker has fills but we have none:", both.filter(pl.col("ours") == 0)["vc"].to_list()[:30])
        print("\nquote-time distribution of our walker fills (hour buckets):")
        if walk.height:
            from ..common.paths import open_ns, SECOND
            start = open_ns(args.day)
            print(walk.with_columns(((pl.col("quote_ns") - start) // (3600 * SECOND)).alias("hour")).group_by("hour").len().sort("hour"))


if __name__ == "__main__":
    main()
