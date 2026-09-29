"""Quote gates: book states in which a maker quote must not rest (found by the Stage 4 slippage screen).

A gate is a boolean time series per (leg, product). It is stored as its change points so the replay can ask
both "is the state good at t?" (may we place the quote) and "when does it next turn bad?" (an extra cancel
trigger, handled exactly like the residual floor: cancel latency applies, fills inside it are races).

    leg       hedge-side L1-L2 gap <= max ticks      extra condition
    S1_buy    futures bid                            spot B1_A1B1 = BidLots1/(BidLots1+AskLots1) < b1_a1b1_max
    S2_sell   spot ask                               spot A1_A1A5 = AskLots1/sum(AskLots1..5)   < a1_a1a5_max
    S1_sell   futures ask   (exit E1)
    S2_buy    spot bid      (exit E2)

    python -m spreadArb.src.backtest.gates --shard 0/4 [--gap 1 --b1a1b1 0.3 --a1a1a5 0.2]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..common.books import INF, BookSeries, load_session
from ..common.paths import CLOSE_SECOND, DATA_ROOT, SECOND, grid_days, open_ns, points_path
from ..points.s2 import tick_vec

LEGS = ("S1_buy", "S2_sell", "S1_sell", "S2_buy")


@dataclass(frozen=True)
class GateSpec:
    hedge_gap_ticks_max: int = 1
    s1_buy_b1_a1b1_max: float = 0.3
    s2_sell_a1_a1a5_max: float = 0.2

    @property
    def tag(self) -> str:
        return f"gap{self.hedge_gap_ticks_max}_b{self.s1_buy_b1_a1b1_max:g}_a{self.s2_sell_a1_a1a5_max:g}"


def gates_path(tag: str, day: str):
    return DATA_ROOT / "gates" / tag / f"Date={day}.parquet"


def gap_ticks(book: BookSeries, side: str) -> np.ndarray:
    """Ticks between the merged L1 and the next displayed level on that side (inf when there is none)."""
    if side == "bid":
        top = book.top_bid_px
        second = np.where((book.bid_qty > 0) & (book.bid_px < top[:, None]), book.bid_px, 0).max(axis=1)
        return np.where(second > 0, (top - second) / tick_vec(np.maximum(second, 1)), np.inf)
    top = book.top_ask_px
    second = np.where((book.ask_qty > 0) & (book.ask_px > top[:, None]), book.ask_px, INF).min(axis=1)
    return np.where(second < INF, (second - top) / tick_vec(np.maximum(top, 1)), np.inf)


def spot_shape(book: BookSeries) -> tuple[np.ndarray, np.ndarray]:
    """The user's tickFeature definitions on the raw five levels: B1_A1B1 (nan -> 0.5) and A1_A1A5."""
    b1, a1 = book.bid_qty[:, 0].astype(float), book.ask_qty[:, 0].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        b1_a1b1 = np.where(b1 + a1 > 0, b1 / (b1 + a1), 0.5)
        total = book.ask_qty.sum(axis=1).astype(float)
        a1_a1a5 = np.where(total > 0, a1 / total, np.nan)
    return b1_a1b1, a1_a1a5


def change_points(ns: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ns.size == 0:
        return ns, ok
    keep = np.r_[True, ok[1:] != ok[:-1]]
    return ns[keep], ok[keep]


def leg_series(spot: BookSeries, fut: BookSeries, spec: GateSpec) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    g = spec.hedge_gap_ticks_max
    b1_a1b1, a1_a1a5 = spot_shape(spot)
    s_bid_ok = spot.valid & (gap_ticks(spot, "bid") <= g)
    s_ask_ok = spot.valid & (gap_ticks(spot, "ask") <= g)
    f_bid_ok = fut.valid & (gap_ticks(fut, "bid") <= g)
    f_ask_ok = fut.valid & (gap_ticks(fut, "ask") <= g)
    out = {"S2_sell": change_points(spot.ns, s_ask_ok & (a1_a1a5 < spec.s2_sell_a1_a1a5_max)),
           "S2_buy": change_points(spot.ns, s_bid_ok),
           "S1_sell": change_points(fut.ns, f_ask_ok)}
    # S1_buy needs both books: evaluate on the union of their update times (forward-filled)
    t = np.union1d(spot.ns, fut.ns)
    i_s = np.searchsorted(spot.ns, t, side="right") - 1
    i_f = np.searchsorted(fut.ns, t, side="right") - 1
    shape_ok = spot.valid & (b1_a1b1 < spec.s1_buy_b1_a1b1_max)
    both = (i_s >= 0) & (i_f >= 0) & shape_ok[np.clip(i_s, 0, None)] & f_bid_ok[np.clip(i_f, 0, None)]
    out["S1_buy"] = change_points(t, both)
    return out


def build_day(day: str, spec: GateSpec) -> dict:
    started = time.time()
    pairs = pl.concat([pl.read_parquet(points_path(day, f"{n}_entries"), columns=["vc", "qc"]) for n in ("s1", "s2")]).unique()
    books, _, _, _ = load_session(day, pairs["vc"].to_list(), pairs["qc"].to_list())
    lo, hi = open_ns(day), open_ns(day) + CLOSE_SECOND * SECOND
    frames = []
    for vc, qc in pairs.iter_rows():
        s, f = books.get(f"S:{vc}"), books.get(f"F:{qc}")
        if s is None or f is None:
            continue
        for leg, (ns, ok) in leg_series(s, f, spec).items():
            m = (ns >= lo - 60 * SECOND) & (ns <= hi)
            ns2, ok2 = change_points(ns[m], ok[m])
            frames.append(pl.DataFrame({"leg": leg, "vc": vc, "ns": ns2, "ok": ok2}))
    out = pl.concat(frames).sort(["leg", "vc", "ns"])
    path = gates_path(spec.tag, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    return dict(day=day, rows=out.height, products=pairs.height, elapsed_s=round(time.time() - started, 1))


class GateBook:
    """One day's gates. Missing (leg, product) means the gate is never good (fail closed)."""

    def __init__(self, day: str, tag: str):
        frame = pl.read_parquet(gates_path(tag, day))
        self.series = {}
        for (leg, vc), g in frame.group_by(["leg", "vc"], maintain_order=True):
            self.series[(leg, vc)] = (g["ns"].to_numpy(), g["ok"].to_numpy())

    def ok_at(self, leg: str, vc: str, t: int) -> bool:
        s = self.series.get((leg, vc))
        if s is None:
            return False
        i = int(np.searchsorted(s[0], t, side="right")) - 1
        return bool(i >= 0 and s[1][i])

    def next_bad(self, leg: str, vc: str, t: int) -> int | None:
        """First time >= t at which the gate is bad; None if it stays good to the end of the day."""
        s = self.series.get((leg, vc))
        if s is None:
            return t
        i = int(np.searchsorted(s[0], t, side="right")) - 1
        if i < 0 or not s[1][i]:
            return t
        return int(s[0][i + 1]) if i + 1 < len(s[0]) else None      # change points alternate: the next one is bad

    def good_share(self, leg: str) -> float:
        tot = good = 0
        for (l, _), (ns, ok) in self.series.items():
            if l != leg or len(ns) < 2:
                continue
            d = np.diff(ns)
            tot += d.sum()
            good += d[ok[:-1]].sum()
        return good / tot if tot else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--b1a1b1", type=float, default=0.3)
    ap.add_argument("--a1a1a5", type=float, default=0.2)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    spec = GateSpec(args.gap, args.b1a1b1, args.a1a1a5)
    i, n = (int(x) for x in args.shard.split("/"))
    days = [d for d in grid_days() if (not args.start or d >= args.start) and (not args.end or d <= args.end)]
    for k, day in enumerate(days):
        if k % n != i or (gates_path(spec.tag, day).exists() and not args.force):
            continue
        try:
            print(json.dumps(build_day(day, spec)), flush=True)
        except Exception as exc:
            print(json.dumps(dict(day=day, error=repr(exc))), flush=True)


if __name__ == "__main__":
    main()
