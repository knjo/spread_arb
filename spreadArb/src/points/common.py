"""Shared pieces for point tables: first-passage queries, gate scans, hedge sweeps."""
from __future__ import annotations

import numpy as np

from ..common.books import BookSeries

BLOCK = 64


class FirstPassage:
    """Answer 'first index >= i0 whose value exceeds thr' with block maxima; nan/inf never exceed."""

    def __init__(self, values: np.ndarray):
        v = np.asarray(values, dtype=float)
        self.values = np.where(np.isfinite(v), v, -np.inf)
        n = len(self.values)
        pad = (-n) % BLOCK
        self.bmax = np.concatenate([self.values, np.full(pad, -np.inf)]).reshape(-1, BLOCK).max(axis=1)

    def first(self, i0: int, thr: float) -> int:
        values, n = self.values, len(self.values)
        if i0 >= n:
            return -1
        b = i0 // BLOCK
        seg = values[i0:min((b + 1) * BLOCK, n)]
        hit = np.flatnonzero(seg > thr)
        if hit.size:
            return i0 + int(hit[0])
        later = np.flatnonzero(self.bmax[b + 1:] > thr)
        if later.size == 0:
            return -1
        bb = b + 1 + int(later[0])
        seg = values[bb * BLOCK:min((bb + 1) * BLOCK, n)]
        return bb * BLOCK + int(np.flatnonzero(seg > thr)[0])


def next_false(ok: np.ndarray) -> np.ndarray:
    """next_false[i] = first index >= i where ok is False; len(ok) if none."""
    n = len(ok)
    out = np.full(n + 1, n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        out[i] = i if not ok[i] else out[i + 1]
    return out


def sweep_hedge(book: BookSeries, side: str, qty: int, t_fill: int, delay_ns: int, timeout_ns: int,
                lo: int, hi: int, end_ns: int):
    """Take the full quantity from L1-L5 from t_fill + delay, retrying on later books.

    Returns (hedge_ns, cash, levels_swept, timed_out) or None when no book ever has the depth.
    """
    th = t_fill + delay_ns
    i = book.index_at(th)
    timed_out = False
    while 0 <= i < len(book.ns):
        if book.ns[i] > th:
            th = int(book.ns[i])
        if th > end_ns:
            break
        if th - t_fill > timeout_ns:
            timed_out = True
        if book.valid[i]:
            taken = book.take(i, side, qty, lo, hi)
            if taken is not None:
                return th, taken[0], taken[1], timed_out
        i += 1
    return None
