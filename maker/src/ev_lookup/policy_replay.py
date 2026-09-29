"""容量分配政策的秒級重放（在 candidate_dump.py 產出的候選池上）。

candidates CSV 每列 = 一筆候選 fill 的完整命運（與分配無關）：
  day0,t0,vc,strm,eu,eb,ntl,res_day,res_te,res_type,pnl_bp
因為部位彼此不互動，任何分配政策 = 按時序在容量下挑子集，幾秒重放完。
用途：比較 FCFS / 單流 / 深水閘門 / bpday 等政策，再把勝者丟回真模擬器驗證
（重放對容量釋放時點有近似，實測比真模擬器樂觀約 17%）。

用法：uv run python policy_replay.py <candidates.csv> [cap_twd]
"""
import sys
import numpy as np
import pandas as pd


def replay(c: pd.DataFrame, policy: str, cap: float, est: dict | None = None):
    days = sorted(c['day0'].unique())
    nd = len(days)
    didx = {d: i for i, d in enumerate(days)}
    c = c.sort_values(['day0', 't0']).reset_index(drop=True)
    d0 = c['day0'].map(didx).to_numpy()
    dr = c['res_day'].map(lambda d: didx.get(d, nd - 1)).to_numpy()
    gross = s2open = 0.0
    rel = []
    twd = np.zeros(nd)
    adm = 0
    for i, row in enumerate(c.itertuples()):
        d, t = d0[i], row.t0
        rel2 = []
        for (rd, rt, ntl, is2) in rel:
            if rd < d or (rd == d and rt <= t):
                gross -= ntl
                s2open -= ntl * is2
            else:
                rel2.append((rd, rt, ntl, is2))
        rel = rel2
        is2 = 1 if row.strm == 'S2' else 0
        if policy == 'S1' and is2:
            continue
        if policy == 'S2' and not is2:
            continue
        if policy == 'deep50' and is2 and row.eb < 50:
            continue
        if policy == 's2cap8' and is2 and s2open + row.ntl > 8e6:
            continue
        if policy == 'bpday':
            b = min(int(np.searchsorted([30, 50, 80], row.eb)), 3)
            e = (est or {}).get(f'{row.strm}_{b}')
            if e is None or e[0] / max(e[1], 0.15) < 8.0:
                continue
        if gross + row.ntl > cap:
            continue
        adm += 1
        gross += row.ntl
        s2open += row.ntl * is2
        twd[d0[i]] += row.pnl_bp * 1e-4 * row.ntl
        rt = row.res_te if row.res_te >= 0 else 16000
        rel.append((dr[i], rt, row.ntl, is2))
    return pd.Series(twd, index=days), adm / nd


if __name__ == '__main__':
    path = sys.argv[1]
    cap = float(sys.argv[2]) if len(sys.argv) > 2 else 20e6
    c = pd.read_csv(path, dtype={'day0': str, 'res_day': str})
    c['days_held'] = (pd.to_datetime(c['res_day']) - pd.to_datetime(c['day0'])).dt.days.clip(lower=0)
    tr = c[c['day0'] < '20260701'].copy()
    tr['ebb'] = np.searchsorted([30, 50, 80], tr['eb']).clip(0, 3)
    est = {f'{s}_{b}': (g['pnl_bp'].mean(), max(g['days_held'].mean(), 0.15))
           for (s, b), g in tr.groupby(['strm', 'ebb'])}
    for pol in ['fcfs', 'S1', 'S2', 'deep50', 's2cap8', 'bpday']:
        twd, adm = replay(c, pol, cap, est)
        mon = twd.groupby(lambda d: d[:6]).mean()
        print(f'{pol:7s}: {twd.mean():>8,.0f}/天 (年化 {twd.mean()*245/cap*100:>3.0f}%) '
              f'admits/日={adm:.0f} | ' + '  '.join(f'{k[-2:]}月{v/1e3:+.0f}k' for k, v in mon.items()))
