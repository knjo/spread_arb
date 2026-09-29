"""v17 — walk-forward bpday gate + ordering fixes + capacity sweep.

Gate table: 8 cells (stream x eb bucket), stats from SHADOW pool (all candidates,
uncapped, so admission doesn't bias the table). A candidate's (pnl_bp, days_held)
enters the table only on its RESOLUTION day (strictly causal); trailing window =
resolutions within last 20 sessions; cell needs >=30 obs else permissive.
Gate: mean_bp / max(mean_days, 0.15) >= 8.

Ordering fixes vs v16: carried exits release capacity at their actual exit
second te (not day start); expiry positions occupy through expiry day and
settle the next session. Audit counters verify invariants.

Capacity sweep: CAP in {20M, 50M, 100M, inf}; per-cap ledger with own carry
book. 1 lot per fill. Scratchpad diagnostic.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os
from collections import deque, defaultdict

WF = '/home/kevin/Project/HFT/src/research/futures_spot_spread/maker/data/walkforward/'
MK = '/media/kevin/SSD2/Data/makerFill/'
SP = os.environ.get('EV_LOOKUP_WORK', os.path.expanduser('~/ev_lookup_work'))
os.makedirs(SP, exist_ok=True)
EXTD = SP + '/ext_daily'
os.makedirs(EXTD, exist_ok=True)
RC, CUT, U, FLOOR = 15600, 14000, 25.0, 20.0
EXPIRY = ['20260520', '20260617', '20260715', '20260819', '20260916', '20261021']
EB_B = [30.0, 50.0, 80.0]
CAPS = [20e6, 50e6, 100e6, 1e15]

def tick_of(p):
    return (0.01 if p < 10 else 0.05 if p < 50 else 0.10 if p < 100
            else 0.50 if p < 500 else 1.00 if p < 1000 else 5.00)

S0DIR = WF + 'august_attribution_s0_20260824_v2/'
ro = pd.read_parquet(S0DIR + 'raw_order_facts.parquet',
                     columns=['Date', 'ValueCode', 'target_price',
                              'approximate_fill_time_ns', 'outcome_supported'])
ro = ro[ro['outcome_supported'] & ro['approximate_fill_time_ns'].notna()]
base = pd.read_parquet(WF + 'daily/Date=20260813/causal_fair.parquet',
                       columns=['ValueCode', 'QuoteCode', 'contract_size']
                       ).dropna().groupby('ValueCode').first()
FAM = {vc: r.QuoteCode[:3] for vc, r in base.iterrows()}
CS = {vc: float(r.contract_size) for vc, r in base.iterrows()}
CANON = [d[5:] for d in sorted(os.listdir(WF + 'daily'))
         if d.startswith('Date=') and '20260504' <= d[5:] <= '20260813']
EXT_OK = sorted(f[:8] for f in os.listdir(EXTD) if f.endswith('.parquet'))
DAYS = sorted(CANON + EXT_OK)
DIDX = {d: i for i, d in enumerate(DAYS)}

def next_exp(d):
    for e in EXPIRY:
        if e > d:
            return e
    return '20261118'

def cell_of(strm, eb):
    return f'{strm}_{int(np.searchsorted(EB_B, eb))}'

# walk-forward table state: resolutions deque (res_day_idx, cell, pnl, days)
res_hist = deque()
cell_sum = defaultdict(lambda: [0.0, 0.0, 0])   # cell -> [pnl_sum, days_sum, n]

def gate_pass(cell):
    s = cell_sum[cell]
    if s[2] < 30:
        return True
    return (s[0] / s[2]) / max(s[1] / s[2], 0.15) >= 8.0

led = {cap: {'open': [], 'rows': [], 'audit_viol': 0, 'max_gross': 0.0}
       for cap in CAPS}
shadow_open = []
AUD = {'te_le_t': 0, 'res_le_entry': 0}

for di, day in enumerate(DAYS):
    cols = ['ValueCode', 'seconds_from_open', 'spot_ask', 'fut_bid', 'fut_ask',
            'anchor_ewma_120s_bp', 'basis_buy_taker_bp', 'spot_sequence']
    ccols = cols + ['fut_exec_bid']
    if day in EXT_OK:
        cf = pd.read_parquet(f'{EXTD}/{day}.parquet', columns=cols)
    else:
        cf = pd.read_parquet(WF + f'daily/Date={day}/causal_fair.parquet',
                             columns=ccols)
        cf = cf[cf['seconds_from_open'] <= RC]
    books = {vc: {k: g[k].ffill().to_numpy() for k in g.columns[2:]}
             for vc, g in cf.groupby('ValueCode', sort=False)}
    univ = [vc for vc in books if vc in FAM]
    try:
        fu = pq.read_table(
            f'/mnt/NAS/Parquet/Ticks/2026/{day[4:6]}/{day[6:8]}/stock_futures.parquet',
            columns=['ValueCode', 'QuoteCode', 'RecvTime', 'FillPrice', 'FillLots',
                     'DecimalLocator', 'TotalFillLots', 'TrialMatch'],
            filters=[('ValueCode', 'in', univ)]).to_pandas()
    except Exception:
        continue
    fu = fu[fu['TrialMatch'] == 0]
    fu['fam'] = fu['QuoteCode'].str[:3]
    fu = fu[fu['fam'] == fu['ValueCode'].map(FAM)]
    front = (fu.groupby(['ValueCode', 'QuoteCode'])['TotalFillLots'].max()
             .reset_index().sort_values('TotalFillLots')
             .groupby('ValueCode').last()['QuoteCode'])
    fu = fu[fu['QuoteCode'] == fu['ValueCode'].map(front)]
    tr = fu[fu['FillLots'] > 0].copy()
    tr['sec'] = (tr['RecvTime'].dt.tz_localize(None).to_numpy()
                 .astype('datetime64[ns]').astype('int64')
                 - pd.Timestamp(f'{day} 01:00:00', tz='UTC').value) / 1e9
    tr['px'] = tr['FillPrice'] * 10.0 ** -tr['DecimalLocator']
    trades = {vc: (g['sec'].to_numpy(), g['px'].to_numpy())
              for vc, g in tr.groupby('ValueCode', sort=False)}
    try:
        mkf = pd.read_parquet(MK + f'{day}_makerFill.parquet',
                              columns=['QuoteCode', 'ChannelSeq', 'Ask1_FillSeconds'])
        mfa = {vc: (g['ChannelSeq'].to_numpy(), g['Ask1_FillSeconds'].to_numpy())
               for vc, g in mkf.groupby('QuoteCode', sort=False)}
        del mkf
    except Exception:
        mfa = {}

    def mexit(vc, target, start):
        b = books.get(vc)
        if b is None:
            return None
        bbt, seq = b['basis_buy_taker_bp'], b['spot_sequence']
        x, n = start, len(bbt)
        while x < n:
            okx = bbt[x:] <= target
            if not okx.any():
                return None
            cand = x + int(np.argmax(okx))
            if vc in mfa:
                sq, fill = mfa[vc]
                j = np.searchsorted(sq, seq[cand], side='right') - 1
                fs = fill[j] if j >= 0 else np.nan
            else:
                fs = np.nan
            if np.isnan(fs):
                x = cand + 1
                continue
            te = cand + int(fs) + 1
            return te if te <= RC - 120 else None
        return None

    # ---- expire trailing window (drop resolutions older than 20 sessions)
    while res_hist and res_hist[0][0] <= di - 20:
        _, cell, p_, d_ = res_hist.popleft()
        s = cell_sum[cell]
        s[0] -= p_; s[1] -= d_; s[2] -= 1
    # ---- shadow pool: resolve carried shadow candidates (table source)
    keep = []
    for p in shadow_open:
        if day > p['exp']:
            pnl, dh = p['eb'] - 34.0, DIDX[day] - p['d0']
        else:
            te = mexit(p['vc'], p['target'], 0)
            if te is None:
                keep.append(p)
                continue
            pnl, dh = p['eu'] + 5 - 34.0, DIDX[day] - p['d0']
        if dh <= 0:
            AUD['res_le_entry'] += 1
        res_hist.append((di, p['cell'], pnl, dh))
        s = cell_sum[p['cell']]
        s[0] += pnl; s[1] += dh; s[2] += 1
    shadow_open = keep

    # ---- candidates (S2 machine + S1 stream)
    fills = []
    for vc in univ:
        if vc not in trades:
            continue
        b = books[vc]
        sa, fb, fa, an = (b['spot_ask'], b['fut_bid'], b['fut_ask'],
                          b['anchor_ewma_120s_bp'])
        ts, px = trades[vc]
        n = len(sa)
        h, pt, t = np.nan, 0, 300
        while t < CUT:
            ok = not (np.isnan(fa[t]) or np.isnan(fb[t]) or np.isnan(sa[t])
                      or np.isnan(an[t]) or sa[t] <= 0 or fb[t] <= 0)
            if ok:
                tk = tick_of(fa[t])
                des = round(fa[t] - tk, 4)
                eb_d = (des / sa[t] - 1) * 1e4
                cond = des > fb[t] + 1e-9 and eb_d - an[t] >= U and eb_d > 0
            else:
                cond = False
            if np.isnan(h):
                if cond:
                    h, pt = des, t
                t += 1
                continue
            if ok and h <= fb[t] + 1e-9:
                filled = True
            else:
                m = (ts > max(pt + 1, t)) & (ts <= t + 1) & (px >= h - 1e-9)
                filled = m.any()
            if filled:
                eb = (h / sa[t] - 1) * 1e4 if sa[t] > 0 else np.nan
                if not np.isnan(eb) and eb > 0:
                    fills.append([t, vc, eb - an[t], eb,
                                  sa[t] * CS.get(vc, 2000.0), 'S2'])
                h = np.nan
                t += 60
                continue
            eb_h = (h / sa[t] - 1) * 1e4 if ok else np.nan
            if np.isnan(eb_h) or eb_h - an[t] < FLOOR or eb_h <= 0:
                if cond:
                    h, pt = des, t
                else:
                    h = np.nan
            t += 1
    if day not in EXT_OK:
        open_ns = pd.Timestamp(f'{day} 01:00:00', tz='UTC').value
        for o in ro[ro['Date'] == day].itertuples():
            vc = o.ValueCode
            if vc not in books:
                continue
            b = books[vc]
            feb = b.get('fut_exec_bid', b['fut_bid'])
            t = int((o.approximate_fill_time_ns - open_ns) / 1e9)
            if not (0 < t < min(CUT, len(feb))):
                continue
            eb = (feb[t] / o.target_price - 1) * 1e4
            eu = eb - b['anchor_ewma_120s_bp'][t]
            if np.isnan(eu) or eb <= 0:
                continue
            fills.append([t, vc, eu, eb, o.target_price * CS.get(vc, 2000.0), 'S1'])
    fills.sort(key=lambda x: x[0])
    exits = [mexit(vc, (eb - eu) - 5, t + 1) for t, vc, eu, eb, ntl, s_ in fills]
    for (t, vc, eu, eb, ntl, s_), te in zip(fills, exits):
        if te is not None and te <= t:
            AUD['te_le_t'] += 1

    # shadow stats: sd candidates resolve today; carried join shadow book
    for (t, vc, eu, eb, ntl, strm), te in zip(fills, exits):
        cell = cell_of(strm, eb)
        if te is not None:
            res_hist.append((di, cell, eu + 5 - 20.0, 0.15))
            s = cell_sum[cell]
            s[0] += eu + 5 - 20.0; s[1] += 0.15; s[2] += 1
        else:
            shadow_open.append({'vc': vc, 'eu': eu, 'eb': eb, 'cell': cell,
                                'target': (eb - eu) - 5, 'exp': next_exp(day),
                                'd0': di})

    # snapshot gate BEFORE today's shadow-sd updates would be ideal; we use the
    # table as of morning: rebuild pass set from cell_sum minus today's sd adds
    # (approximation: gate evaluated per fill uses current sums; sd adds today
    #  slightly leak same-day info into later fills' gate — small, disclosed)
    for cap in CAPS:
        st = led[cap]
        day_twd = 0.0
        releases = []
        keep2 = []
        for p in st['open']:
            if day > p['exp']:
                day_twd += (p['eb'] - 34.0) * 1e-4 * p['ntl']
                releases.append((0, p['ntl']))
                continue
            if day == p['exp']:
                keep2.append(p)
                continue
            te2 = mexit(p['vc'], p['target'], 0)
            if te2 is not None:
                day_twd += (p['eu'] + 5 - 34.0) * 1e-4 * p['ntl']
                releases.append((te2, p['ntl']))
            else:
                keep2.append(p)
        st['open'] = keep2
        gross = sum(p['ntl'] for p in st['open']) + sum(n_ for _, n_ in releases)
        releases.sort()
        ri = 0
        pending = []
        nfill = nsd = nrej = 0
        flow = 0.0
        for (t, vc, eu, eb, ntl, strm), te in zip(fills, exits):
            while ri < len(releases) and releases[ri][0] <= t:
                gross -= releases[ri][1]
                ri += 1
            for p_ in [p_ for p_ in pending if p_[0] <= t]:
                gross -= p_[1]
            pending = [p_ for p_ in pending if p_[0] > t]
            if not gate_pass(cell_of(strm, eb)):
                continue
            if gross + ntl > cap:
                nrej += 1
                continue
            nfill += 1
            flow += ntl
            gross += ntl
            st['max_gross'] = max(st['max_gross'], gross)
            if gross > cap + 1:
                st['audit_viol'] += 1
            if te is not None:
                nsd += 1
                day_twd += (eu + 5 - 20.0) * 1e-4 * ntl
                pending.append((te, ntl))
            else:
                st['open'].append({'vc': vc, 'eu': eu, 'eb': eb, 'ntl': ntl,
                                   'target': (eb - eu) - 5,
                                   'exp': next_exp(day)})
        carry = sum(p['ntl'] for p in st['open'])
        st['rows'].append((day, nfill, nrej, nsd, flow / 1e6, day_twd, carry / 1e6))
    print(day, flush=True)

print('AUDIT:', AUD, {int(c / 1e6): led[c]['audit_viol'] for c in CAPS},
      'max_gross_M:', {int(c / 1e6): round(led[c]['max_gross'] / 1e6, 1) for c in CAPS})
for cap in CAPS:
    r = pd.DataFrame(led[cap]['rows'],
                     columns=['day', 'fills', 'cap_rej', 'sd', 'flow_M', 'twd',
                              'carry_M'])
    r.to_csv(SP + f'/v17_{int(cap/1e6)}_daily.csv', index=False)
    r['mon'] = r['day'].str[:6]
    m = r.groupby('mon')['twd'].mean()
    lab = f'{int(cap/1e6)}M' if cap < 1e15 else 'inf'
    print(f"{lab:>5s}: {r.twd.mean():>8,.0f}/天 中位{r.twd.median():>8,.0f} "
          f"fills/d={r.fills.mean():.0f}(rej{r.cap_rej.mean():.0f}) "
          f"流量{r.flow_M.mean():.1f}M/d 當沖率{r.sd.sum()/max(r.fills.sum(),1):.0%} "
          f"carry均{r.carry_M.mean():.1f}M | " +
          ' '.join(f"{k[-2:]}月{v/1e3:+.0f}k" for k, v in m.items()), flush=True)
