"""v15 dump: resolve every candidate uncapped; allocation policies replayed later.
CAND rows: day0,t0,vc,strm,eu,eb,ntl,res_day,res_te,res_type,pnl_bp

Phase 1: rebuild any missing ext_daily 1Hz grids (post-0813 days) from raw ticks.
Phase 2: day-by-day simulation with true cross-day carry:
  entry  = low-sensitivity fut Ask maker at A1-1 (EV floor U-5, crossed=fill,
           trade prints >= held price from the second after posting)
  hedge  = spot taker at that second's ask (locked eb must be > 0)
  exit   = same-day spot Ask1-queue maker at anchor(fill)-5;
           carried -> retry maker exit on later sessions (cost 34bp),
           contract expiry -> settle at basis-0 floor (eb-34)
  capital= 20M gross (open positions incl carry), 1 lot per fill
Writes daily CSV; prints monthly summary. Scratchpad diagnostic.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os, sys, traceback

WF = '/home/kevin/Project/HFT/src/research/futures_spot_spread/maker/data/walkforward/'
MK = '/media/kevin/SSD2/Data/makerFill/'
SP = os.environ.get('EV_LOOKUP_WORK', os.path.expanduser('~/ev_lookup_work'))
os.makedirs(SP, exist_ok=True)
EXTD = SP + '/ext_daily'
os.makedirs(EXTD, exist_ok=True)
os.makedirs(EXTD, exist_ok=True)
RC, CUT, U, FLOOR, CAP = 15600, 14000, 25.0, 20.0, 20e6
EXPIRY = ['20260520', '20260617', '20260715', '20260819', '20260916', '20261021']
EXT_DAYS = ['20260814', '20260817', '20260818', '20260819', '20260820', '20260821',
            '20260824', '20260825', '20260826', '20260827', '20260831',
            '20260901', '20260902']

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
UNIV0 = sorted(base.index)
FAM = {vc: r.QuoteCode[:3] for vc, r in base.iterrows()}
CS = {vc: float(r.contract_size) for vc, r in base.iterrows()}
ALPHA = 1 - np.exp(-1 / 120.0)

def build_ext(day):
    out = f'{EXTD}/{day}.parquet'
    if os.path.exists(out):
        return True
    try:
        pq.read_schema(f'/mnt/NAS/Parquet/Ticks/2026/{day[4:6]}/{day[6:8]}/stock_futures.parquet')
    except Exception:
        return False
    open_ns = pd.Timestamp(f'{day} 01:00:00').value
    sp = pq.read_table(f'/media/kevin/SSD2/Data/tickData/{day}_StockTick.parquet',
                       columns=['RecvTime', 'ValueCode', 'ChannelSeq', 'TrialMatch',
                                'BidPrice1', 'AskPrice1', 'BidLots1'],
                       filters=[('ValueCode', 'in', UNIV0)]).to_pandas()
    sp = sp[sp['TrialMatch'] == 0]
    sp['sec'] = (sp['RecvTime'].to_numpy().astype('datetime64[ns]').astype('int64')
                 - open_ns) // 10**9
    sp = sp[(sp['sec'] >= 0) & (sp['sec'] <= RC)]
    fu = pq.read_table(f'/mnt/NAS/Parquet/Ticks/2026/{day[4:6]}/{day[6:8]}/stock_futures.parquet',
                       columns=['RecvTime', 'ValueCode', 'QuoteCode', 'TrialMatch',
                                'BidPrice1', 'AskPrice1', 'DecimalLocator',
                                'TotalFillLots'],
                       filters=[('ValueCode', 'in', UNIV0)]).to_pandas()
    fu = fu[fu['TrialMatch'] == 0]
    fu['fam'] = fu['QuoteCode'].str[:3]
    fu = fu[fu['fam'] == fu['ValueCode'].map(FAM)]
    front = (fu.groupby(['ValueCode', 'QuoteCode'])['TotalFillLots'].max()
             .reset_index().sort_values('TotalFillLots')
             .groupby('ValueCode').last()['QuoteCode'])
    fu = fu[fu['QuoteCode'] == fu['ValueCode'].map(front)]
    fu['sec'] = (fu['RecvTime'].dt.tz_localize(None).to_numpy()
                 .astype('datetime64[ns]').astype('int64') - open_ns) // 10**9
    fu = fu[(fu['sec'] >= 0) & (fu['sec'] <= RC)]
    scale = 10.0 ** -fu['DecimalLocator']
    fu['fb'] = fu['BidPrice1'] * scale
    fu['fa'] = fu['AskPrice1'] * scale
    frames = []
    idx = np.arange(RC + 1)
    for vc in UNIV0:
        s = sp[sp['ValueCode'] == vc]
        f = fu[fu['ValueCode'] == vc]
        if len(s) < 100 or len(f) < 100:
            continue
        sg = s.groupby('sec').last().reindex(idx).ffill()
        fg = f.groupby('sec').last().reindex(idx).ffill()
        sb = sg['BidPrice1'].to_numpy().astype(float).copy()
        sa = sg['AskPrice1'].to_numpy().astype(float).copy()
        fb = fg['fb'].to_numpy().astype(float).copy()
        fa = fg['fa'].to_numpy().astype(float).copy()
        bad = (sb <= 0) | (sa <= 0) | (fb <= 0) | (fa <= 0)
        sb[bad] = np.nan; sa[bad] = np.nan; fb[bad] = np.nan; fa[bad] = np.nan
        bm = ((fb + fa) / (sb + sa) - 1) * 1e4
        anchor = np.roll(pd.Series(bm).ewm(alpha=ALPHA, ignore_na=True)
                         .mean().to_numpy(), 1)
        anchor[0] = np.nan
        frames.append(pd.DataFrame({
            'ValueCode': vc, 'seconds_from_open': idx, 'spot_ask': sa,
            'fut_bid': fb, 'fut_ask': fa, 'basis_mid_bp': bm,
            'anchor_ewma_120s_bp': anchor,
            'basis_buy_taker_bp': (fa / sb - 1) * 1e4,
            'spot_sequence': sg['ChannelSeq'].to_numpy(),
            'contract_size': CS[vc]}))
    pd.concat(frames).to_parquet(out, index=False)
    return True

CANON = [d[5:] for d in sorted(os.listdir(WF + 'daily'))
         if d.startswith('Date=') and '20260504' <= d[5:] <= '20260813']
EXT_OK = [d for d in EXT_DAYS if build_ext(d)]
print('ext ready:', len(EXT_OK), flush=True)
DAYS = sorted(CANON + EXT_OK)

def next_exp(d):
    for e in EXPIRY:
        if e > d:
            return e
    return '20261118'

open_pos, rows, CAND = [], [], []
for di, day in enumerate(DAYS):
    try:
        cols = ['ValueCode', 'seconds_from_open', 'spot_ask', 'fut_bid', 'fut_ask',
                'anchor_ewma_120s_bp', 'basis_buy_taker_bp', 'spot_sequence',
                'fut_exec_bid'] if day not in EXT_OK else ['ValueCode', 'seconds_from_open', 'spot_ask', 'fut_bid', 'fut_ask', 'anchor_ewma_120s_bp', 'basis_buy_taker_bp', 'spot_sequence']
        if day in EXT_OK:
            cf = pd.read_parquet(f'{EXTD}/{day}.parquet', columns=cols)
        else:
            cf = pd.read_parquet(WF + f'daily/Date={day}/causal_fair.parquet',
                                 columns=cols)
            cf = cf[cf['seconds_from_open'] <= RC]
        books = {vc: {k: g[k].ffill().to_numpy() for k in cols[2:]}
                 for vc, g in cf.groupby('ValueCode', sort=False)}
        univ = [vc for vc in books if vc in FAM]
        try:
            fu = pq.read_table(
                f'/mnt/NAS/Parquet/Ticks/2026/{day[4:6]}/{day[6:8]}/stock_futures.parquet',
                columns=['ValueCode', 'QuoteCode', 'RecvTime', 'FillPrice',
                         'FillLots', 'DecimalLocator', 'TotalFillLots', 'TrialMatch'],
                filters=[('ValueCode', 'in', univ)]).to_pandas()
        except Exception:
            rows.append((day, 0, 0, 0, 0.0, 0.0, sum(p['ntl'] for p in open_pos) / 1e6, 'no_fut'))
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
                                  columns=['QuoteCode', 'ChannelSeq',
                                           'Ask1_FillSeconds'])
            mfa = {vc: (g['ChannelSeq'].to_numpy(),
                        g['Ask1_FillSeconds'].to_numpy())
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

        day_twd = 0.0
        keep = []
        for p in open_pos:
            if day >= p['exp']:
                CAND.append([p['day0'], p['t0'], p['vc'], p['strm'], p['eu'],
                             p['eb'], p['ntl'], day, -1, 'expiry', p['eb'] - 34.0])
                continue
            te = mexit(p['vc'], p['target'], 0)
            if te is not None:
                CAND.append([p['day0'], p['t0'], p['vc'], p['strm'], p['eu'],
                             p['eb'], p['ntl'], day, te, 'maker34',
                             p['eu'] + 5 - 34.0])
            else:
                keep.append(p)
        open_pos = keep
        gross = 0.0

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
                        fills.append([t, vc, eb - an[t], eb, sa[t] * CS.get(vc, 2000.0)])
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
        # ---- S1 stream (spot bid maker entries)
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
        else:
            for vc in univ:
                b = books[vc]
                pass  # ext-day S1 synth omitted (S1 weight small post-0813)
        for f in fills:
            if len(f) == 5:
                f.append('S2')
        fills.sort(key=lambda x: x[0])
        pending = []
        nfill = nsd = nrej = 0
        flow = 0.0
        for t, vc, eu, eb, ntl, strm in fills:
            te = mexit(vc, (eb - eu) - 5, t + 1)
            nfill += 1
            if te is not None:
                nsd += 1
                CAND.append([day, t, vc, strm, eu, eb, ntl, day, te,
                             'sd', eu + 5 - 20.0])
            else:
                open_pos.append({'vc': vc, 'eu': eu, 'eb': eb, 'ntl': ntl,
                                 'target': (eb - eu) - 5, 'exp': next_exp(day),
                                 'strm': strm, 'day0': day, 't0': t})
        carry = sum(p['ntl'] for p in open_pos)
        rows.append((day, nfill, nrej, nsd, flow / 1e6, day_twd, carry / 1e6, f"{s_twd['S1']:.0f}|{s_twd['S2']:.0f}"))
        print(day, nfill, round(day_twd), flush=True)
    except Exception:
        traceback.print_exc()
        rows.append((day, 0, 0, 0, 0.0, 0.0, sum(p['ntl'] for p in open_pos) / 1e6, 'err'))

for p in open_pos:
    CAND.append([p['day0'], p['t0'], p['vc'], p['strm'], p['eu'], p['eb'],
                 p['ntl'], '20260902', -1, 'censored', p['eb'] - 34.0])
c = pd.DataFrame(CAND, columns=['day0', 't0', 'vc', 'strm', 'eu', 'eb', 'ntl',
                                'res_day', 'res_te', 'res_type', 'pnl_bp'])
c.to_csv(SP + '/v15_candidates.csv', index=False)
print('candidates:', len(c))
print(c.groupby(['strm', 'res_type']).size().to_string())
