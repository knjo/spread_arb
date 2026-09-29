"""稽核第 2 項量化：掛單占額度（pending reservation）對 S2 廣度與損益的影響。

四個變體（20M、85 日、其餘同 v18）：
  v18repro   : 成交時查容量；負 basis 成交刪除（重現 v18）
  nodelete   : 成交時查容量；所有成交入帳（含 eb<=0）
  reserveF   : 掛單時保留名目額度（gross + working <= cap），同秒競爭 FCFS；成交必入帳
  reserveE   : 同上，但同秒競爭按掛單時 est 高者優先
  reserveP   : 掛單只保留 25% 名目（≈成交機率），成交時若超帽 → 立即 taker 沖銷
               （成本 = 期貨 spread + 5bp），不建部位。實務最可能的運作模式。
S2 的 est / 格閘門在「掛單時」用掛單時的 eb_if 評（不再是成交後）。
查表以開盤凍結快照（修正稽核第 1 項的表洩漏）；eovn 隨實際出場秒遞減。
S2 成交時間記在 print 時戳所在秒之後的第一格（不再前移）。
S1 沿用 S0 真實 fill 流（成交時查 gross + working）。
輸出：每變體每日 CSV 與摘要。EV_LOOKUP_WORK 為工作目錄（需已有 ext_daily 快取）。
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os, sys, zlib
from collections import deque, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_rules import est_short, cell_of, cell_gate, shadow_price

WF = '/home/kevin/Project/HFT/src/research/futures_spot_spread/maker/data/walkforward/'
MK = '/media/kevin/SSD2/Data/makerFill/'
SP = os.environ.get('EV_LOOKUP_WORK', os.path.expanduser('~/ev_lookup_work'))
EXTD = SP + '/ext_daily'
RC, CUT, U, FLOOR, CAP = 15600, 14000, 25.0, 20.0, 20e6
EXPIRY = ['20260520', '20260617', '20260715', '20260819', '20260916', '20261021']
TODB = np.array([0, 3600, 9000, 1e9])
VARIANTS = os.environ.get('EV_VARIANTS', 'v18repro,nodelete,reserveF,reserveE,reserveP,cancelfull,glide30,softfull').split(',')
RES_FRAC = 0.25   # reserveP: 掛單保留名目的比例;超額成交立即 taker 沖銷
LAT = float(os.environ.get('EV_LAT', '0.10'))
# cancelfull/glide/softfull: 同秒 race 成交的條件機率 = P(gap<Δ)/P(gap<1s)。
# 實測(7 日 1,639 個相鄰成交間隔): Δ=20ms→0.046, 50ms→0.071, 100ms→0.129, 200ms→0.26。
# 0.10 是 09-08 之前未驗證的猜測值,保留為預設以利對照。

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
if os.environ.get('EV_DAYS'):
    DAYS = [d for d in DAYS if d in os.environ['EV_DAYS'].split(',')]
DIDX = {d: i for i, d in enumerate(DAYS)}

def next_exp(d):
    for e in EXPIRY:
        if e > d:
            return e
    return '20261118'

# ---- walk-forward shadow tables (shared; frozen at open)
res_hist = deque()
cell_sum = defaultdict(lambda: [0.0, 0.0, 0])
psd_cnt = defaultdict(lambda: [0, 0])
shadow_open = []

def tod(t):
    return int(np.searchsorted(TODB, t, side='right') - 1)

def make_state():
    return {'open': [], 'rows': [], 'rej': deque(maxlen=5), 'rej_today': 0.0}

ST = {v: make_state() for v in VARIANTS}

for di, day in enumerate(DAYS):
    cols = ['ValueCode', 'seconds_from_open', 'spot_ask', 'fut_bid', 'fut_ask',
            'anchor_ewma_120s_bp', 'basis_buy_taker_bp', 'spot_sequence']
    if day in EXT_OK:
        cf = pd.read_parquet(f'{EXTD}/{day}.parquet', columns=cols)
    else:
        cf = pd.read_parquet(WF + f'daily/Date={day}/causal_fair.parquet',
                             columns=cols + ['fut_exec_bid'])
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

    # ---- freeze tables at open (before any today update)
    while res_hist and res_hist[0][0] <= di - 20:
        _, cell, p_, d_ = res_hist.popleft()
        s = cell_sum[cell]; s[0] -= p_; s[1] -= d_; s[2] -= 1
    cell_frozen = {k: list(v) for k, v in cell_sum.items()}
    psd_frozen = {k: (v[1] / v[0] if v[0] >= 200 else 0.68) for k, v in psd_cnt.items()}
    def p_sd(strm, t):
        return psd_frozen.get((strm, tod(t)), 0.68)
    def gate(cell):
        s = cell_frozen.get(cell, [0.0, 0.0, 0])
        return cell_gate(s[0], s[1], s[2])

    # ---- per-product precompute for S2 quote machine
    prod = {}
    for vc in univ:
        if vc not in trades:
            continue
        b = books[vc]
        sa, fb, fa, an = b['spot_ask'], b['fut_bid'], b['fut_ask'], b['anchor_ewma_120s_bp']
        n = len(sa)
        tk = np.array([tick_of(x) if not np.isnan(x) else np.nan for x in fa])
        des = np.round(fa - tk, 4)
        with np.errstate(divide='ignore', invalid='ignore'):
            eb_d = (des / sa - 1) * 1e4
        ok = ~(np.isnan(fa) | np.isnan(fb) | np.isnan(sa) | np.isnan(an)) & (sa > 0) & (fb > 0)
        cond = ok & (des > fb + 1e-9) & (eb_d - an >= U) & (eb_d > 0)
        cond[:300] = False; cond[CUT:] = False
        prod[vc] = dict(sa=sa, fb=fb, fa=fa, an=an, des=des, eb=eb_d, ok=ok, cond=cond,
                        ts=trades[vc][0], px=trades[vc][1], ntl=CS.get(vc, 2000.0), n=n)

    # ---- S1 fills (S0 stream)
    s1 = []
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
            s1.append((t, vc, eu, eb, o.target_price * CS.get(vc, 2000.0)))
    s1.sort()

    # ---- per-variant simulation (second-stepped for S2 quotes)
    shadow_add = []   # (t, strm, eu, eb, ntl, te)  from unconstrained S2 (variant v18repro) + S1
    def run_variant(V):
        st = ST[V]
        reserve = V.startswith('reserve')
        soft = V.startswith('softfull')          # softfull / softfull_nn(刪負basis成交) / softfull_fg(成交時再過EV閘)
        cancelfull = V in ('cancelfull', 'glide30') or soft
        frac = RES_FRAC if V == 'reserveP' else 1.0
        n_unwind = 0; n_cross = 0; full_since = -1
        def capfn(t):
            if V != 'glide30':
                return CAP
            return 30e6 if t < 3600 else max(CAP, 30e6 - 10e6 * (t - 3600) / (14400 - 3600))
        delete_neg = V in ('v18repro', 'softfull_nn')
        lam = shadow_price(list(st['rej']), CAP)
        day_twd = 0.0
        # carried exits -> release events at te
        releases, keep2 = [], []
        for p in st['open']:
            if day > p['exp']:
                day_twd += (p['eb'] - 34.0) * 1e-4 * p['ntl']; releases.append((0, p['ntl'])); continue
            if day == p['exp']:
                keep2.append(p); continue
            te2 = mexit(p['vc'], p['target'], 0)
            if te2 is not None:
                day_twd += (p['eu'] + 5 - 34.0) * 1e-4 * p['ntl']; releases.append((te2, p['ntl']))
            else:
                keep2.append(p)
        st['open'] = keep2
        gross = sum(p['ntl'] for p in st['open']) + sum(n_ for _, n_ in releases)
        eovn = gross
        releases.sort(); ri = 0
        pending = []           # same-day exits (te, ntl)
        working = {}           # vc -> dict(h, pt, ntl, eu_q, eb_q)
        s1i = 0
        nfill = {'S1': 0, 'S2': 0}; nsd = 0; nrej = 0; nq_skip = 0; nq_post = 0; neg_booked = 0
        flow = 0.0
        def release_upto(t):
            nonlocal gross, eovn, ri, pending
            while ri < len(releases) and releases[ri][0] <= t:
                gross -= releases[ri][1]; eovn -= releases[ri][1]; ri += 1
            done = [p_ for p_ in pending if p_[0] <= t]
            for p_ in done:
                gross -= p_[1]
            pending = [p_ for p_ in pending if p_[0] > t]
        def book(t, vc, eu, eb, ntl, strm):
            nonlocal gross, eovn, day_twd, nsd, flow
            te = mexit(vc, (eb - eu) - 5, t + 1)
            nfill[strm] += 1; flow += ntl; gross += ntl
            if te is not None:
                nsd += 1; day_twd += (eu + 5 - 20.0) * 1e-4 * ntl; pending.append((te, ntl))
            else:
                eovn += ntl
                st['open'].append({'vc': vc, 'eu': eu, 'eb': eb, 'ntl': ntl,
                                   'target': (eb - eu) - 5, 'exp': next_exp(day), 'day': day})
            return te
        for t in range(300, CUT):
            release_upto(t)
            # S1 fills at this second
            while s1i < len(s1) and s1[s1i][0] == t:
                _, vc, eu, eb, ntl = s1[s1i]; s1i += 1
                working_ntl = sum(w['ntl'] for w in working.values()) * frac if reserve else 0.0
                est = est_short(eu, eb, p_sd('S1', t), slot_free=(eovn + ntl <= CAP))
                if est < lam * 0.6 or not gate(cell_of('S1', eb)):
                    continue
                if cancelfull and gross + ntl > capfn(t):
                    if full_since >= 0 and full_since < t:
                        nrej += 1; continue
                    if soft:
                        if np.random.default_rng(int(t) * 7919 + zlib.crc32(vc.encode()) % 1000).random() > LAT:
                            continue
                        n_unwind += 1          # race 成交,保留部位(下方照常 book)
                    else:
                        day_twd -= LAT * 25.0 * 1e-4 * ntl; n_unwind += LAT; continue
                if not cancelfull and gross + working_ntl + ntl > CAP:
                    nrej += 1; st['rej_today'] += max(est, 0) * 1e-4 * ntl; continue
                te = book(t, vc, eu, eb, ntl, 'S1')
                if V == 'v18repro':
                    shadow_add.append((t, 'S1', eu, eb, ntl, te))
            # S2 working quotes: fills / EV band
            for vc in list(working):
                w = working[vc]; P = prod[vc]
                if not P['ok'][t]:
                    del working[vc]; continue
                h = w['h']
                if h <= P['fb'][t] + 1e-9:
                    filled = True
                else:
                    lo = np.searchsorted(P['ts'], max(w['pt'] + 1, t), side='right')
                    hi = np.searchsorted(P['ts'], t + 1, side='right')
                    filled = bool(hi > lo and (P['px'][lo:hi] >= h - 1e-9).any())
                if filled:
                    tf = min(t + 1, P['n'] - 1)          # 成交記在 print 之後的第一格
                    sa_f = P['sa'][tf]
                    del working[vc]
                    if np.isnan(sa_f) or sa_f <= 0:
                        continue
                    eb = (h / sa_f - 1) * 1e4
                    eu = eb - P['an'][tf]
                    if np.isnan(eu):
                        continue
                    if eb <= 0 and delete_neg:
                        continue
                    if eb <= 0:
                        neg_booked += 1
                    if (not reserve and not cancelfull) or V == 'softfull_fg':
                        # v18 式「成交後再用實現 eu 過閘」= 事後刪成交;softfull_fg 只為量化此效應
                        est = est_short(eu, eb, p_sd('S2', tf), slot_free=(eovn + w['ntl'] <= CAP))
                        if est < lam * 0.6 or not gate(cell_of('S2', eb)):
                            continue
                        if not cancelfull and gross + w['ntl'] > CAP:
                            nrej += 1; st['rej_today'] += max(est, 0) * 1e-4 * w['ntl']; continue
                    if (reserve or cancelfull) and gross + w['ntl'] > capfn(tf) and not soft:      # 超額成交:立即反向沖銷,不建部位
                        spr = (P['fa'][tf] - P['fb'][tf]) / P['fb'][tf] * 1e4 if P['fb'][tf] > 0 else 30.0
                        k = LAT if cancelfull else 1.0        # cancelfull 為撤單 race,按延遲折算
                        day_twd -= k * (min(max(spr, 5.0), 200.0) + 5.0) * 1e-4 * w['ntl']
                        n_unwind += k
                        continue
                    elif gross + w['ntl'] > capfn(tf) and soft:
                        # race 超額:以 LAT 機率真的成交並保留;其餘視為撤單成功
                        if np.random.default_rng(int(tf) * 7919 + zlib.crc32(vc.encode()) % 1000).random() > LAT:
                            continue
                        n_unwind += 1
                    te = book(tf, vc, eu, eb, w['ntl'], 'S2')
                    if V == 'v18repro':
                        shadow_add.append((tf, 'S2', eu, eb, w['ntl'], te))
                    w['cool'] = t + 60
                    continue
                eb_h = (h / P['sa'][t] - 1) * 1e4
                if eb_h - P['an'][t] < FLOOR or eb_h <= 0:
                    if P['cond'][t]:
                        w['h'], w['pt'] = P['des'][t], t
                    else:
                        del working[vc]
            # S2 new quotes: candidates this second
            cands = [vc for vc, P in prod.items()
                     if vc not in working and P['cond'][t] and t >= P.get('cool', 0)]
            if cancelfull:
                remaining = capfn(t) - gross
                if remaining <= 0:
                    if full_since < 0:
                        full_since = t
                else:
                    full_since = -1
                for vc in list(working):
                    if working[vc]['ntl'] > remaining:
                        del working[vc]
                for vc in cands:
                    P = prod[vc]; ebq = P['eb'][t]; euq = ebq - P['an'][t]; ntl = P['sa'][t] * P['ntl']
                    if ntl > remaining:
                        nq_skip += 1; continue
                    est = est_short(euq, ebq, p_sd('S2', t), slot_free=(eovn + ntl <= CAP))
                    if est < lam * 0.6 or not gate(cell_of('S2', ebq)):
                        continue
                    working[vc] = {'h': P['des'][t], 'pt': t, 'ntl': ntl}; nq_post += 1
                if V == 'glide30' and t % 60 == 0 and gross > capfn(t):
                    cand_pos = []
                    for i_, p in enumerate(st['open']):
                        b = books.get(p['vc'])
                        if b is None or np.isnan(b['basis_buy_taker_bp'][t]):
                            continue
                        cand_pos.append((b['basis_buy_taker_bp'][t] - p['target'], i_))
                    cand_pos.sort()
                    for d_, i_ in cand_pos:
                        if gross <= capfn(t):
                            break
                        p = st['open'][i_]
                        cost = 20.0 if p.get('day') == day else 34.0
                        day_twd += (p['eu'] + 5 - max(d_, 0.0) - cost) * 1e-4 * p['ntl']
                        gross -= p['ntl']; eovn -= p['ntl']; p['_x'] = True; n_cross += 1
                    st['open'] = [p for p in st['open'] if not p.get('_x')]
            elif reserve:
                # gate at quote time
                scored = []
                for vc in cands:
                    P = prod[vc]; ebq = P['eb'][t]; euq = ebq - P['an'][t]; ntl = P['sa'][t] * P['ntl']
                    est = est_short(euq, ebq, p_sd('S2', t), slot_free=(eovn + ntl <= CAP))
                    if est < lam * 0.6 or not gate(cell_of('S2', ebq)):
                        continue
                    scored.append((est, vc, ntl))
                if V == 'reserveE':
                    scored.sort(reverse=True)
                for est, vc, ntl in scored:
                    working_ntl = sum(w['ntl'] for w in working.values()) * frac
                    if gross + working_ntl * 1.0 + ntl * frac > CAP:
                        nq_skip += 1; st['rej_today'] += max(est, 0) * 1e-4 * ntl; continue
                    working[vc] = {'h': prod[vc]['des'][t], 'pt': t, 'ntl': ntl}
                    nq_post += 1
            else:
                for vc in cands:
                    P = prod[vc]
                    working[vc] = {'h': P['des'][t], 'pt': t, 'ntl': P['sa'][t] * P['ntl']}
                    nq_post += 1
            for vc, w in list(working.items()):
                if 'cool' in w:
                    prod[vc]['cool'] = w['cool']; del working[vc]
        for vc in prod:
            prod[vc].pop('cool', None)
        release_upto(RC)
        if soft:
            over = sum(p['ntl'] for p in st['open']) - CAP
            if over > 0:
                cand_pos = []
                for i_, p in enumerate(st['open']):
                    b = books.get(p['vc'])
                    if b is None or np.isnan(b['basis_buy_taker_bp'][15480]):
                        continue
                    cand_pos.append((b['basis_buy_taker_bp'][15480] - p['target'], i_))
                cand_pos.sort()
                for d_, i_ in cand_pos:
                    if over <= 0:
                        break
                    p = st['open'][i_]
                    cost = 20.0 if p.get('day') == day else 34.0
                    day_twd += (p['eu'] + 5 - max(d_, 0.0) - cost) * 1e-4 * p['ntl']
                    over -= p['ntl']; p['_x'] = True; n_cross += 1
                st['open'] = [p for p in st['open'] if not p.get('_x')]
        st['rej'].append(st['rej_today']); st['rej_today'] = 0.0
        carry = sum(p['ntl'] for p in st['open'])
        st['rows'].append((day, nfill['S1'], nfill['S2'], nsd, nrej, nq_post, nq_skip,
                           neg_booked, flow / 1e6, day_twd, carry / 1e6, n_unwind, n_cross))
    for V in VARIANTS:
        run_variant(V)
    # ---- shadow table updates (from unconstrained-ish v18repro stream; after all decisions)
    for (t, strm, eu, eb, ntl, te) in shadow_add:
        cell = cell_of(strm, eb)
        c = psd_cnt[(strm, tod(t))]; c[0] += 1; c[1] += int(te is not None)
        if te is not None:
            res_hist.append((di, cell, eu + 5 - 20.0, 0.15))
            s = cell_sum[cell]; s[0] += eu + 5 - 20.0; s[1] += 0.15; s[2] += 1
        else:
            shadow_open.append({'vc': vc, 'eu': eu, 'eb': eb, 'cell': cell,
                                'target': (eb - eu) - 5, 'exp': next_exp(day), 'd0': di})
    keep = []
    for p in shadow_open:
        if day > p['exp']:
            pnl, dh = p['eb'] - 34.0, di - p['d0']
        else:
            te = mexit(p['vc'], p['target'], 0)
            if te is None:
                keep.append(p); continue
            pnl, dh = p['eu'] + 5 - 34.0, di - p['d0']
        res_hist.append((di, p['cell'], pnl, dh))
        s = cell_sum[p['cell']]; s[0] += pnl; s[1] += dh; s[2] += 1
    shadow_open = keep
    print(day, {V: round(ST[V]['rows'][-1][9]) for V in VARIANTS}, flush=True)

for V in VARIANTS:
    r = pd.DataFrame(ST[V]['rows'], columns=['day', 'f_S1', 'f_S2', 'sd', 'cap_rej',
                                             'q_post', 'q_skip', 'neg_booked', 'flow_M',
                                             'twd', 'carry_M', 'unwind', 'cross'])
    r.to_csv(SP + f"/resv_{V}{os.environ.get('EV_TAG', '')}_daily.csv", index=False)
    fills = r.f_S1 + r.f_S2
    print(f"{V:9s}: {r.twd.mean():>8,.0f}/天 中位{r.twd.median():>7,.0f} "
          f"S1 {r.f_S1.mean():.0f}/d S2 {r.f_S2.mean():.0f}/d 當沖率{r.sd.sum()/max(fills.sum(),1):.0%} "
          f"掛出{r.q_post.mean():.0f} 額度不足未掛{r.q_skip.mean():.0f} 負basis入帳{r.neg_booked.sum()} "
          f"carry{r.carry_M.mean():.1f}M 超額沖銷{r.unwind.sum():.1f} 滑降cross{r.cross.sum()}", flush=True)
