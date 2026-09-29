"""從原始 tick 重建 canonical walkforward/daily 之後日期的 1Hz 格（研究用近似）。

輸出欄位對齊回測所需子集：spot_bid/ask/bid2/lots/sequence、fut_bid/ask(=exec)、
basis_mid、anchor_ewma_120s、basis_buy/sell_taker、contract_size。
商品宇宙、合約家族、contract_size 凍結自 canonical 2026-08-13；前月合約 =
同家族取當日最大 TotalFillLots（8/19 到期換月自動處理）。
簡化（已揭露）：只丟 TrialMatch 列，沒有 ref-band / trial-match 重開等合法性邏輯。
已知壞檔：NAS 2026-08-28 stock_futures.parquet（4 bytes）→ 跳過。

用法：EV_LOOKUP_WORK=~/ev_lookup_work uv run python ext_grid_builder.py 20260814 20260817 ...
"""
import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WF = '/home/kevin/Project/HFT/src/research/futures_spot_spread/maker/data/walkforward/'
WORK = os.environ.get('EV_LOOKUP_WORK', os.path.expanduser('~/ev_lookup_work'))
OUT = WORK + '/ext_daily'
RC = 15600
ALPHA = 1 - np.exp(-1 / 120.0)


def build(day: str, univ, fam, csize) -> bool:
    os.makedirs(OUT, exist_ok=True)
    out = f'{OUT}/{day}.parquet'
    if os.path.exists(out):
        return True
    fut_path = f'/mnt/NAS/Parquet/Ticks/2026/{day[4:6]}/{day[6:8]}/stock_futures.parquet'
    try:
        pq.read_schema(fut_path)
    except Exception as e:
        print(day, 'SKIP bad futures file:', str(e)[:60])
        return False
    open_ns = pd.Timestamp(f'{day} 01:00:00').value
    sp = pq.read_table(f'/media/kevin/SSD2/Data/tickData/{day}_StockTick.parquet',
                       columns=['RecvTime', 'ValueCode', 'ChannelSeq', 'TrialMatch',
                                'BidPrice1', 'BidPrice2', 'AskPrice1', 'BidLots1'],
                       filters=[('ValueCode', 'in', univ)]).to_pandas()
    sp = sp[sp['TrialMatch'] == 0]
    sp['sec'] = (sp['RecvTime'].to_numpy().astype('datetime64[ns]').astype('int64')
                 - open_ns) // 10**9
    sp = sp[(sp['sec'] >= 0) & (sp['sec'] <= RC)]
    fu = pq.read_table(fut_path,
                       columns=['RecvTime', 'ValueCode', 'QuoteCode', 'TrialMatch',
                                'BidPrice1', 'AskPrice1', 'DecimalLocator', 'TotalFillLots'],
                       filters=[('ValueCode', 'in', univ)]).to_pandas()
    fu = fu[fu['TrialMatch'] == 0]
    fu['fam'] = fu['QuoteCode'].str[:3]
    fu = fu[fu['fam'] == fu['ValueCode'].map(fam)]
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
    frames, idx = [], np.arange(RC + 1)
    for vc in univ:
        s, f = sp[sp['ValueCode'] == vc], fu[fu['ValueCode'] == vc]
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
        anchor = np.roll(pd.Series(bm).ewm(alpha=ALPHA, ignore_na=True).mean().to_numpy(), 1)
        anchor[0] = np.nan
        frames.append(pd.DataFrame({
            'ValueCode': vc, 'seconds_from_open': idx,
            'spot_bid': sb, 'spot_bid2': sg['BidPrice2'].to_numpy(), 'spot_ask': sa,
            'spot_bid_lots': sg['BidLots1'].to_numpy(),
            'spot_sequence': sg['ChannelSeq'].to_numpy(),
            'fut_exec_bid': fb, 'fut_exec_ask': fa, 'fut_bid': fb, 'fut_ask': fa,
            'basis_mid_bp': bm, 'anchor_ewma_120s_bp': anchor,
            'basis_buy_taker_bp': (fa / sb - 1) * 1e4,
            'basis_sell_taker_bp': (fb / sa - 1) * 1e4,
            'contract_size': csize[vc]}))
    pd.concat(frames).to_parquet(out, index=False)
    print(day, len(frames), 'products')
    return True


if __name__ == '__main__':
    base = pd.read_parquet(WF + 'daily/Date=20260813/causal_fair.parquet',
                           columns=['ValueCode', 'QuoteCode', 'contract_size']
                           ).dropna().groupby('ValueCode').first()
    univ = sorted(base.index)
    fam = {vc: r.QuoteCode[:3] for vc, r in base.iterrows()}
    csize = {vc: float(r.contract_size) for vc, r in base.iterrows()}
    for day in sys.argv[1:]:
        build(day, univ, fam, csize)
