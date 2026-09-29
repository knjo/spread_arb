"""Render reviewed numeric inputs without reselecting profitable trades or dates."""
import argparse
import json
from pathlib import Path

import polars as pl


def number(value,digits=0):
    return '未定義' if value is None else f'{value:,.{digits}f}'


def percent(value):
    return '未定義' if value is None else f'{value*100:.2f}%'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def write(output):
    result=json.loads((output/'comparison.json').read_text())
    assert result['status']=='completed'
    rows=result['summaries']
    assert all(r['observed_sessions']==85 and r['postwarm_observed_sessions']==65
               and r['postwarm_first_day']=='20260601' for r in rows), 'full-period report required'
    short={'q_net_release25_deep':'舊 S2 時鐘＋歷史增額',
           'q_event20_deep':'立即補 S2／20M',
           'q_event_release25_deep':'立即補 S2＋歷史增額'}
    whole=[];turnover=[];costs=[];execution=[];streams=[];calibration=[];hedges=[]
    for r in rows:
        name=r['portfolio'];label=short[name]
        folder=Path(r['source']).parent/'report'/name
        whole.append([label,number(r['net_per_observed_day_twd'],2),
            number(r['postwarm_net_per_observed_day_twd'],2),percent(r['postwarm_simple_annual_250']),
            percent(r['postwarm_annual_250_on_peak_committed']),number(r['postwarm_net_per_day_delta_vs_old_twd'],2)])
        turnover.append([label,number(r['postwarm_entries_per_day'],2),number(r['postwarm_s2_entries_per_day'],2),
            number(r['postwarm_normal_close_cash_turns_per_20M_per_day'],3),
            number(r['postwarm_stock_buy_cash_per_day_twd']),number(r['postwarm_expiry_accounting_cash_per_day_twd']),
            percent(r['postwarm_stock_cash_utilization_20M'])])
        costs.append([label,number(r['realized_twd']),number(r['final_marked_inventory_twd']),
            number(r['actual_cash_funding_2pct_twd']),number(r['expiry_net_twd']),
            number(r['peak_stock_cash_twd']),number(r['peak_committed_twd']),
            number(r['committed_over25_intraday_seconds']/3600,2)])
        hedges.append([label,r['s2_stock_hedged_entries'],r['s2_entries_without_completed_stock_hedge'],
            number(r['s2_stock_hedge_delay_p50_ms'],2),number(r['s2_stock_hedge_delay_p95_ms'],2),
            number(r['s2_stock_hedge_delay_max_ms'],2),r['s2_stock_hedge_later_than_configured'],
            r['s2_stock_hedge_later_than5s']])
        stream_frame=pl.read_csv(folder/'streams.csv')
        funding_total=stream_frame['funding_twd'].sum()
        for row in stream_frame.iter_rows(named=True):
            cash_days=row['funding_twd']*365/.02
            streams.append([label,row['stream'],row['fills'],number(row['postwarm_net_per_observed_day_twd'],2),
                percent(row['funding_twd']/funding_total if funding_total else None),
                number(row['net_equity_twd']/cash_days*10_000 if cash_days else None,2)])
        for row in pl.read_csv(folder/'q_calibration.csv').iter_rows(named=True):
            calibration.append([label,row['stream'],row['fills'],row['unresolved'],
                number(row['q_net_bp'],2),number(row['actual_net_bp'],2),
                number(row['q_days'],2),number(row['actual_days'],2)])
        latency=pl.read_csv(folder/'quote_latency_exposure.csv').filter(pl.col('stream')=='S2')
        if latency.height:
            l=latency.row(0,named=True)
            execution.append([label,l['maker_entries'],l['under50ms'],percent(l['under50ms']/l['maker_entries']),
                number(l['under50ms_net_twd']),number(r['F_outbound_messages']/r['observed_sessions'],1),
                r['F_max_messages_per_second']])
    release=[[short[r['portfolio']],r['stream'],'carry' if r['carry'] else '當日新倉',
        percent(r['predicted_today_fraction']),percent(r['actual_today_fraction']),
        percent(r['predicted_next_fraction']),percent(r['actual_next_fraction'])] for r in result['release_calibration']]
    timing=[[short[r['portfolio']],number(r['mean_opening_committed_twd']),number(r['mean_committed_at10_twd']),
        number(r['mean_closing_committed_twd']),number(r['mean_actual_carry_market_release_before10_twd']),
        percent(r['fraction_observed_closes_over20']),percent(r['fraction_observed_closes_over25'])]
        for r in result['capacity_timing']]
    periods=[[short[r['portfolio']],r['first_day']+'–'+r['last_day'],
        '雙流新指令' if r['mode']=='both_entry_sources' else 'S2新單＋既有S1 carry',r['observed_sessions'],
        number(r['net_per_observed_day_twd'],2),number(r['normal_close_cash_turns_per_20M_per_day'],3)]
        for r in result['availability_periods']]
    weighted=[[short[r['portfolio']],r['stream'],number(r['predicted_nominal_weighted_bp'],2),
        number(r['actual_nominal_weighted_bp'],2),number(r['actual_net_twd'])]
        for r in result['weighted_q_calibration']]
    sections=[
        '# 現行 Q 的立即補 S2：完整數值比較',
        '本檔由已通過核驗的相同期間輸出產生。數值審閱與完整研究結論另見 canonical 報告。所有收益都是整體 S1＋S2，非加在舊收益上的增量。',
        '## 收益',
        '全期為 85 個資料日，包含 20 日空倉暖機；6/1–9/2 為暖機後 65 個資料日。兩組歷史增額策略只有 S2 時鐘不同；20M 組同時改了額度，不可將其差額全部歸因取消 CD。',
        table(['政策','全期淨利／日','暖機後淨利／日','暖機後250日年化／20M','同年化／實際峰值本金','暖機後日均較舊版'],whole),
        '年化僅為規劃換算；20M×30%／250 日需要 24,000 元／日。淨利包含期末持倉估值與 2% 假設現貨資金成本，並非全部已落袋收益。',
        '## 新單來源可用期間',
        table(['政策','期間','新單來源','資料日','淨利／日','正常平倉本金／20M／日'],periods),
        '既有S1新指令到8/13，後段13個資料日只有S2新進場，仍承接S1持倉及其損益。這個分段由來源覆蓋範圍決定，沒有重設庫存或重訓Q，也不挑選高收益日。完整65日仍一起呈現。若分段端點缺官方估值，分段收益維持未定義。',
        '## 進場與真正完成的週轉',
        table(['政策','進場筆／日','其中S2筆／日','正常平倉本金／20M／日','股票新買入元／日','C8釋放原始本金元／日','日內平均股票投入／20M'],turnover),
        '以上為暖機後期間。流量／日以65個資料日為分母；日內平均股票投入則平均所有暖機後回放session，包括行情中斷日的延續庫存。正常平倉週轉須兩腳完成；撤掉的未成交預留不算，新進場後仍持有的 carry 也不算完成一輪。C8 另列，未混入正常市場平倉週轉。',
        '## 現金、估值與超額',
        table(['政策','已實現損益','期末持倉估值','2%實際資金成本','C8淨損益歸因','股票現金峰值','committed峰值','日內超25M總小時'],costs),
        '20M／25M 為准入規則，延遲撤單的實際成交可超額。C8 是到期 basis-zero 帳務慣例，未按原始現貨深度清算；其收益歸因不能直接當成刪除 C8 後的另一條可執行收益路徑。期貨保證金融資未另外建模。',
        '## 10:00 前後與收盤容量',
        table(['政策','平均開盤committed','平均10:00committed','平均收盤committed','舊carry於10:00前實際市場釋放／日','收盤超20M日比例','收盤超25M日比例'],timing),
        '使用暖機後65個資料日；10:00數值包含該timestamp全部實際帳本變動。舊carry釋放排除C8到期帳務，亦不把當日新進新出的量混入。開盤值是前一日延續帳本、當日到期事件處理以前的值。',
        '## S1／S2 歸因',
        table(['政策','進場腳','成交筆','暖機後淨損益／日','股票資金日占比','每股票本金日淨bp'],streams),
        '股票資金日為實際原始成本乘持有日曆時間；損益含期末估值。此為原投組的資金生產力歸因，重新分配後的收益仍須依成交與持倉時序回放。',
        '## 成熟合約的 Q 校準',
        table(['政策','進場腳','成熟成交筆','未解決','預測淨bp','已結案實際淨bp','預測日曆日','已結案實際日曆日'],calibration),
        '成熟分母為原合約到期已過的全部成交；實際 bp／持有期只在已結案樣本可得，因此未解決數必須一併看。現行表保持原政策訓練來源，沒有用新版同日結果回填。',
        table(['政策','進場腳','已結案名目加權預測bp','同群實現bp','同群淨損益'],weighted),
        '名目加權表的預測與實際都限於同一群成熟已結案交易；一般交易按現貨原始本金、partial rollback按原准入名目，沿用Q校準的報酬分母。這是事後檢查，不影響盤中查表。',
        '## 實際存續部位的釋放預測',
        table(['政策','進場腳','風險集','今日折扣預測','今日市場實際','隔日增量預測','隔日市場實際'],release),
        '依部位原始名目加權，每5分鐘第一個非空風險集；同一部位重複觀察有相關性。隔日欄只比較結果已知的日期，排除行情中斷與區間外。正常／強制兩腳市場平倉可釋放，C8 不列作市場釋放。',
        '## 報價延遲與訊息量曝險',
        table(['政策','S2已完成現貨hedge筆','仍未完成','hedge中位ms','hedge95%分位ms','最長ms','超過設定50ms筆','超過5秒筆'],hedges),
        '此表量測期貨maker成交到現貨hedge完成的實際回放時間。現貨hedge須L1–L5足額才整筆執行，不足則保留期貨曝險並重試，未模擬先成交部分現貨。延遲次數本身不代表單一失敗原因；未完成者另列，不以已完成群的分位數掩蓋。新S2須等實際hedge完成才解除同商品補單鎖定。',
        table(['政策','S2成交筆','掛後50ms內成交','占比','此群淨損益歸因','期貨訊息／全期資料日','期貨單秒峰值'],execution),
        '回放新單立即取得模擬 queue、未執行實際下單訊息節流；50ms內成交與其損益僅為曝險歸因，不能事後刪掉這些單宣稱另一個績效。市場熱度相近仍不足以保證實盤與研究相同。',
        '完整機器數值、月別、持有時間、融資敏感度與圖表見 comparison.json、policies.csv 及各策略 report；所有固定政策均保留。',
    ]
    path=output/'findings.md';path.write_text('\n\n'.join(sections)+'\n')
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    print(write(p.parse_args().output))
