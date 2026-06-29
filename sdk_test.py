from sdk_core import TwMarketData
from mysql import StrategyMySQLLoader

tw = TwMarketData()

df = tw.get_equity_basic_info(date=20260609,ins_type='stock')

# 開盤參考價
ref_price = df['opening_ref_price']
# 若標記為X，代表可當沖
allow_day_trade_mark = df['allow_day_trade_mark'] == "X"

mysql = StrategyMySQLLoader()
futures_info = mysql.get_futures_basic_info(date=20260609)
print(futures_info)