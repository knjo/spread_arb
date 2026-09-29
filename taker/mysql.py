from sqlalchemy import create_engine, text
import pandas as pd
import os

try:
    from dotenv import load_dotenv
except ImportError:  # HFT 主環境未裝 python-dotenv，直接讀環境變數即可
    def load_dotenv():
        return False


class BaseMySQLLoader:
    def __init__(self, user_name='data.admin', user_pwd='automated', db_host="192.168.1.187"):
        load_dotenv()
        db_url = f"mysql+pymysql://{user_name}:{user_pwd}@{os.getenv('MYSQL_HOST') or db_host}:3306"
        self.engine = create_engine(db_url, pool_pre_ping=True)

    def execute(self, query, params=None):
        with self.engine.begin() as conn:
            conn.execute(text(query), params or {})

    def query(self, query, params=None):
        with self.engine.begin() as conn:
            return pd.read_sql(text(query), conn, params=params or {})

    def to_sql(self, df, db_name, table_name, if_exists="append"):
        if df.empty:
            return
        with self.engine.begin() as conn:
            df.to_sql(table_name, con=conn, index=False, if_exists=if_exists, schema=db_name)


class StrategyMySQLLoader(BaseMySQLLoader):
    def __init__(self) -> None:
        super().__init__()

    def is_trade_day(self, cal_date):
        query_str = f"SELECT DayType FROM Common.calendar_view where date = {cal_date};"
        df = self.query(query_str)
        # L5 防呆：查無此日（df 空）原會 .iloc[0] 觸發 IndexError；與同類函式一致 fail-loud。
        if df.empty:
            raise Exception(f"calendar_view 查無 {cal_date}（日曆缺該日，無法判斷是否交易日）")
        return df['DayType'].iloc[0] == 'TradeDay'

    def get_last_trade_day(self, date):
        query_str = f"SELECT Date FROM Common.calendar_view where date < '{date}' and DayType = 'TradeDay' order by Date desc limit 1;"
        df = self.query(query_str)
        if df.empty:
            raise Exception("No date found")
        return df['Date'].iloc[0].strftime("%Y%m%d")

    def get_futures_basic_info(self, date):
        query_str = fr"SELECT quote_code, value_code, ref_price, contract_size, decimal_locator, end_date FROM ProductInfo.taifex_pib_view where date= {date} and prod_kind = 'stock' and ins_type = 'futures';"
        df = self.query(query_str)
        if df.empty:
            raise Exception("No date found")
        return df

    def get_futures_settle_price(self, date):
        query_str = fr"SELECT quote_code,settlement_price FROM MarketInfo.taifex_futures_trades_daily where date = {date} and trading_session = 'day' and char_length(quote_code) = 5;"
        df = self.query(query_str)
        if df.empty:
            raise Exception("No date found")
        return df

    def get_stock_closing_price(self, date):
        query_str = fr"SELECT quote_code, close_price FROM MarketInfo.twse_security_trades_daily where date = {date} and char_length(quote_code) = 4;"
        df = self.query(query_str)
        if df.empty:
            raise Exception("No date found")
        return df
