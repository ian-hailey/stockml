import pandas as pd
import signals
import numpy as np
import ohlcv

class Symbol(object):
    def __init__(self, db, symbol):
        self.db = db
        self.symbol = symbol
        self.info = db.get_symbol_info(symbol)

    def generate_ohlcv(self, datetime=None):
        days = pd.date_range(self.info['start'].date(), self.info['end'].date(), freq='1D')
        days = days.normalize()
        for day in days:
            if day.weekday() < 5:
                df_day = self.db.get_symbol_ohlcv(self.symbol, [day, day + pd.DateOffset(hours=24)], resolution='days')
                if df_day.size == 0:
                    df = self.db.get_symbol_ohlcv(self.symbol, [day, day + pd.DateOffset(hours=24)])
                    if df.size != 0:
                        df = ohlcv.Ohlcv(df)
                        df_day = df.resample(period='1d')
                        print("Inserting OHLCV day {} {}".format(self.symbol, df_day.data.values))
                        self.db.insert_ohlcv_days(df_day.data, self.symbol)
        pass

    def generate_signals(self, datetime=None):
        days = pd.date_range(self.info['start'].date(), self.info['end'].date(), freq='1D')
        days = days.normalize()
        for day in days:
            if day.weekday() < 5:
                print("Checking signals for {} day {}".format(self.symbol, day))
                zc_signal = self.db.get_symbol_signals(self.symbol, [day + pd.DateOffset(hours=9), day + pd.DateOffset(hours=9, minutes=1)])
                if zc_signal.size == 0:
                    df = self.db.get_symbol_ohlcv(self.symbol, [day, day + pd.DateOffset(hours=24)])
                    if df.size != 0:
                        df = ohlcv.Ohlcv(df)
                        df.fill_gaps()
                        signal = signals.Signals()
                        zc_signal = signal.zc_from_df(df, np.array([day]), stop=16, thread_count=1)
                        self.db.insert_signals(zc_signal, self.symbol)
        pass

