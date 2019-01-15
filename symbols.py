import pandas as pd
import signals
import numpy as np
import ohlcv
from multiprocessing import Pool

class Symbol(object):
    def __init__(self, db, symbol, autocreate=False):
        self.db = db
        self.symbol = symbol
        self.info = db.get_symbol_info(symbol)
        if self.info is None:
            self.db.create_symbol_schema(self.symbol)
            self.info = self.db.get_symbol_info(self.symbol)

    def create(self):
        self.db.create_symbol_schema(self.symbol)
        self.info = self.db.get_symbol_info(self.symbol)

    def generate_ohlcv(self, datetime=None, thread_count=2):
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
                        print("Inserting {} OHLCV day {} Close {:.2f} Volume {:d}".format(self.symbol,
                                                                                          day.strftime('%Y-%m-%d'),
                                                                                          df_day.data.values[0][3],
                                                                                          int(df_day.data.values[0][
                                                                                                  4])))
                        self.db.insert_ohlcv_days(df_day.data, self.symbol)
        pass

    def signal_callback(self, results):
        self.db.insert_signals(results, self.symbol)

    def generate_signals_day(self, day, df):
        print("Checking signals for {} day {}".format(self.symbol, day.strftime('%Y-%m-%d')))
        df = ohlcv.Ohlcv(df)
        df.fill_gaps()
        signal = signals.Signals()
        zc_signal = signal.zc_from_df(df, np.array([day]), stop=16, thread_count=1)
        return zc_signal

    def generate_signals(self, datetime=None, recalc=False, thread_count=1):
        days = pd.date_range(self.info['start'].date(), self.info['end'].date(), freq='1D')
        days = days.normalize()
        with Pool(processes=thread_count) as pool:
            for day in days:
                if day.weekday() < 5:
                    if recalc is False:
                        zc_signal = self.db.get_symbol_signals(self.symbol, [day + pd.DateOffset(hours=9),
                                                                   day + pd.DateOffset(hours=9, minutes=1)])
                    if (recalc is True) or (zc_signal.size != 0):
                        df = self.db.get_symbol_ohlcv(self.symbol, [day, day + pd.DateOffset(hours=24)])
                        if df.size != 0:
                            if thread_count > 1:
                                pool.apply_async(self.generate_signals_day, args=(day, df), callback=self.signal_callback)
                            else:
                                results = self.generate_signals_day(day, df)
                                self.db.insert_signals(results, self.symbol)
            pool.close()
            pool.join()
        pass

    def import_ohlcv_csv(self, filename):
        self.db.import_ohlcv_csv(filename, self.symbol)
        pass
