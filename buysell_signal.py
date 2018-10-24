import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from multiprocessing import Pool

class signal(object):
    def __init__(self):
        self.signal = pd.DataFrame(columns=['buysell'])

    def callback(self, results):
        self.signal = pd.concat([self.signal, results])

    def plot(self, day, close, ma5, buysell, openIndex):
        fig = plt.figure(frameon=False, figsize=(8, 4), dpi=100)
        fig.set_dpi(140)
        fig.set_size_inches(60, 15)
        ax = fig.add_subplot(111)
        ax.plot(close)
        ax.plot(ma5, ls='--')
        ax2 = ax.twinx()
        ax2.plot(buysell, color='green')
        plt.axhline(y=0.0, ls='--', color='grey')
        plt.axvline(x=openIndex, ls='--', color='grey')
        filename = "wdc_buysell/wdc_{}.jpg".format(day.strftime('%Y-%m-%d'))
        fig.savefig(filename)
        plt.close()

    def generate(self, df_day_s, day, stop=10, resolution='M'):
        dates_pre = pd.date_range(day + pd.DateOffset(hours=4), day + pd.DateOffset(hours=9.5), freq='S')
        df_pre = df_day_s.daterange(dates_pre)
        df_pre_close = df_pre.data['Close'].iloc[-1]
        df_pre_vol = df_pre.data['Volume'].sum()
        if df_pre_vol:
            df_day_s.compute_ma(time=5, sample=60)
            df_day_s.compute_ma(time=10, sample=60)
            df_day_m = df_day_s.resample('60s')
            # be careful not to change the order of these otherwise the row index mapping will change
            df_day_m.compute_ma(time=5, source='Close')
            df_day_m.compute_ma(time=10, source='Close')

            # create down samples minute resolution first 30 minutes
            dates_open_s = pd.date_range(day + pd.DateOffset(hours=4.0), day + pd.DateOffset(hours=stop), freq='S')
            df_open_s = df_day_s.daterange(dates_open_s, ohlcvOnly=False)
            dates_open_m = pd.date_range(day + pd.DateOffset(hours=4.0), day + pd.DateOffset(hours=stop), freq='60S')
            df_open_m = df_day_m.daterange(dates_open_m, ohlcvOnly=False)
            open = df_open_s.data['Open'][day.strftime('%Y-%m-%d') + ' 09:30:00']
            df_open_m.data['Close'] = df_open_m.data['Close'] / open
            df_open_m.data['ma5'] = df_open_m.data['ma5'] / open
            df_open_m.compute_gradient(source='ma5')
            zero_crossings = np.where(np.diff(np.sign(df_open_m.data['ma5grad'].values)))[0]
            if resolution is 'S':
                df_open = df_open_s
                zero_crossings = zero_crossings * 60
            else:
                df_open = df_open_m
            openIndex = df_open.data.index.get_loc(day.strftime('%Y-%m-%d') + ' 09:30:00')
            print("{} pre_close={} pre_vol={}".format(day.strftime('%Y-%m-%d'), df_pre_close, df_pre_vol))
            signal = pd.DataFrame(0.0, index=df_open.data.index, columns=['buysell'])
            nearest_zc = -1
            signal_values = signal['buysell'].values
            for rowIndex in range(df_open.data.index.size):
                if nearest_zc < rowIndex:
                    nearest_zc = min(zero_crossings, key=lambda x: x < rowIndex)
                    nearest_zc_ma5 = df_open.data.iloc[nearest_zc].ma5
                row = df_open.data.values[rowIndex] # not using iloc object as slow
    #            print("row={} nearest_zc={} maDelta={}".format(rowIndex, nearest_zc, row.ma5grad))
                if nearest_zc >= rowIndex:
                    signal_values[rowIndex] = nearest_zc_ma5 - row[5]
            self.plot(day, df_open.data['Close'].values, df_open.data['ma5'].values, signal['buysell'].values, openIndex)
            signal['buysell'].fillna(value=0.0, inplace=True)
            return signal
        return None

    def from_df(self, df, days, stop=10, thread_count=1):
        if thread_count > 1:
            with Pool(processes=thread_count) as pool:
                for i in range(days.size):
                    dates_day_s = pd.date_range(days[i] + pd.DateOffset(hours=4), days[i] + pd.DateOffset(hours=stop), freq='S')
                    df_day_s = df.daterange(dates_day_s)
                    pool.apply_async(self.generate, args=(df_day_s, days[i], stop, 'S'), callback=self.callback)
                pool.close()
                pool.join()
        else:
            for i in range(days.size):
                dates_day_s = pd.date_range(days[i] + pd.DateOffset(hours=4), days[i] + pd.DateOffset(hours=stop), freq='S')
                df_day_s = df.daterange(dates_day_s)
                results = self.generate(df_day_s, days[i], stop, 'S')
                self.signal = pd.concat([self.signal, results])
        return self.signal
