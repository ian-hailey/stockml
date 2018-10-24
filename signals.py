import matplotlib
matplotlib.use('Agg')
import pandas as pd
import time
from multiprocessing import Pool

class signals(object):
    def __init__(self):
        self.signal_results = pd.DataFrame(columns=['t_out', 'gain', 't_delta', 'gradient'])

    def find_signals(self, df_day_s, day, stop=10, resolution='M'):
        results = []
        gain = 0.0
        dates_pre = pd.date_range(day + pd.DateOffset(hours=4), day + pd.DateOffset(hours=9.5), freq='S')
        df_pre = df_day_s.daterange(dates_pre)
        df_pre_close = df_pre.data['Close'].iloc[-1]
        df_pre_vol = df_pre.data['Volume'].sum()
        if df_pre_vol:
            df_day_s.compute_stoch(fastk=14*60, slowd=60)
            df_day_s.compute_ma(time=5, sample=60)
            df_day_s.compute_ma(time=10, sample=60)
            df_day_m = df_day_s.resample('60s')
            df_day_m.compute_rsi()
            df_day_m.compute_stoch()
            df_day_m.compute_ma(time=5, source='Close')
            df_day_m.compute_ma(time=10, source='Close')

            # create down samples minute resolution first 30 minutes
            dates_open_s = pd.date_range(day + pd.DateOffset(hours=9.5), day + pd.DateOffset(hours=stop), freq='S')
            df_open_s = df_day_s.daterange(dates_open_s, ohlcvOnly=False)
            dates_open_m = pd.date_range(day + pd.DateOffset(hours=9.5), day + pd.DateOffset(hours=stop), freq='60S')
            df_open_m = df_day_m.daterange(dates_open_m, ohlcvOnly=False)

            open = df_open_s.data['Open'][0]
            timeIn = 0
            stochIn = 0
            rsiIn = 0
            spIn = 0
            maIn = 0

            print("{} pre_close={} pre_vol={}".format(day.strftime('%Y-%m-%d'), df_pre_close, df_pre_vol))
            if resolution is 'S':
                df_open = df_open_s
            else:
                df_open = df_open_m
            for row in df_open.data.itertuples():
                minute = row.Index.strftime('%Y-%m-%d %H:%M')
                row_m = df_open_m.data.loc[minute]
                wick = (row.High - row.Low) / ((row.Close - row.Open) + 0.0001)
                maDelta = (row.ma5 / open) - (row.ma10 / open)
     #           print("{} maIn={:.2f} rsiIn={:.0f} stochIn={:.0f} wick={:.4f} SP={:.2f}".format(row.Index.strftime('%Y-%m-%d %H:%M'),
     #                                                                               maDelta, row.rsi, row.stoch_k, wick, row.Close))
                if spIn == 0 and row.Close > row.Open and row_m.rsi > 50 and row.stoch_k > 70 and maDelta > 0.001:
                    spIn = row.Close
                    stochIn = row.stoch_k
                    rsiIn = row_m.rsi
                    maIn = row.ma5 - row.ma10
                    timeIn = row.Index
                elif spIn != 0 and (row_m.rsi < (rsiIn - 5) or row.stoch_k < 60 or maDelta < -0.001):
                    delta = (row.Close / spIn) - 1.0
                    if delta > 0.003:
                        gain = gain + delta
                        print("Trade Day {}".format(day.strftime('%Y-%m-%d')))
                        print("  In  {} maIn={:.2f} rsiIn={:.0f} stochIn={:.0f} SP={:.2f}".format(timeIn.strftime('%H:%M:%S'), maIn, rsiIn, stochIn, spIn))
                        print("  Out {} maDelta={:.2f} rsi={:.0f} stoch={:.0f} SP={:.2f} ({:.2f}%)".format(row.Index.strftime('%H:%M:%S'), row.ma5 - row.ma10, row_m.rsi, row.stoch_k, row.Close, delta*100))
                        results.append([gain, timeIn, row.Index])
                    spIn = rsiSig = stochSig = 0
            if spIn != 0:
                delta = (df_open.data['Close'][-1] / spIn) - 1.0
                if delta > 0.003:
                    gain = gain + delta
                    print("Trade Day {}".format(day.strftime('%Y-%m-%d')))
                    print("  In  {} maIn={:.2f} rsiIn={:.0f} stochIn={:.0f} SP={:.2f}".format(timeIn.strftime('%H:%M:%S'), maIn, rsiIn, stochIn, spIn))
                    print("  Out {} maDelta={:.2f} rsi={:.0f} stoch={:.0f} SP={:.2f} ({:.2f}%)".format(row.Index.strftime('%H:%M:%S'), row.ma5 - row.ma10, row_m.rsi, row.stoch_k, row.Close, delta * 100))
                    results.append([gain, timeIn, row.Index])
        return results


    def add_singal(self, gain, t_in, t_out):
        if gain != 0.0:
            t_delta = int((t_out - t_in).total_seconds())
            gradient = gain / t_delta
            if gradient > 2.7e-6:
                self.signal_results.loc[t_in] = [t_out, gain, t_delta, gain / t_delta]

    def find_signals_callback(self, results):
        for result in results:
            [gain, t_in, t_out] = result
            self.add_singal(gain, t_in, t_out)

    def signals_from_df(self, df, days, stop=10, thread_count=1):
        start_time = time.time()
        if thread_count > 1:
            with Pool(processes=thread_count) as pool:
                for i in range(days.size):
                    dates_day_s = pd.date_range(days[i] + pd.DateOffset(hours=4), days[i] + pd.DateOffset(hours=stop), freq='S')
                    df_day_s = df.daterange(dates_day_s)
                    pool.apply_async(self.find_signals, args=(df_day_s, days[i], stop, 'M'), callback=self.find_signals_callback)
                pool.close()
                pool.join()
        else:
            for i in range(days.size):
                dates_day_s = pd.date_range(days[i] + pd.DateOffset(hours=4), days[i] + pd.DateOffset(hours=stop), freq='S')
                df_day_s = df.daterange(dates_day_s)
                results = self.find_signals(df_day_s, days[i], stop, 'M')
                for result in results:
                    [gain, t_in, t_out] = result
                    self.add_singal(gain, t_in, t_out)
        gains = pd.DataFrame(index=days, columns=['gain', 'yoy'])
        gains['gain'].fillna(value=0.0, inplace=True)
        gains['yoy'].fillna(value=0.0, inplace=True)
        last_total = total_gains = total_losses = total = 0.0
        for result in self.signal_results.itertuples():
            if result.gain > 0.0:
                total_gains = total_gains + result.gain
            else:
                total_losses = total_losses + result.gain
            total = total + result.gain
            gains['gain'].loc[result.Index.strftime('%Y-%m-%d')] += result.gain * 100
            gains['yoy'].loc[result.Index.strftime('%Y-%m-%d')] += total * 100
            if total != last_total:
                print("total={:.2f}%".format(total * 100))
                last_total = total
        print("total={:.2f}% (gains={:.2f}% losses={:.2f}%)".format(total * 100, total_gains * 100, total_losses * 100))
        print("--- %s seconds ---" % (time.time() - start_time))
        ax = gains.plot()
        fig = ax.get_figure()
        fig.set_dpi(140)
        fig.savefig("gains.jpg")