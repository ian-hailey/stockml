import data
import buysell_signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time
import numpy as np
from matplotlib.dates import DateFormatter
from multiprocessing import Pool, Lock
from scipy.signal import argrelextrema

def plot_signal_lines(ax, signals, line_width):
    if signals is not None:
        for signal in signals.itertuples():
            plt.axvline(x=signal.Index, linewidth=line_width, color='grey')
            plt.axvline(x=signal.t_out, linewidth=line_width, ls='--', color='grey')

def plot_day(df_day_s, day, stop=10, signals=None, plotMA5s=False):
    # create date time index for the pre-market
    dates_pre = pd.date_range(day + pd.DateOffset(hours=4), day + pd.DateOffset(hours=9.5), freq='S')
    df_pre = df.daterange(dates_pre)
    df_pre_close = df_pre.data['Close'].iloc[-1]
    df_pre_vol = df_pre.data['Volume'].sum()
    if df_pre_vol:
        df_day_s.compute_rsi()
        df_day_s.compute_stoch(fastk=14*60, slowd=60)
        df_day_s.compute_ma(time=5, sample=60)
        df_day_s.compute_ma(time=10, sample=60)

        df_day_m = df_day_s.resample('60s')
        df_day_m.compute_rsi()
        df_day_m.compute_stoch()
        df_day_m.compute_ma(time=5, source='Close')
        df_day_m.compute_ma(time=10, source='Close')

        # create date time index for the first 30 minutes of market second resolution
        dates_open_s = pd.date_range(day + pd.DateOffset(hours=9.5), day + pd.DateOffset(hours=stop), freq='S')
        df_open_s = df_day_s.daterange(dates_open_s, ohlcvOnly=False)

        # create down samples minute resolution first 30 minutes
        dates_open_m = pd.date_range(day + pd.DateOffset(hours=9.5), day + pd.DateOffset(hours=stop), freq='60S')
        df_open_m = df_day_m.daterange(dates_open_m, ohlcvOnly=False)

        fig = plt.figure(1)
        fig.set_dpi(140)
        fig.set_size_inches(60, 15)
        line_width = 1.0
        plot_size = (5, 10)  # 3 rows 10 columns
        gridspec.GridSpec(plot_size[0], plot_size[1])

        # Plot the candles
        ax = plt.subplot2grid((plot_size[0], plot_size[1]), (0, 0), colspan=10, rowspan=3)  # plot at row 0 column 1:9
        df_open_m.plot_candlestick(ax)

        # Plot the volume
        ax2 = ax.twinx()
        df_open_m.plot_volume_bars(ax2)

        # Plot any signal lines
        plot_signal_lines(ax, signals, line_width)

        # Plot the SP
        df_open_s.data['Close'] = df_open_s.data['Close'] / df_open_s.data['Open'][0]
        df_open_s.data['Close'].plot(ax=ax, x_compat=True, color='orange', label='SPs')
        df_open_m.data['Close'] = df_open_m.data['Close'] / df_open_m.data['Open'][0]
        df_open_m.data['Close'].plot(ax=ax, x_compat=True, color='black', label='SPm')

        # Plot the MA
        if plotMA5s:
            df_open_s.data['ma5'] = df_open_s.data['ma5'] / df_open_s.data['Open'][0]
            df_open_s.data['ma5'].plot(ax=ax, x_compat=True, color='purple', label='MA(5s)')
            df_open_s.data['ma10'] = df_open_s.data['ma10'] / df_open_s.data['Open'][0]
            df_open_s.data['ma10'].plot(ax=ax, x_compat=True, color='purple', ls='--', label='MA(10s)')

        df_open_m.data['ma5'] = df_open_m.data['ma5'] / df_open_m.data['Open'][0]
        df_open_m.data['ma5'].plot(ax=ax, x_compat=True, color='blue', label='MA(5)')
        df_open_m.data['ma10'] = df_open_m.data['ma10'] / df_open_m.data['Open'][0]
        df_open_m.data['ma10'].plot(ax=ax, x_compat=True, color='blue', ls='--', label='MA(10)')
        ax.legend(loc='upper left', prop={'size': 6})
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        for label in ax.get_xticklabels():
            label.set_rotation(0)
        ax.get_xaxis().set_visible(False)

        # Plot the rsi
        ax = plt.subplot2grid((plot_size[0], plot_size[1]), (3, 0), colspan=10, rowspan=1)  # plot at row 3 column 1:9
        ax.get_xaxis().set_visible(False)
        df_open_s.data['rsi'].plot(ax=ax, linewidth=line_width, ls='--', label='RSI')
        df_open_m.data['rsi'].plot(ax=ax, linewidth=line_width, label='RSI')
        ax.axhline(80, color='green', ls='--')
        ax.axhline(20, color='red', ls='--')
        ax.set_ylim(0, 100)
        ax.set_ylabel('RSI', fontsize=10)
        plot_signal_lines(ax, signals, line_width)

        # Plot the STOCH 14
        ax = plt.subplot2grid((plot_size[0], plot_size[1]), (4, 0), colspan=10, rowspan=1)  # plot at row 4 column 1:9
        df_open_s.data['stoch_k'].plot(color='black', linewidth=line_width, label='k', ls='--')
        df_open_s.data['stoch_d'].plot(color='red', linewidth=line_width, label='d', ls='--')
        df_open_m.data['stoch_k'].plot(color='black', linewidth=line_width, label='k')
        df_open_m.data['stoch_d'].plot(color='red', linewidth=line_width, label='d')
        ax.set_ylim(0, 100)
        ax.set_ylabel('STOCH', fontsize=10)
        ax.legend(loc='upper left', prop={'size': 6})
        plot_signal_lines(ax, signals, line_width)

        fig.tight_layout()

        print("{} PreMarket Close {} Volume {}".format(day.strftime('%Y-%m-%d'), df_pre_close, df_pre_vol))
        if signals is not None:
            signal_str = "trade_"
        else:
            signal_str = ""
        filename = "wdc_30mins/{}wdc_30mins_{}.jpg".format(signal_str, day.strftime('%Y-%m-%d'))
        fig.savefig(filename)
        plt.close()

def daily_plots(df, days, stop=10, signals=None, thread_count=1):
    start_time = time.time()
    if thread_count > 1:
        with Pool(processes=thread_count) as pool:
            for i in range(days.size):
                dates_day_s = pd.date_range(days[i] + pd.DateOffset(hours=4), days[i] + pd.DateOffset(hours=stop), freq='S')
                df_day_s = df.daterange(dates_day_s)
                if days[i].strftime('%Y-%m-%d') in signals.index:
                    signals_day = signals.loc[days[i].strftime('%Y-%m-%d')]
                else:
                    signals_day = None
                pool.apply_async(plot_day, args=(df_day_s, days[i], stop, signals_day))
            pool.close()
            pool.join()
    else:
        for i in range(days.size):
            dates_day_s = pd.date_range(days[i] + pd.DateOffset(hours=4), days[i] + pd.DateOffset(hours=stop), freq='S')
            df_day_s = df.daterange(dates_day_s)
            if days[i].strftime('%Y-%m-%d') in signals.index:
                signals_day = signals.loc[days[i].strftime('%Y-%m-%d')]
            else:
                signals_day = None
            plot_day(df_day_s, days[i], stop, signals_day)
    print("--- %s seconds ---" % (time.time() - start_time))

def signal_plot(day, close, ma5, buysell, preds, openIndex):
    fig = plt.figure(frameon=False, figsize=(8, 4), dpi=100)
    fig.set_dpi(140)
    fig.set_size_inches(60, 15)
    ax = fig.add_subplot(111)
    ax.plot(close)
    ax.plot(ma5, ls='--')
    ax2 = ax.twinx()
    ax2.plot(buysell, color='green', ls='--')
    ax2.plot(preds, color='green')
    plt.axhline(y=0.0, ls='--', color='grey')
    plt.axvline(x=openIndex, ls='--', color='grey')
    filename = "wdc_buysell/wdc_pred_{}.jpg".format(day.strftime('%Y-%m-%d'))
    fig.savefig(filename)
    plt.close()

def signal_day(df_day_s, day, stop=10, resolution='M'):
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
        if resolution is 'S':
            df_open = df_open_s
        else:
            df_open = df_open_m
        openIndex = df_open.data.index.get_loc(day.strftime('%Y-%m-%d') + ' 09:30:00')
        print("{} pre_close={} pre_vol={}".format(day.strftime('%Y-%m-%d'), df_pre_close, df_pre_vol))
        signal_plot(day, df_open.data['Close'].values, df_open.data['ma5'].values, df_open.data['buysell'].values, df_open.data['preds'].values, openIndex)

def signal_gains(df_day_s, day, stop=10, openThreshold=0.5, closeThreshold=0.0):
    print("Day {} ".format(day.strftime('%Y-%m-%d')))
    dates_open_s = pd.date_range(day + pd.DateOffset(hours=9.5), day + pd.DateOffset(hours=stop), freq='S')
    df_open_s = df_day_s.daterange(dates_open_s, ohlcvOnly=False)
    spIn = 0
    timeIn = None
    predsIn = 0
    totalGain = 0
    for row in df_open_s.data.itertuples():
        if np.isnan(row.preds) == False:
            if spIn == 0:
                if abs(row.preds) > openThreshold:
                    spIn = row.Open
                    timeIn = row.Index
                    predsIn = row.preds
            else:
                spOut = row.Open
                timeOut = row.Index
                predsOut = row.preds
                gain = (spOut / spIn) - 1.0
                if predsIn > 0 and predsOut <= closeThreshold:
                    totalGain = totalGain + gain
                    print("  Long  Trade In {} SP={:.2f} - Out {} SP={:.2f} Gain={:.2f}%".format(timeIn.strftime('%H:%M:%S'), spIn,
                                                                                        timeOut.strftime('%H:%M:%S'), spOut,
                                                                                        gain*100))
                    spIn = 0
                elif predsIn < 0 and predsOut >= closeThreshold:
                    totalGain = totalGain - gain
                    print("  Short Trade In {} SP={:.2f} - Out {} SP={:.2f} Gain={:.2f}%".format(timeIn.strftime('%H:%M:%S'), spIn,
                                                                                                timeOut.strftime('%H:%M:%S'), spOut,
                                                                                                -gain * 100))
                    spIn = 0
    if totalGain != 0:
        print("  Gain={:.2f}%".format(totalGain * 100))
    return totalGain

print("Using:", matplotlib.get_backend())

simulate_trades = True
generate_buysell = False

ohlcv_file = "../wdcdata/wdc_ohlcv_1_year_2016.csv"

df = data.ohlcv_csv(ohlcv_file)
df.fill_gaps()

print("loaded data types:\n" + str(df.data.dtypes))
print(df.data.index)
print(df.data.dtypes)
print(df.data)

df_days = df.resample()
df_days.compute_ma()
df_days.data['Close'].plot()
plt.close()

if simulate_trades is True:
    preds = pd.read_csv(ohlcv_file + ".preds", header=0, index_col=0, parse_dates=True, infer_datetime_format=True)
    buysell = pd.read_csv(ohlcv_file + ".buysell", header=0, index_col=0, parse_dates=True, infer_datetime_format=True)

    days = pd.date_range(preds.index[0], preds.index[-1], freq='1D')
    days = days.normalize()
    totalGain = 0
    for i in range(days.size):
        dates_day_s = pd.date_range(days[i] + pd.DateOffset(hours=4), days[i] + pd.DateOffset(hours=16), freq='S')
        df_day_s = df.daterange(dates_day_s)
        df_day_s.data = df_day_s.data.join(buysell)
        df_day_s.data = df_day_s.data.join(preds)
        results = signal_day(df_day_s, days[i], 16, 'S')
    #    totalGain = totalGain + signal_gains(df_day_s, days[i], 16)
    print("Total Gain={:.2f}%".format(totalGain * 100))

if generate_buysell is True:
    threads = 6
    days = pd.date_range(df.data.index[0], df.data.index[-1], freq='1D')
    days = days.normalize()
    buysell = buysell_signal.signal()
    signal = buysell.from_df(df, days, stop=16, thread_count=threads)
    signal.to_csv(ohlcv_file + ".buysell")
    #signals_from_df(df, days, stop=16, thread_count=threads)
    #daily_plots(df, days, stop=16, signals=signal_results, thread_count=threads)
    #signal_results.to_csv("wdc_ohlcv_1_year_signals.csv")

pass