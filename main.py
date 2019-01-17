import ohlcv
import signals
import symbols
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time
import numpy as np
import sys
import getopt
import db
from matplotlib.dates import DateFormatter
from multiprocessing import Pool, Lock
from scipy.signal import argrelextrema
from datasets import Dataset

def plot_signal_lines(ax, signals, line_width):
    if signals is not None:
        for signal in signals.itertuples():
            plt.axvline(x=signal.Index, linewidth=line_width, color='grey')
            plt.axvline(x=signal.t_out, linewidth=line_width, ls='--', color='grey')

def plot_day(df_day_s, day, stop=10, signals=None, plotMA5s=False):
    # create date time index for the pre-market
    dates_pre = pd.date_range(day + pd.DateOffset(hours=4), day + pd.DateOffset(hours=9.5), freq='S')
    df_pre = df_day_s.daterange(dates_pre)
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
    filename = "wdc_buysell/wdc_pred_2017_{}.jpg".format(day.strftime('%Y-%m-%d'))
    fig.savefig(filename)
    plt.close()

def log_print(log_str, log_file=None):
    if log_file is None:
        print(log_str)
    else:
        log_file.write(log_str + '\n')

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

def signal_gains(df_day, day, stop=10, openThreshold=0.02, closeThreshold=0.0, log_file=None):
    logstr = "Day {} ".format(day.strftime('%Y-%m-%d'))
    log_print(logstr, log_file)
    dates_open_s = pd.date_range(day + pd.DateOffset(hours=9.5), day + pd.DateOffset(hours=stop), freq='S')
    df_open_s = df_day.daterange(dates_open_s, ohlcvOnly=False)
    spIn = 0
    timeIn = None
    predIn = 0
    totalGain = 0
    for row in df_open_s.data.itertuples():
        if np.isnan(row.pred) == False:
            if spIn == 0:
                if abs(row.pred) > openThreshold:
                    spIn = row.Open
                    timeIn = row.Index
                    predIn = row.pred
            else:
                spOut = row.Open
                timeOut = row.Index
                predOut = row.pred
                gain = (spOut / spIn) - 1.0
                if predIn > 0 and predOut <= closeThreshold:
                    totalGain = totalGain + gain
                    logstr = "  Long  Trade In {} SP={:.2f} - Out {} SP={:.2f} Gain={:.2f}%".format(timeIn.strftime('%H:%M:%S'), spIn,
                                                                                        timeOut.strftime('%H:%M:%S'), spOut,
                                                                                        gain*100)
                    log_print(logstr, log_file)
                    spIn = 0
                elif predIn < 0 and predOut >= closeThreshold:
                    totalGain = totalGain - gain
                    logstr = "  Short Trade In {} SP={:.2f} - Out {} SP={:.2f} Gain={:.2f}%".format(timeIn.strftime('%H:%M:%S'), spIn,
                                                                                                timeOut.strftime('%H:%M:%S'), spOut,
                                                                                                -gain * 100)
                    log_print(logstr, log_file)
                    spIn = 0
    if totalGain != 0:
        logstr = "  Gain={:.2f}%".format(totalGain * 100)
        log_print(logstr, log_file)
    return totalGain

print("Using:", matplotlib.get_backend())

enddate = None
num_days = 0
symbol = None
sql_host = "192.168.88.1"

try:
    opts, args = getopt.getopt(sys.argv[1:],"hc:o:p:s:e:d:",["preds=","symbol=", "enddate=", "days="])
except getopt.GetoptError:
    print('model.py -p<weights> -s<symbol> -e<end date> -c<number of days>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print("model.py -p<weights> -s<symbol> -e<end date> -c<number of days>")
        sys.exit()
    elif opt == '-s':
        symbol = arg
    elif opt == '-e':
        end_date = pd.to_datetime(arg)
    elif opt == '-d':
        num_days = int(arg)
    elif opt == '-p':
        preds_file = arg
    elif opt == '-o':
        start_time = pd.to_datetime(arg)
    elif opt == '-c':
        end_time = pd.to_datetime(arg)

if end_date is not None and symbol is not None and num_days is not 0:
    dba = db.Db(host=sql_host)
    symbol = symbols.Symbol(dba, 'WDC', autocreate=True)
    data = Dataset(symbol=symbol, end_date=pd.to_datetime(end_date), num_days=num_days, normalise=False)
    data.select_day(day_index=0)
    day_range = data.get_date_range()
    day_size = data.get_day_size()
else:
    print("Error: command line args")
    sys.exit(2)

print("Pediction File " + preds_file)
preds = pd.read_csv(preds_file, header=0, index_col=0, parse_dates=True, infer_datetime_format=True)

gains_file = open('gains_' + data.get_id() + '.txt', "w")
totalGain = 0
dayGains = []
for day_index in range(day_size):
    day_date = data.day_data[day_index].day.strftime('%Y-%m-%d')
    df_day = data.get_day(day_index=day_index)
    df_day.data['pred'] = preds[day_date]
    df_day.data['pred'] = df_day.data['pred'].fillna(0)
    gain = signal_gains(df_day, data.day_data[day_index].day, 16, log_file=gains_file)
    dayGains.append(gain)
    totalGain = totalGain + gain
log_print("Total Gain={:.2f}%".format(totalGain * 100), gains_file)
gains_file.close()
xs = np.arange(0, len(dayGains))
plt.bar(xs, dayGains)
plt.xlabel('day')
plt.ylabel('gains')
plt.savefig('gainbars_' + data.get_id())
plt.cla()
plt.hist(dayGains)
plt.xlabel('gains')
plt.ylabel('days')
plt.savefig('gainhist_' + data.get_id())

pass