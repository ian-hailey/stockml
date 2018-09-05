import data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time
from matplotlib.dates import DateFormatter
from multiprocessing import Pool

def plot_day(df, day):
    # create date time index for the pre-market
    dates_pre = pd.date_range(day + pd.DateOffset(hours=4), day + pd.DateOffset(hours=9.5), freq='S')
    df_pre = df.daterange(dates_pre)
    df_pre_close = df_pre.data['Close'].iloc[-1]
    df_pre_vol = df_pre.data['Volume'].sum()
    if df_pre_vol:
        # create down samples minute resolution and include premarket
        dates_day_s = pd.date_range(day + pd.DateOffset(hours=4), day + pd.DateOffset(hours=10), freq='S')
        df_day_s = df.daterange(dates_day_s)
        df_day_s.compute_rsi()
        df_day_s.compute_stoch(fastk=14*60, slowd=60)
        df_day_s.compute_ma(time=5, sample=60)
        df_day_s.compute_ma(time=10, sample=60)

        df_day_m = df_day_s.resample('60s')
        df_day_m.compute_rsi()
        df_day_m.compute_stoch()
        df_day_m.compute_ma(time=5, source='Open')
        df_day_m.compute_ma(time=10, source='Open')

        # create date time index for the first 30 minutes of market second resolution
        dates_open_s = pd.date_range(day + pd.DateOffset(hours=9.5), day + pd.DateOffset(hours=10), freq='S')
        df_open_s = df_day_s.daterange(dates_open_s, ohlcvOnly=False)

        # create down samples minute resolution first 30 minutes
        dates_open_m = pd.date_range(day + pd.DateOffset(hours=9.5), day + pd.DateOffset(hours=10), freq='60S')
        df_open_m = df_day_m.daterange(dates_open_m, ohlcvOnly=False)

        fig = plt.figure(1)
        fig.set_dpi(140)
        #        fig.set_size_inches(100, 15)
        line_width = 1.0
        plot_size = (5, 10)  # 3 rows 10 columns
        gridspec.GridSpec(plot_size[0], plot_size[1])

        # Plot the candles
        ax = plt.subplot2grid((plot_size[0], plot_size[1]), (0, 0), colspan=10, rowspan=3)  # plot at row 0 column 1:9
        df_open_m.plot_candlestick(ax)

        # Plot the volume
        ax2 = ax.twinx()
        df_open_m.plot_volume_bars(ax2)

        # Plot the SP
        df_open_s.data['Close'] = df_open_s.data['Close'] / df_open_s.data['Open'][0]
        df_open_s.data['Close'].plot(ax=ax, x_compat=True, color='orange', label='SPs')
        df_open_m.data['Close'] = df_open_m.data['Close'] / df_open_m.data['Open'][0]
        df_open_m.data['Close'].plot(ax=ax, x_compat=True, color='black', label='SPs')

        # Plot the MA
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

        # Plot the STOCH 14
        ax = plt.subplot2grid((plot_size[0], plot_size[1]), (4, 0), colspan=10, rowspan=1)  # plot at row 4 column 1:9
        df_open_s.data['stoch_k'].plot(color='black', linewidth=line_width, label='k', ls='--')
        df_open_s.data['stoch_d'].plot(color='red', linewidth=line_width, label='d', ls='--')
        df_open_m.data['stoch_k'].plot(color='black', linewidth=line_width, label='k')
        df_open_m.data['stoch_d'].plot(color='red', linewidth=line_width, label='d')
        ax.set_ylim(0, 100)
        ax.set_ylabel('STOCH', fontsize=10)
        ax.legend(loc='upper left', prop={'size': 6})

        fig.tight_layout()

        print("{} PreMarket Close {} Volume {}".format(day.strftime('%Y-%m-%d'), df_pre_close, df_pre_vol))
        filename = "wdc_30mins/wdc_30mins_{}.jpg".format(day.strftime('%Y-%m-%d'))
        fig.savefig(filename)
        plt.close()

def find_signals(df_day_s, day, resolution='M'):
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
        df_day_m.compute_ma(time=5, source='Open')
        df_day_m.compute_ma(time=10, source='Open')

        # create down samples minute resolution first 30 minutes
        dates_open_s = pd.date_range(day + pd.DateOffset(hours=9.5), day + pd.DateOffset(hours=10), freq='S')
        df_open_s = df_day_s.daterange(dates_open_s, ohlcvOnly=False)
        dates_open_m = pd.date_range(day + pd.DateOffset(hours=9.5), day + pd.DateOffset(hours=10), freq='60S')
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
            if spIn == 0 and row.Close > row.Open and row_m.rsi > 70 and row.stoch_k > 70 and maDelta > 0.001:
                spIn = row.Close
                stochIn = row.stoch_k
                rsiIn = row_m.rsi
                maIn = row.ma5 - row.ma10
                timeIn = row.Index
            elif spIn != 0 and (row_m.rsi < 60 or row.stoch_k < 60 or maDelta < -0.001):
                delta = (row.Close / spIn) - 1.0
                if delta > 0.003:
                    gain = gain + delta
                    print("Trade Day {}".format(day.strftime('%Y-%m-%d')))
                    print("  In  {} maIn={:.2f} rsiIn={:.0f} stochIn={:.0f} SP={:.2f}".format(timeIn.strftime('%H:%M:%S'), maIn, rsiIn, stochIn, spIn))
                    print("  Out {} maDelta={:.2f} rsi={:.0f} stoch={:.0f} SP={:.2f} ({:.2f}%)".format(row.Index.strftime('%H:%M:%S'), row.ma5 - row.ma10, row_m.rsi, row.stoch_k, row.Close, delta*100))
                spIn = rsiSig = stochSig = 0
        if spIn != 0:
            delta = (df_open.data['Close'][-1] / spIn) - 1.0
            if delta > 0.003:
                gain = gain + delta
                print("Trade Day {}".format(day.strftime('%Y-%m-%d')))
                print("  In  {} maIn={:.2f} rsiIn={:.0f} stochIn={:.0f} SP={:.2f}".format(timeIn.strftime('%H:%M:%S'), maIn, rsiIn, stochIn, spIn))
                print("  Out {} maDelta={:.2f} rsi={:.0f} stoch={:.0f} SP={:.2f} ({:.2f}%)".format(row.Index.strftime('%H:%M:%S'), row.ma5 - row.ma10, row_m.rsi, row.stoch_k, row.Close, delta * 100))
    return [gain, day]

signal_results = []

def find_signals_callback(results):
    signal_results.append(results)

def signals_from_df(df, days):
    start_time = time.time()
    with Pool(processes=4) as pool:
        for i in range(days.size):
            dates_day = pd.date_range(days[i] + pd.DateOffset(hours=4), days[i] + pd.DateOffset(hours=10), freq='S')
            df_day_s = df.daterange(dates_day)
            pool.apply_async(find_signals, args=(df_day_s, days[i], 'M'), callback=find_signals_callback)
        pool.close()
        pool.join()

    gains = pd.DataFrame(index=days, columns=['gain', 'yoy'])
    gains['gain'].fillna(value=0.0, inplace=True)
    gains['yoy'].fillna(value=0.0, inplace=True)
    last_total = total_gains = total_losses = total = 0.0
    for [gain, day] in signal_results:
        if gain > 0.0:
            total_gains = total_gains + gain
        else:
            total_losses = total_losses + gain
        total = total + gain
        gains['gain'].loc[day] = gain * 100
        gains['yoy'].loc[day] = total * 100
        if total != last_total:
            print("total={:.2f}%".format(total * 100))
            last_total = total
    print("total={:.2f}% (gains={:.2f}% losses={:.2f}%)".format(total * 100, total_gains * 100, total_losses * 100))

    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    last_total = total_gains = total_losses = total = 0.0
    for i in range(days.size):
        dates_day = pd.date_range(days[i] + pd.DateOffset(hours=4), days[i] + pd.DateOffset(hours=10), freq='S')
        df_day_s = df.daterange(dates_day)
        [gain, day] = find_signals(df_day_s, days[i])
        if gain > 0.0:
            total_gains = total_gains + gain
        else:
            total_losses = total_losses + gain
        total = total + gain
        gains['gain'][i] = gain * 100
        gains['yoy'][i] = total * 100
        if total != last_total:
            print("total={:.2f}%".format(total * 100))
            last_total = total
    print("total={:.2f}% (gains={:.2f}% losses={:.2f}%)".format(total * 100, total_gains * 100, total_losses * 100))
    print("--- %s seconds ---" % (time.time() - start_time))
    ax = gains.plot()
    fig = ax.get_figure()
    fig.set_dpi(140)
    fig.savefig("gains.jpg")

def daily_plots(df, days):
    for i in range(days.size):
        plot_day(df, days[i])

df = data.ohlcv_csv("../ib/wdc_ohlcv_1_year.csv")
df.fill_gaps()

print("loaded data types:\n" + str(df.data.dtypes))
print(df.data.index)
print(df.data.dtypes)
print(df.data)

df_days = df.resample()
df_days.compute_ma()
df_days.data['Close'].plot()
plt.close()

days = pd.date_range(df.data.index[0], df.data.index[-1], freq='1D')
days = days.normalize()

signals_from_df(df, days)
#daily_plots(df, days)


pass