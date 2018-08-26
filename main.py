import data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib.dates import DateFormatter

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

for i in range(days.size):
    # create date time index for the pre-market
    dates_pre = pd.date_range(days[i] + pd.DateOffset(hours=4), days[i] + pd.DateOffset(hours=9.5), freq='S')
    df_pre = df.daterange(dates_pre)
    df_pre_close = df_pre.data['Close'].iloc[-1]
    df_pre_vol = df_pre.data['Volume'].sum()
    if df_pre_vol:
        # create date time index for the first 30 minutes of market second resolution
        dates_open30s = pd.date_range(days[i] + pd.DateOffset(hours=9.5), days[i] + pd.DateOffset(hours=10), freq='S')
        df_open30 = df.daterange(dates_open30s)
        df_open30.compute_rsi()
        df_open30.compute_stoch()
        df_open30.compute_ma(time=5)
        df_open30.compute_ma(time=10)

        # create down samples minute resolution and include premarket
        dates_morning = pd.date_range(days[i] + pd.DateOffset(hours=4), days[i] + pd.DateOffset(hours=10), freq='S')
        df_morning = df.daterange(dates_morning)
        df_morning_mins = df_morning.resample('60s')
        df_morning_mins.compute_rsi()
        df_morning_mins.compute_stoch()
        df_morning_mins.compute_ma(time=5)
        df_morning_mins.compute_ma(time=10)

        # create down samples minute resolution first 30 minutes
        dates_open30m = pd.date_range(days[i] + pd.DateOffset(hours=9.5), days[i] + pd.DateOffset(hours=10), freq='60S')
        df_open30m = df_morning_mins.daterange(dates_open30m, ohlcvOnly=False)

        fig = plt.figure(1)
#        fig.set_dpi(300)
#        fig.set_size_inches(100, 15)
        line_width = 1.0
        plot_size = (5, 10) # 3 rows 10 columns
        gridspec.GridSpec(plot_size[0], plot_size[1])

        # Plot the candles
        ax = plt.subplot2grid((plot_size[0], plot_size[1]), (0, 0), colspan=10, rowspan=3) # plot at row 0 column 1:9
        df_open30m.plot_candlestick(ax)
#        candlestick_ohlc(ax, df_open30m, width=10.0, alpha=0.2)
        # Plot the MA
        df_open30m.data['ma5'].plot(ax=ax, x_compat=True, color='blue')
        df_open30m.data['ma10'].plot(ax=ax, x_compat=True, color='blue', ls='--')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        for label in ax.get_xticklabels():
            label.set_rotation(0)
        ax.get_xaxis().set_visible(False)

        # Plot the volume
        ax2 = ax.twinx()
        df_open30m.plot_volume_bars(ax2)

        # Plot the rsi
        ax = plt.subplot2grid((plot_size[0], plot_size[1]), (3, 0), colspan=10, rowspan=1) # plot at row 3 column 1:9
        ax.get_xaxis().set_visible(False)
        df_open30m.data['rsi'].plot(ax=ax, linewidth=line_width)
        ax.axhline(80, color='green', ls='--')
        ax.axhline(20, color='red', ls='--')
        ax.set_ylim(0,100)

        # Plot the STOCH 14
        ax = plt.subplot2grid((plot_size[0], plot_size[1]), (4, 0), colspan=10, rowspan=1) # plot at row 4 column 1:9
        df_open30m.data['stoch_k'].plot(color='black', linewidth=line_width)
        df_open30m.data['stoch_d'].plot(color='red', linewidth=line_width)
        ax.set_ylim(0,100)

        fig.tight_layout()

        print("{} PreMarket Close {} Volume {}".format(days[i].strftime('%Y-%m-%d'), df_pre_close, df_pre_vol))
        filename = "wdc_30mins/wdc_30mins_{}.jpg".format(days[i].strftime('%Y-%m-%d'))
        fig.savefig(filename)
        plt.close()

pass