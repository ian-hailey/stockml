import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#matplotlib.use('Agg')
from matplotlib.dates import DateFormatter

import ohlcv

df_day_m = ohlcv.Ohlcv_csv("../mu1day18.csv")
df_day_m.data = df_day_m.data.sort_index()
df_day_m.compute_rsi()
df_day_m.compute_stoch()
df_day_m.compute_ma(time=5, source='Close')
df_day_m.compute_ma(time=10, source='Close')

fig = plt.figure(1)
fig.set_dpi(140)
fig.set_size_inches(60, 15)
line_width = 1.0
plot_size = (5, 10)  # 3 rows 10 columns
gridspec.GridSpec(plot_size[0], plot_size[1])

# Plot the candles
ax = plt.subplot2grid((plot_size[0], plot_size[1]), (0, 0), colspan=10, rowspan=3)  # plot at row 0 column 1:9
df_day_m.plot_candlestick(ax, relativeOpen=False)

# Plot the volume
ax2 = ax.twinx()
df_day_m.plot_volume_bars(ax2)

df_day_m.data['Close'] = df_day_m.data['Close']
df_day_m.data['Close'].plot(ax=ax, x_compat=True, color='black', label='SPm')

# Plot the MA
df_day_m.data['ma5'] = df_day_m.data['ma5']
df_day_m.data['ma5'].plot(ax=ax, x_compat=True, color='purple', label='MA(5s)')
df_day_m.data['ma10'] = df_day_m.data['ma10']
df_day_m.data['ma10'].plot(ax=ax, x_compat=True, color='purple', ls='--', label='MA(10s)')

df_day_m.data['ma5'] = df_day_m.data['ma5']
df_day_m.data['ma5'].plot(ax=ax, x_compat=True, color='blue', label='MA(5)')
df_day_m.data['ma10'] = df_day_m.data['ma10']
df_day_m.data['ma10'].plot(ax=ax, x_compat=True, color='blue', ls='--', label='MA(10)')
ax.legend(loc='upper left', prop={'size': 6})
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
for label in ax.get_xticklabels():
    label.set_rotation(0)
ax.get_xaxis().set_visible(False)

# Plot the rsi
ax = plt.subplot2grid((plot_size[0], plot_size[1]), (3, 0), colspan=10, rowspan=1)  # plot at row 3 column 1:9
ax.get_xaxis().set_visible(False)
df_day_m.data['rsi'].plot(ax=ax, linewidth=line_width, label='RSI')
ax.axhline(80, color='green', ls='--')
ax.axhline(20, color='red', ls='--')
ax.set_ylim(0, 100)
ax.set_ylabel('RSI', fontsize=10)

# Plot the STOCH 14
ax = plt.subplot2grid((plot_size[0], plot_size[1]), (4, 0), colspan=10, rowspan=1)  # plot at row 4 column 1:9
df_day_m.data['stoch_k'].plot(color='black', linewidth=line_width, label='k')
df_day_m.data['stoch_d'].plot(color='red', linewidth=line_width, label='d')
ax.set_ylim(0, 100)
ax.set_ylabel('STOCH', fontsize=10)
ax.legend(loc='upper left', prop={'size': 6})

fig.tight_layout()

plt.show()

fig.savefig("ohlcv_test.jpg")