import ohlcv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from datasets import Dataset
from sklearn.model_selection import train_test_split
import cProfile as profile
import db
import symbols

pr = profile.Profile()
pr.disable()

def plot_ohlcv(ax, ohlcv, width=0.2, colorup='green', colordown='red', alpha=1.0):
    wickWidth = max(0.5, width / 5)
    open = ohlcv[0][0]
    t0 = 0.00
    t1 = 1.00
    toffset = (t1 - t0) * 0.15
    tdelta = (t1 - t0) * 0.70
    tmid = ((t1 - t0) / 2)

    for index in range(ohlcv.shape[0]):
        row_open = (ohlcv[index][0] / open) - 1.0
        row_high = (ohlcv[index][1] / open) - 1.0
        row_low = (ohlcv[index][2] / open) - 1.0
        row_close = (ohlcv[index][3] / open) - 1.0

        if row_close >= row_open:
            color = colorup
            lower = row_open
            upper = row_close
        else:
            color = colordown
            lower = row_close
            upper = row_open

        if row_high > upper:
            vlineWick = Line2D(
                xdata=(index + tmid, index + tmid), ydata=(upper, row_high),
                color=color,
                linewidth=wickWidth,
                antialiased=True,
            )
            vlineWick.set_alpha(alpha)
            ax.add_line(vlineWick)

        if row_low < lower:
            vlineWick = Line2D(
                xdata=(index + tmid, index + tmid), ydata=(row_low, lower),
                color=color,
                linewidth=wickWidth,
                antialiased=True,
            )
            vlineWick.set_alpha(alpha)
            ax.add_line(vlineWick)

        rect = Rectangle(
            xy=(index + toffset, lower),
            width=tdelta,
            height=upper - lower,
            facecolor=color,
            edgecolor=color,
        )
#        print("x={} y={} height={}".format(index + toffset, lower, upper - lower))
        rect.set_alpha(alpha)
        ax.add_patch(rect)

def plot_day_2d(data):
    state = data.get_next_second()
    plt.close()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 360)
    plot_ohlcv(ax, state)
    fig.tight_layout()
    plt.autoscale(tight=True)

def save_plots(data):
    fig = plt.figure(frameon=False, figsize=(8, 4), dpi=200)
    canvas_width, canvas_height = fig.canvas.get_width_height()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 540)

    for m in range(int((6.5*3600)/60)):
        x_state, y_state = data.get_second(0, m*60)
        ax.set_xlim(0, data.hist_days + data.hist_mins + data.hist_secs)
        x = np.empty((data.hist_days + data.hist_mins + data.hist_secs, data.feature_size))
        x[0:data.hist_days,] = x_state[0]
        x[data.hist_days:data.hist_days+data.hist_mins,] = x_state[1]
        x[data.hist_days+data.hist_mins:data.hist_days+data.hist_mins+data.hist_secs,] = x_state[2]
        plt.plot(x[:,1])
        plt.plot(x[:,2])
        plt.plot(x[:,3])
#        plot_ohlcv(ax, state)
        plt.axvline(x=240, ls='--', color='grey')
        plt.axvline(x=480, ls='--', color='grey')
        plt.figtext(0.1, 0, "Y={:.3f}".format(y_state))
        fig.tight_layout()
        plt.autoscale(tight=True)
        filename = "data/wdc_n_{}_{}.jpg".format(data.day.strftime('%Y-%m-%d'), m)
        fig.savefig(filename)
        plt.clf()

print("Using:", matplotlib.get_backend())

end_date = '2018-12-31'
num_days = 1
symbol = 'WDC'
sql_host = "192.168.88.1"
dba = db.Db(host=sql_host)
symbol = symbols.Symbol(dba, 'WDC', autocreate=True)
data = Dataset(symbol=symbol, end_date=pd.to_datetime(end_date), num_days=num_days, normalise=True)
data.select_day(day_index=0)
day_range = data.get_date_range()
day_size = data.get_day_size()

# filter out all the pre-post market
pre_time = 4.0
start_time = 9.5
end_time = 16.0

start_time = time.time()
save_plots(data)
print("--- %s seconds ---" % (time.time() - start_time))

pass