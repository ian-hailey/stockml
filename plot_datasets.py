import data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from datasets import dataset
from sklearn.model_selection import train_test_split
import cProfile as profile
from data_generator import data_generator

test1day = False
testall = True
doplots = False

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
    fig = plt.figure(frameon=False, figsize=(8, 4), dpi=100)
    canvas_width, canvas_height = fig.canvas.get_width_height()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 540)

    for m in range(3600):
        x_state, y_state = data.get_next_second()
        ax.set_xlim(0, x_state.shape[0])
        plt.plot(x_state[:,1])
        plt.plot(x_state[:,2])
        plt.plot(x_state[:,3])
#        plot_ohlcv(ax, state)
        plt.axvline(x=240, ls='--', color='grey')
        plt.axvline(x=480, ls='--', color='grey')
        plt.figtext(0.1, 0, "Y={}".format(y_state))
        fig.tight_layout()
        plt.autoscale(tight=True)
        filename = "data/wdc_n_{}_{}.jpg".format(data.day.strftime('%Y-%m-%d'), m)
        fig.savefig(filename)
        plt.cla()

print("Using:", matplotlib.get_backend())

# load OHLCV
df = data.ohlcv_csv("../wdcdata/wdc_ohlcv_1_year_2016.csv")

# resample data to include all seconds
df = df.resample(period='1s')

# load buysell data
buysell = pd.read_csv("../wdcdata/wdc_ohlcv_1_year_2016.csv.buysell", header=0, index_col=0, parse_dates=True, infer_datetime_format=True)

# merge the two
df.data = df.data.join(buysell)

print("loaded data types:\n" + str(df.data.dtypes))
print(df.data.index)
print(df.data.dtypes)
print(df.data)

data = dataset(df)
print(data.get_date_range())
error = data.select_day(dayDate='2017-04-26')
#error = data.select_day(dayDate='2018-04-25')

# filter out all the pre-post market
pre_time = 4.0
start_time = 9.5
end_time = 16.0
day_range = data.get_date_range()
buysell_range = buysell.between_time(start_time='09:30', end_time='16:00')
buysell_range = buysell_range[str(day_range[0].date()):str(day_range[1].date())]
buysell_day = buysell_range.resample('1d', fill_method=None).sum()
datetime_index = np.empty((len(buysell_day)*(int((end_time-start_time)*3600)+1), 2), dtype=int)
id_index = 0
for day in range(len(buysell_day)):
    if buysell_day.values[day][0] != 0.0:
        for sec in range(int((end_time-start_time) * 3600)+1):
            datetime_index[id_index] = (day, int((start_time-pre_time)*3600)+sec)
            id_index = id_index + 1
datetime_index = np.resize(datetime_index, (id_index, 2))
datetime_index_train, datetime_index_validate = train_test_split(datetime_index, stratify=None, test_size=0.20)

if doplots is True:
    start_time = time.time()
    save_plots(data)
    print("--- %s seconds ---" % (time.time() - start_time))

if testall is True:
    training_generator = data_generator(datetime_index, data, batch_size=9000)

    pr.enable()
    start_time = time.time()
    for m in range(training_generator.__len__()):
        X, y = training_generator.__getitem__(m)
    print("--- %s seconds ---" % (time.time() - start_time))
    pr.disable()
    pr.dump_stats('profile.pstat')

if test1day is True:
    pr.enable()
    start_time = time.time()
    for m in range(23400):
        x_state, y_state = data.get_next_second()
        if x_state is not None:
            x_state3d = x_state.reshape(9, 60, 8)
    #        print("x_state3d={} y_state={}".format(x_state3d.shape, y_state))
        else:
            print("x_state is None")
    pr.disable()
    print("--- %s seconds ---" % (time.time() - start_time))
    pr.dump_stats('profile.pstat')

#    data.plot_day_2d()
#    state3d = state.reshape(6, 60, 5)

pass