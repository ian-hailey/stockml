import data
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import time

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

class dataset(object):
    def __init__(self, df):
        self.error = False
        self.data = df
        self.day_index = 0
        self.sec_index = 0
        df_days = pd.date_range(df.data.index[0], df.data.index[-1], freq='1D')
        df_days = df_days.normalize()
        if df_days.size > 240:
            self.days = pd.date_range(df_days[0] + 240, df_days[-1], freq='1D')
            self.days = self.days.normalize()
            self.data_d = self.data.resample(period='1d')
        else:
            self.error = True

    def set_date_range(self, begin, end):
        self.days = pd.date_range(self.days[begin] + 240, self.days[end], freq='1D')
        self.days = self.days.normalize()

    def get_date_range(self):
        return self.days[0], self.days[-1]

    def reset_day_index(self):
        self.day_index = 0

    def increment_day_index(self):
        self.day_index = self.day_index + 1

    def plot_day_2d(self):
        state = self.get_next_second()
        plt.close()
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 360)
        plot_ohlcv(ax, state)
        fig.tight_layout()
        plt.autoscale(tight=True)

    def select_day(self, day=None, start=4.0, stop=16.0):
        error = False
        if day is None and self.day_index < self.days.size:
            self.day = self.days[self.day_index]
        else:
            self.day_index = self.days.get_loc(day)
            self.day = self.days[self.day_index]
        if self.day is not None:
            dates_day_s = pd.date_range(self.day + pd.DateOffset(hours=start), self.day + pd.DateOffset(hours=stop), freq='S')
            self.data_day_s = self.data.daterange(dates_day_s, ohlcvOnly=False)
            self.data_day_m = self.data_day_s.resample(period='60s')
            dates_day = pd.date_range(self.day - pd.DateOffset(days=240), self.day - pd.DateOffset(days=1), freq='1D')
            self.data_day_d = self.data_d.daterange(dates_day)
            self.reset_sec_index()
        else:
            error = True
        return error

    def reset_sec_index(self, time='09:30:00'):
        self.sec_index = self.data_day_s.data.index.get_loc(self.day.strftime('%Y-%m-%d') + ' ' + time)
        self.sec_offset = self.sec_index

    def get_next_second(self):
        x_state = None
        y_state = self.data_day_s.data.values[self.sec_index][5]
        if np.isnan(y_state) == False:
            minute_index = int(self.sec_index / 60)
            # add last 240 days
            x_state = self.data_d.data.values[self.day_index:self.day_index+240, :]
            # add last 240 minutes
            x_state = np.concatenate((x_state, self.data_day_m.data.values[minute_index - 240:minute_index, :]), axis=0)
            # add last 60 seconds
            x_state = np.concatenate((x_state, self.data_day_s.data.values[self.sec_index-59:self.sec_index+1,:5]), axis=0)
            open = self.data_day_s.data.values[self.sec_index][0]
            x_state[:,:4] = x_state[:,:4] / open
            # add time stamp planes
            tmonth = np.full((x_state.shape[0], 1), int(self.day.strftime('%m')))
            tday = np.full((x_state.shape[0], 1), int(self.day.strftime('%d')))
            tsec = np.full((x_state.shape[0], 1), self.sec_index - self.sec_offset)
            x_state = np.append(x_state, tmonth, 1)
            x_state = np.append(x_state, tday, 1)
            x_state = np.append(x_state, tsec, 1)
            open = self.data_day_s.data.values[self.sec_index][0]
        self.sec_index = self.sec_index + 1
        return x_state, y_state

def save_plots(data):
    fig = plt.figure(frameon=False, figsize=(8, 4), dpi=100)
    canvas_width, canvas_height = fig.canvas.get_width_height()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 360)

    for m in range(3600):
        x_state, y_state = data.get_next_second()
        ax.set_xlim(0, x_state.shape[0])
        plt.plot(x_state[:,1])
        plt.plot(x_state[:,2])
        plt.plot(x_state[:,3])
#        plot_ohlcv(ax, state)
        plt.figtext(0.1, 0, "Y={}".format(y_state))
        fig.tight_layout()
        plt.autoscale(tight=True)
        filename = "data/wdc_{}_{}.jpg".format(data.day.strftime('%Y-%m-%d'), m)
        fig.savefig(filename)
        plt.cla()

print("Using:", matplotlib.get_backend())

# load OHLCV
df = data.ohlcv_csv("../ib/wdc_ohlcv_1_year.csv")
df.fill_gaps()

# load buysell data
buysell = pd.read_csv("wdc_ohlcv_1_year_buysell.csv", header=0, index_col=0, parse_dates=True, infer_datetime_format=True)

# merge the two
df.data = df.data.join(buysell)

print("loaded data types:\n" + str(df.data.dtypes))
print(df.data.index)
print(df.data.dtypes)
print(df.data)

data = dataset(df)
print(data.get_date_range())
error = data.select_day('2018-04-25')

start_time = time.time()
#save_plots(data)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
for m in range(23400):
    x_state, y_state = data.get_next_second()
    if x_state is not None:
        x_state3d = x_state.reshape(9, 60, 8)
        print("x_state3d={} y_state={}".format(x_state3d.shape, y_state))
print("--- %s seconds ---" % (time.time() - start_time))

#    data.plot_day_2d()
#    state3d = state.reshape(6, 60, 5)

pass