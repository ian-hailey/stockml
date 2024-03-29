import numpy as np
import talib
import pandas as pd
import psycopg2
from matplotlib.dates import date2num
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

class Ohlcv(object):
    def __init__(self, data):
        self.data = data

    def plot_candlestick(self, ax, width=0.2, colorup='green', colordown='red', alpha=1.0, relativeOpen=True):

        wickWidth = max(0.5, width / 5)
        open = self.data['Open'][0]
        t0 = date2num(self.data.index[0])
        t1 = date2num(self.data.index[1])
        toffset = (t1 - t0) * 0.15
        tdelta = (t1 - t0) * 0.70
        tmid = ((t1 - t0) / 2)

        for row in self.data.itertuples():
            t = date2num(row[0])
            if relativeOpen:
                row_open = row[1] / open
                row_high = row[2] / open
                row_low = row[3] / open
                row_close = row[4] / open
            else:
                row_open = row[1]
                row_high = row[2]
                row_low = row[3]
                row_close = row[4]

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
                    xdata=(t+tmid, t+tmid), ydata=(upper, row_high),
                    color=color,
                    linewidth=wickWidth,
                    antialiased=True,
                )
                vlineWick.set_alpha(alpha)
                ax.add_line(vlineWick)

            if row_low < lower:
                vlineWick = Line2D(
                    xdata=(t+tmid, t+tmid), ydata=(row_low, lower),
                    color=color,
                    linewidth=wickWidth,
                    antialiased=True,
                )
                vlineWick.set_alpha(alpha)
                ax.add_line(vlineWick)

            rect = Rectangle(
                xy=(t+toffset, lower),
                width=tdelta,
                height=upper-lower,
                facecolor=color,
                edgecolor=color,
            )
            rect.set_alpha(alpha)
            ax.add_patch(rect)
            ax.set_ylabel('SP (Relative to Open)', fontsize=8)

    def plot_volume_bars(self, ax, colorup='green', colordown='red', alpha=0.3, yscale=0.3):
        t0 = date2num(self.data.index[0])
        t1 = date2num(self.data.index[1])
        toffset = (t1 - t0) * 0.15
        tdelta = (t1 - t0) * 0.70

        for row in self.data.itertuples():
            t = date2num(row[0])
            open = row[1]
            close = row[4]
            volume = row[5]

            if close >= open:
                color = colorup
            else:
                color = colordown

            rect = Rectangle(
                xy=(t+toffset, 0),
                width=tdelta,
                height=volume/1000,
                facecolor=color,
                edgecolor=color,
            )
            rect.set_alpha(alpha)
            ax.add_patch(rect)
        ax.set_ylim(0, self.data['Volume'].max()/(yscale*1000))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.set_ylabel('Volume (K)', fontsize=8)

    def fill_gaps(self):
        self.data['Close'].fillna(method="ffill", inplace=True)
        self.data['Close'].fillna(method="bfill", inplace=True)
        self.data['Open'].fillna(self.data['Close'], inplace=True)
        self.data['High'].fillna(self.data['Close'], inplace=True)
        self.data['Low'].fillna(self.data['Close'], inplace=True)
        self.data['Volume'].fillna(value=0, inplace=True)


    def resample(self, period='1d', fillGaps=True):
        resample_open = self.data['Open'].resample(period).first()
        resample_high = self.data['High'].resample(period).max()
        resample_low = self.data['Low'].resample(period).min()
        resample_close = self.data['Close'].resample(period).last()
        resample_volume = self.data['Volume'].resample(period).sum()
        resample = pd.DataFrame(index=resample_open.index)
        resample['Open'] = resample_open.values
        resample['High'] = resample_high.values
        resample['Low'] = resample_low.values
        resample['Close'] = resample_close.values
        resample['Volume'] = resample_volume.values
        resample_ohlcv = Ohlcv(resample)
        if fillGaps:
            resample_ohlcv.fill_gaps()
        return resample_ohlcv

    def daterange(self, dates, ohlcvOnly=True, fillGaps=True):
        range = pd.DataFrame(index=dates)
        if ohlcvOnly:
            range = range.join(self.data.ix[:,0:5])
        else:
            range = range.join(self.data)
        range_ohlcv = Ohlcv(range)
        if fillGaps:
            range_ohlcv.fill_gaps()
        return range_ohlcv

    def compute_gradient(self, source):
        column = str(source) + 'grad'
        self.data[column] = np.gradient(np.asarray(self.data[source]))

    def compute_ma(self, time=200, sample=1, source='Close'):
        column = 'ma' + str(time)
        period = sample * time
        self.data[column] = talib.MA(np.asarray(self.data[source]), timeperiod=period, matype=0)

    def compute_macd(self, fast=12, slow=26, signal=9):
        self.data['macd'], self.data['macdsignal'], self.data['macdhist'] = talib.MACD(np.asarray(self.data['Close']), fastperiod=fast, slowperiod=slow, signalperiod=signal)

    def compute_rsi(self):
        self.data['rsi'] = talib.RSI(np.asarray(self.data['Close']))

    def compute_bb(self, time=10, devup=2, devdown=2):
        self.data['bb_upper'], self.data['bb_mid'], self.data['bb_lower'] = talib.BBANDS(np.asarray(self.data['Close']), timeperiod=time, nbdevup=devup, nbdevdn=devdown)

    def compute_stoch(self, fastk=14, slowk=3, slowd=3):
        self.data['stoch_k'], self.data['stoch_d'] = talib.STOCH(np.asarray(self.data['High']), np.asarray(self.data['Low']), np.asarray(self.data['Close']),
                                                                 fastk_period=fastk, slowk_period=slowk, slowk_matype=0, slowd_period=slowd, slowd_matype=0)

class Ohlcv_csv(Ohlcv):
    def __init__(self, csvfile):
        self.csvfile = csvfile
        self.headers = ["Date", "Open", "High", "Low", "Close", "Volume"]
        data = pd.read_csv(self.csvfile, names=self.headers, header=0, index_col=0, parse_dates=True, infer_datetime_format=True)
        super().__init__(data)

