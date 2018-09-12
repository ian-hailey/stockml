import data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time
from matplotlib.dates import DateFormatter
from multiprocessing import Pool


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

    def select_day(self, day=None, start=4.0, stop=16.0):
        error = False
        if day is None and self.day_index < self.days.size:
            self.day = self.days[self.day_index]
        else:
            self.day = day
        if self.day is not None:
            dates_day_s = pd.date_range(self.day + pd.DateOffset(hours=start), self.day + pd.DateOffset(hours=stop), freq='S')
            self.data_day_s = self.data.daterange(dates_day_s)
            self.data_day_m = self.data_day_s.resample(period='60s')
            dates_day = pd.date_range(self.day - pd.DateOffset(days=240), self.day - pd.DateOffset(days=1), freq='1D')
            self.data_day_d = self.data_d.daterange(dates_day)
            self.day_index = self.day_index + 1
            self.reset_sec_index()
        else:
            error = True
        return error

    def reset_sec_index(self, time='09:00:00'):
        self.sec_index = self.data_day_s.data.index.get_loc(self.day.strftime('%Y-%m-%d') + ' ' + time)

    def get_next_second(self):
        for s in range(60):
            print("{},{},{},{},{}".format(self.data_day_s.data['Open'].iloc[self.sec_index],
                                          self.data_day_s.data['High'].iloc[self.sec_index],
                                          self.data_day_s.data['Low'].iloc[self.sec_index],
                                          self.data_day_s.data['Close'].iloc[self.sec_index],
                                          self.data_day_s.data['Volume'].iloc[self.sec_index]))
            self.sec_index = self.sec_index + 1


print("Using:", matplotlib.get_backend())

df = data.ohlcv_csv("../ib/wdc_ohlcv_1_year.csv")
df.fill_gaps()

print("loaded data types:\n" + str(df.data.dtypes))
print(df.data.index)
print(df.data.dtypes)
print(df.data)

data = dataset(df)
print(data.get_date_range())
error = data.select_day()
for m in range(60):
    data.get_next_second()

pass