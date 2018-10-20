import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

class dataset_day(object):
    def __init__(self, data, day, start, stop):
        self.day = day
        self.date_month = int(day.strftime('%m'))
        self.date_day = int(day.strftime('%d'))
        dates_day_s = pd.date_range(self.day + pd.DateOffset(hours=start), self.day + pd.DateOffset(hours=stop), freq='S')
        self.data_s = data.daterange(dates_day_s, ohlcvOnly=False)
        self.data_m = self.data_s.resample(period='60s')
        self.day_open_s = self.data_s.data.index.get_loc(day.strftime('%Y-%m-%d') + ' 09:30:00')
        self.open = self.data_s.data.values[self.day_open_s][0]
        self.data_s.data.iloc[:, 0:4] /= self.open
        self.data_m.data.iloc[:, 0:4] /= self.open
        self.data_s_values = self.data_s.data.values
        self.data_m_values = self.data_m.data.values
        pass

class dataset(object):
    def __init__(self, df, hist_days=240, hist_mins=240, hist_secs=60, start=4.0, stop=16.0):
        self.error = False
        self.data = df
        self.day_index = 0
        self.sec_index = 0
        self.hist_days = hist_days
        self.hist_mins = hist_mins
        self.hist_secs = hist_secs
        self.day_data = []
        self.feature_size = 5 + 3 # OHLCV + tm + td + ts
        df_days = pd.date_range(df.data.index[0], df.data.index[-1], freq='1D')
        df_days = df_days.normalize()
        if df_days.size > self.hist_days:
            self.days = pd.date_range(df_days[0] + self.hist_days, df_days[-1], freq='1D')
            self.days = self.days.normalize()
            self.data_d = self.data.resample(period='1d')
            self.data_d_values = self.data_d.data.values
            for day in range(self.days.size):
                self.day_data.append(dataset_day(self.data, self.days[day], start, stop))
        else:
            self.error = True

    def set_date_range(self, begin, end):
        self.days = pd.date_range(self.days[begin] + self.hist_days, self.days[end], freq='1D')
        self.days = self.days.normalize()

    def get_date_range(self):
        return self.days[0], self.days[-1]

    def get_day_size(self):
        return self.days.size

    def reset_day_index(self):
        self.day_index = 0

    def increment_day_index(self):
        self.day_index = self.day_index + 1

    def select_day(self, dayDate=None, dayIndex=None):
        error = False
        if dayDate is None and self.day_index < self.days.size:
            self.day = self.days[self.day_index]
        elif dayIndex is not None:
            self.day_index = dayIndex
            self.day = self.days[self.day_index]
        else:
            self.day_index = self.days.get_loc(dayDate)
            self.day = self.days[self.day_index]
        if self.day is None:
            error = True
        else:
            self.sec_index = self.day_data[self.day_index].day_open_s
        return error

    def get_seconds_remain(self):
        return self.day_data[self.day_index].data_s.data.__len__() - self.sec_index

    def get_feature_size(self):
        return self.feature_size

    def get_second(self, day_index, sec_index, train=True):
        x_index = 0
        x_state = np.empty((self.hist_days+self.hist_mins+self.hist_secs, self.feature_size))
        if train:
            y_state = self.day_data[day_index].data_s_values[sec_index][5]
        else:
            y_state = None
        if np.isnan(y_state) == False:
            minute_index = int(sec_index / 60)
            # add last 240 days
            x_state[x_index:self.hist_days, 0:5] = self.data_d_values[day_index:day_index+self.hist_days, :] / self.day_data[day_index].open
            x_index = x_index + self.hist_days
            # add last 240 minutes
            x_state[x_index:x_index+self.hist_mins, 0:5] = self.day_data[day_index].data_m_values[minute_index - self.hist_mins:minute_index, :]
            # add last 60 seconds
            x_index = x_index + self.hist_mins
            x_state[x_index:x_index+self.hist_secs, 0:5] = self.day_data[day_index].data_s_values[sec_index-(self.hist_secs-1):sec_index+1,:5]
            # add time stamp planes
            x_state[:, 5:6] = self.day_data[day_index].date_month
            x_state[:, 6:7] = self.day_data[day_index].date_day
            x_state[:, 7:8] = sec_index-self.day_data[day_index].day_open_s
        return x_state, y_state

    def get_next_second(self):
        x_state, y_state = self.get_second(self.day_index, self.sec_index)
        self.sec_index = self.sec_index + 1
        return x_state, y_state
