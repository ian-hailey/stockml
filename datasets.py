import ohlcv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import mpu.ml as ml

class Dataset_day(object):
    def __init__(self, day, time_active, symbol, normalise):
        self.day = day
        self.date_month = int(day.strftime('%m'))
        self.date_day = int(day.strftime('%d'))
        self.date_wday = day.weekday()
        dates_day_s = pd.date_range(self.day + pd.DateOffset(hours=time_active[0]), self.day + pd.DateOffset(hours=time_active[1]), freq='S')
        self.signal = symbol.db.get_symbol_signals(symbol.symbol, [day, day + pd.DateOffset(days=1)])
        self.signal_values = self.signal.values
        self.signal_values = self.signal_values.reshape(self.signal_values.shape[0], )
        self.data_s = symbol.db.get_symbol_ohlcv(symbol.symbol, [day, day + pd.DateOffset(days=1)])
        self.data_s = ohlcv.Ohlcv(self.data_s)
        self.data_s = self.data_s.resample(period='1s')
        self.data_s = self.data_s.daterange(dates_day_s)
        self.data_s = self.data_s.daterange(dates_day_s, ohlcvOnly=False)
        self.data_m = self.data_s.resample(period='60s')
        self.day_open_s = self.data_s.data.index.get_loc(day.strftime('%Y-%m-%d') + ' 09:30:00')
        self.open = self.data_s.data.values[self.day_open_s][0]
        if normalise:
            self.data_s.data.iloc[:, 0:4] /= self.open
            self.data_m.data.iloc[:, 0:4] /= self.open
        self.data_s_values = self.data_s.data.values
        self.data_m_values = self.data_m.data.values
        pass

class Dataset(object):
    def __init__(self, symbol, end_date, num_days, hist_conf=(240, 240, 60), time_active=(4.0, 16.0), normalise=True):
        self.error = False
        self.symbol = symbol
        self.day_index = 0
        self.sec_index = 0
        self.hist_days, self.hist_mins, self.hist_secs = hist_conf
        self.feature_size = 5 # OHLCV
        self.external_size = 439 # month 12, day 31, week day 5, minute 391 (9:30 - 16:00)
        self.id = "{}_{}_{}d_{}hd_{}hm_{}hs".format(symbol.symbol, end_date.strftime('%Y_%m_%d'), num_days, hist_conf[0], hist_conf[1], hist_conf[2])
        self.day_data = []
        self.data_d = self.symbol.db.get_symbol_ohlcv(self.symbol.symbol, [end_date + pd.DateOffset(days=1)], num_days+self.hist_days+1, resolution='days')
        if self.data_d.shape[0] == self.hist_days + num_days + 1:
            self.data_d_values = self.data_d.values
            for day_index in range(num_days):
                self.day_data.append(Dataset_day(self.data_d.iloc[day_index+self.hist_days+1].name, time_active, self.symbol, normalise))
        else:
            self.error = True

    def get_index(self, day_start_time, day_end_time):
        index = None
        day_open_time = pd.to_datetime('09:30:00')
        day_start_s = int((day_start_time - day_open_time).total_seconds())
        day_end_s = int((day_end_time - day_open_time).total_seconds()) + 1
        for day in self.day_data:
            day_open_s = self.day_data[self.day_index].day_open_s
            if index is None:
                index = day.data_s.data.index[day_open_s+day_start_s:day_open_s+day_end_s]
            else:
                index = index.append(day.data_s.data.index[day_open_s+day_start_s:day_open_s+day_end_s])
        return index

    def get_date_range(self):
        return self.day_data[0].day, self.day_data[-1].day

    def get_day_size(self):
        return self.day_data.__len__()

    def reset_day_index(self):
        self.day_index = 0

    def increment_day_index(self):
        self.day_index = self.day_index + 1

    def select_day(self, day_date=None, day_index=None):
        error = False
        if day_date is None and self.day_index < self.hist_days:
            self.day = self.day_data[self.day_index].day
        elif day_index is not None:
            self.day_index = day_index
            self.day = self.day_data[self.day_index].day
        else:
            self.day_index = self.days.get_loc(day_date)
            self.day = self.day_data[self.day_index].day
        if self.day is None:
            error = True
        self.sec_index = 0
        return error

    def get_day(self, day_index=None, resolution='secs', ohlvc_only=False):
        if day_index is None:
            day_index = self.day_index
        if resolution == 'secs':
            data = self.day_data[day_index].data_s
        else:
            data = self.day_data[day_index].data_m
        if ohlvc_only is False:
            data.data['zc'] = self.day_data[day_index].signal_values
        return data

    def get_seconds_remain(self):
        return self.day_data[self.day_index].data_s.data.__len__() - self.sec_index

    def get_feature_size(self):
        return self.feature_size

    def get_external_size(self):
        return self.external_size

    def get_id(self):
        return self.id

    def get_second(self, day_index, sec_index, train=True):
#        print("day={} sec={} month={} day={} wday={}".format(day_index, sec_index,
#                                                             self.day_data[day_index].date_month,
#                                                             self.day_data[day_index].date_day,
#                                                             self.day_data[day_index].date_wday))
        x_state = []
        sec_index = self.day_data[self.day_index].day_open_s + sec_index
        if train:
            y_state = self.day_data[day_index].signal_values[sec_index]
        else:
            y_state = None
        minute_index = int(sec_index / 60)
        # d - last 240 days
        x_state.append(self.data_d_values[day_index:day_index+self.hist_days, :] / self.day_data[day_index].open)
        # m - last 240 minutes
        x_state.append(self.day_data[day_index].data_m_values[minute_index - self.hist_mins:minute_index, :])
        # s - last 60 seconds
        x_state.append(self.day_data[day_index].data_s_values[sec_index-(self.hist_secs-1):sec_index+1,:5])
        # add external features datetime etc.
        x_external = np.empty(self.external_size)
        x_external[0:12] = ml.indices2one_hot([self.day_data[day_index].date_month-1], 12)[0]
        x_external[12:12+31] = ml.indices2one_hot([self.day_data[day_index].date_day-1], 31)[0]
        x_external[12+31:12+31+5] = ml.indices2one_hot([self.day_data[day_index].date_wday], 5)[0]
        x_external[12+31+5:12+31+5+391] = ml.indices2one_hot([int((sec_index - self.day_data[day_index].day_open_s) / 60)], 391)[0]
        x_external = x_external.reshape(self.external_size)
        x_state.append(x_external)
        return x_state, y_state

    def get_next_second(self):
        x_state, y_state = self.get_second(self.day_index, self.sec_index)
        self.sec_index = self.sec_index + 1
        return x_state, y_state
