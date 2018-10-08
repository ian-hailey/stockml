import data
import resnet
import time
import pandas as pd
import numpy as np
from datasets import dataset

# load OHLCV
df = data.ohlcv_csv("../ib/wdc_ohlcv_1_year.csv")

# load buysell data
buysell = pd.read_csv("wdc_ohlcv_1_year_buysell.csv", header=0, index_col=0, parse_dates=True, infer_datetime_format=True)

# merge the two
df.data = df.data.join(buysell)

print("loaded data types:\n" + str(df.data.dtypes))
print(df.data.index)
print(df.data.dtypes)
print(df.data)

hist_days=240
hist_mins=240
hist_secs=60
z_dim = int((hist_days + hist_mins + hist_secs) / 60)

data = dataset(df, hist_days=hist_days, hist_mins=hist_mins, hist_secs=hist_secs)
data.select_day(dayIndex=0)

print(data.get_date_range())
print("daysize={} daysecs={}".format(data.get_day_size(), data.get_seconds_remain()))

y_train = np.zeros([data.get_day_size()*data.get_seconds_remain()])
x_train = np.zeros([data.get_day_size()*data.get_seconds_remain(), z_dim, 60, data.get_feature_size()])
row_index = 0
start_time = time.time()
for day in range(data.get_day_size()):
    error = data.select_day(dayIndex=day)
    while data.get_seconds_remain() != 0:
        x_state, y_state = data.get_next_second()
        if x_state is not None:
            x_state3d = x_state.reshape(z_dim, 60, data.get_feature_size())
            x_train[row_index] = x_state3d
            y_train[row_index] = y_state
            row_index += 1
    print("dayIndex={} row_index={} y_train={}".format(day, row_index, y_train.size))
#print("x_state3d={} y_state={}".format(x_state3d.shape, y_state))
print("--- %s seconds ---" % (time.time() - start_time))
pass