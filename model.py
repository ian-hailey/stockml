import data
import resnet
import time
import pandas as pd
import numpy as np
from datasets import dataset
from data_generator import data_generator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split

# load OHLCV
df = data.ohlcv_csv("../ib/wdc_ohlcv_1_year.csv")

# resample data to include all seconds
df = df.resample(period='1s')

# load buysell data
buysell = pd.read_csv("wdc_ohlcv_1_year_buysell.csv", header=0, index_col=0, parse_dates=True, infer_datetime_format=True)

# merge the two
df.data = df.data.join(buysell)

print("loaded data types:\n" + str(df.data.dtypes))
print(df.data.index)
print(df.data.dtypes)
print(df.data)

# setup the dataset generator
hist_days=240
hist_mins=240
hist_secs=60
y_dim = int((hist_days + hist_mins + hist_secs) / 60)

data = dataset(df, hist_days=hist_days, hist_mins=hist_mins, hist_secs=hist_secs)
data.select_day(dayIndex=0)
day_range = data.get_date_range()

feature_planes = data.get_feature_size()

print(day_range)
print("daysize={} daysecs={}".format(data.get_day_size(), data.get_seconds_remain()))

# filter out all the pre-post market
buysell_day = buysell.between_time(start_time='09:30', end_time='16:00')
buysell_day = buysell_day[str(day_range[0].date()):str(day_range[1].date())]

buysell_day_train, buysell_day_validate = train_test_split(buysell_day, stratify=None, test_size=0.20)

# build the resnet model
model = resnet.ResnetBuilder.build_resnet_18((feature_planes, y_dim, 60), 1)
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])

# data generators
batch_size = 8000
training_generator = data_generator(buysell_day_train.index, data, (y_dim, 60, feature_planes), batch_size=batch_size)
validation_generator = data_generator(buysell_day_validate.index, data, (y_dim, 60, feature_planes), batch_size=batch_size)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_wdc.csv')

# train model on dataset
model.fit_generator(generator=training_generator,
                    steps_per_epoch=buysell_day_train.size // batch_size,
                    validation_data=validation_generator,
                    validation_steps=buysell_day_validate.size // batch_size,
#                    use_multiprocessing=True,
#                    workers=6,
                    epochs=200, verbose=1, max_q_size=100,
                    callbacks = [lr_reducer, early_stopper, csv_logger])

y_train = np.zeros([data.get_day_size()*data.get_seconds_remain()])
x_train = np.zeros([data.get_day_size()*data.get_seconds_remain(), y_dim, 60, feature_planes])
row_index = 0
start_time = time.time()
for day in range(data.get_day_size()):
    error = data.select_day(dayIndex=day)
    while data.get_seconds_remain() != 0:
        x_state, y_state = data.get_next_second()
        if x_state is not None:
            x_state3d = x_state.reshape(y_dim, 60, feature_planes)
            x_train[row_index] = x_state3d
            y_train[row_index] = y_state
            row_index += 1
    print("dayIndex={} row_index={} y_train={}".format(day, row_index, y_train.size))
#print("x_state3d={} y_state={}".format(x_state3d.shape, y_state))
print("--- %s seconds ---" % (time.time() - start_time))
pass