import data
import resnet
import time
import pandas as pd
import numpy as np
import sys
import getopt
from datasets import dataset
from data_generator import data_generator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# files
#ohlcv_file = "../wdcdata/wdc_ohlcv_1_year.csv"
#buysell_file = "../wdcdata/wdc_ohlcv_1_year_buysell.csv"
ohlcv_file = "../wdcdata/wdc_ohlcv_1_year_2016.csv"
buysell_file = None
saved_model = None

try:
    opts, args = getopt.getopt(sys.argv[1:],"hp:",["saved_model="])
except getopt.GetoptError:
    print('model.py -p<weights>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print("model.py -p<weights>")
        sys.exit()
    elif opt == '-p':
        saved_model = arg

if saved_model != None:
    print("Prediction Mode: Input file is {}".format(saved_model))
    train = False
else:
    print("Training Mode")
    train = True

# Setup the dataset generator params
hist_days=240
hist_mins=240
hist_secs=60
pre_time = 4.0
start_time = 9.5
end_time = 16.0
batch_size = 9000

# load OHLCV
df = data.ohlcv_csv(ohlcv_file)

# Resample data to include all seconds
df = df.resample(period='1s')

if buysell_file is not None:
    # load buysell data
    buysell = pd.read_csv(buysell_file, header=0, index_col=0, parse_dates=True, infer_datetime_format=True)
    # merge the two
    df.data = df.data.join(buysell)

print("loaded data types:\n" + str(df.data.dtypes))
print(df.data.index)
print(df.data.dtypes)
print(df.data)

y_dim = int((hist_days + hist_mins + hist_secs) / 60)

# Create dataset instance
data = dataset(df, hist_days=hist_days, hist_mins=hist_mins, hist_secs=hist_secs)
data.select_day(dayIndex=0)
day_range = data.get_date_range()
day_size = data.get_day_size()
data_days = data.data.resample()
feature_planes = data.get_feature_size()

print(day_range)
print("daysize={} daysecs={}".format(day_size, data.get_seconds_remain()))

datetime_index = np.empty((day_size*(int((end_time-start_time)*3600)+1), 2), dtype=int)
datetime_df = pd.DataFrame()
id_index = 0
for day in range(day_size):
    if data_days.data['Volume'][day_range[0] + pd.DateOffset(days=day)] != 0.0:
        datetime_day = pd.date_range(day_range[0] + pd.DateOffset(days=day) + pd.DateOffset(hours=start_time),
                                     day_range[0] + pd.DateOffset(days=day) + pd.DateOffset(hours=end_time), freq='S')
        datetime_day_df = pd.DataFrame(index=datetime_day)
        datetime_df = pd.concat([datetime_df, datetime_day_df])
        for sec in range(int((end_time-start_time) * 3600)+1):
            datetime_index[id_index] = (day, int((start_time - pre_time) * 3600) + sec)
            id_index = id_index + 1
datetime_index = np.resize(datetime_index, (id_index, 2))
datetime_index_train, datetime_index_validate = train_test_split(datetime_index, stratify=None, test_size=0.20)

# build the resnet model
model = resnet.ResnetBuilder.build_resnet_18((feature_planes, y_dim, 60), 1)
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])

if train:
    # data generators
    training_generator = data_generator(datetime_index_train, data, (y_dim, 60, feature_planes), batch_size=batch_size)
    validation_generator = data_generator(datetime_index_validate, data, (y_dim, 60, feature_planes), batch_size=batch_size)
    # fit callbacks
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('resnet18_wdc.csv')

    # checkpoint
    filepath="weights-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

    # train model on dataset
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=len(datetime_index_train) // batch_size,
                        validation_data=validation_generator,
                        validation_steps=len(datetime_index_validate) // batch_size,
    #                    use_multiprocessing=True,
    #                    workers=6,
                        epochs=10, verbose=1, max_q_size=10,
                        callbacks = [lr_reducer, early_stopper, csv_logger, checkpoint])
else:
    # data generator
    predict_generator = data_generator(datetime_index, data, (y_dim, 60, feature_planes), batch_size=batch_size, shuffle=False, train=False)
    # load weights
    model.load_weights(saved_model)
    # predict from dataset
    results = model.predict_generator(generator=predict_generator, steps=int(np.ceil(len(datetime_index) / batch_size)), verbose=1, max_q_size=10)
    print(results)
    resultsall = results[:datetime_index.shape[0]]
    datetime_df['preds'] = resultsall
    datetime_df.to_csv(ohlcv_file + ".preds")
pass