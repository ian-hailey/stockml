import db
import symbols
import ohlcv
import tresnet
import time
import pandas as pd
import numpy as np
import sys
import getopt
import os
from datasets import Dataset
from data_generator import data_generator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# files
symbol_name = None
saved_model = None
batch_size = 2000

try:
    opts, args = getopt.getopt(sys.argv[1:],"hp:s:",["saved_model="])
except getopt.GetoptError:
    print('model.py -p<weights>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print("model.py -s<symbol> -p<weights>")
        sys.exit()
    elif opt == '-s':
        symbol_name = arg
    elif opt == '-p':
        saved_model = arg
    elif opt == '-b':
        batch_size = int(arg)

if saved_model != None:
    print("Prediction Mode: for {} model {} batch_size={}".format(symbol_name, saved_model, batch_size))
    train = False
else:
    print("Training Mode for {} batch_size={}".format(symbol_name, batch_size))
    train = True

# Setup the dataset generator params
hist_days=240
hist_mins=240
hist_secs=60
pre_time = 4.0
start_time = 9.5
end_time = 16.0

# Connect to DB
dba = db.Db(host='192.168.88.1')
symbol = symbols.Symbol(dba, symbol_name)

# Create dataset instance
data = Dataset(symbol=symbol, end_date=pd.to_datetime('2018-12-31'), num_days=60, hist_conf=(hist_days, hist_mins, hist_secs))
data.select_day(dayIndex=0)
day_range = data.get_date_range()
day_size = data.get_day_size()
feature_planes = data.get_feature_size()
external_size = data.get_external_size()

print(day_range)
print("daysize={} daysecs={}".format(day_size, data.get_seconds_remain()))

datetime_index = np.empty((day_size*(int((end_time-start_time)*3600)+1), 2), dtype=int)
secs = np.arange(start=int((start_time - pre_time) * 3600), stop=int((end_time - pre_time) * 3600) + 1)
id_index = 0
for day in range(day_size):
    datetime_index[id_index:id_index + secs.__len__(), 0] = day
    datetime_index[id_index:id_index + secs.__len__(), 1] = secs
    id_index = id_index + secs.__len__()
datetime_index_train, datetime_index_validate = train_test_split(datetime_index, stratify=None, test_size=0.20)

# build the resnet model
model = tresnet.TResnetBuilder.build_tresnet_18([(feature_planes, hist_days),
                                                 (feature_planes, hist_mins),
                                                 (feature_planes, hist_secs)], external_size, 1)
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])

if train:
    # data generators
    training_generator = data_generator(datetime_index_train, data, batch_size=batch_size)
    validation_generator = data_generator(datetime_index_validate, data, batch_size=batch_size)
    # fit callbacks
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('tresnet18_wdc.csv')

    # create folder for weights
    subfolder = os.path.splitext(os.path.basename(data.get_id()))[0]
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # checkpoint
    filepath=subfolder + "/tresnet18-{epoch:02d}-{val_loss:.2f}.hdf5"
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
    predict_generator = data_generator(datetime_index, data, batch_size=batch_size, shuffle=False, train=False)
    # load weights
    model.load_weights(saved_model)
    # predict from dataset
    results = model.predict_generator(generator=predict_generator, steps=int(np.ceil(len(datetime_index) / batch_size)), verbose=1, max_q_size=10)
    print(results)
    resultsall = results[:datetime_index.shape[0]]
    datetime_df = pd.DataFrame()
    datetime_df['preds'] = resultsall
    datetime_df.to_csv(data.get_id() + ".preds")
pass