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
import tensorflow as tf
from datasets import Dataset
from data_generator import data_generator
from keras.optimizers import adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split

def set_GPU_opts():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

# files
symbol_name = None
saved_model = None
end_date = None
num_days = 0
batch_size = 2000

try:
    opts, args = getopt.getopt(sys.argv[1:],"hb:e:d:s:p:",["batchsize=","enddate=","days=","symbol=","savedmodel="])
except getopt.GetoptError:
    print('model.py -p<weights> -s<symbol> -e<end date> -c<number of days> -b<batch size>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('model.py -p<weights> -s<symbol> -e<end date> -c<number of days> -b<batch size>')
        sys.exit()
    elif opt == '-s':
        symbol_name = arg
    elif opt == '-b':
        batch_size = int(arg)
    elif opt == '-e':
        end_date = pd.to_datetime(arg)
    elif opt == '-d':
        num_days = int(arg)
    elif opt == '-p':
        saved_model = arg

if end_date is not None and symbol_name is not None and num_days is not 0:
    if saved_model != None:
        print("Prediction Mode: for {} model {} batch_size={}".format(symbol_name, saved_model, batch_size))
        train = False
    else:
        print("Training Mode for {} batch_size={}".format(symbol_name, batch_size))
        train = True
else:
    print("Error: command line args")
    sys.exit(2)

# Setup the dataset generator params
hist_days=240
hist_mins=240
hist_secs=60
pre_time = pd.to_datetime('04:00:00')
start_time = pd.to_datetime('09:30:00')
end_time = pd.to_datetime('16:00:00')

# Connect to DB
dba = db.Db(host='192.168.88.1')
symbol = symbols.Symbol(dba, symbol_name)

data = Dataset(symbol=symbol, end_date=pd.to_datetime(end_date), num_days=num_days, hist_conf=(hist_days, hist_mins, hist_secs))
data.select_day(day_index=0)
day_range = data.get_date_range()
day_size = data.get_day_size()
feature_planes = data.get_feature_size()
external_size = data.get_external_size()

print(day_range)
print("daysize={} daysecs={}".format(day_size, data.get_seconds_remain()))

day_secs = int((end_time-start_time).total_seconds())+1
datetime_index = np.empty((day_size*day_secs, 2), dtype=int)
secs = np.arange(day_secs)
id_index = 0
for day in range(day_size):
    datetime_index[id_index:id_index + secs.__len__(), 0] = day
    datetime_index[id_index:id_index + secs.__len__(), 1] = secs
    id_index = id_index + secs.__len__()
datetime_index_train, datetime_index_validate = train_test_split(datetime_index, stratify=None, test_size=0.10, shuffle=False)

# configure the GPU
set_GPU_opts()

# build the resnet model
model = tresnet.TResnetBuilder.build_tresnet_18([(feature_planes, hist_days),
                                                 (feature_planes, hist_mins),
                                                 (feature_planes, hist_secs)], external_size, 1)
optimiser = adam(lr = 0.001)
model.compile(loss='mean_squared_error',
              optimizer=optimiser,
              metrics=['mse'])

if train:
    # data generators
    training_generator = data_generator(datetime_index_train, data, batch_size=batch_size)
    validation_generator = data_generator(datetime_index_validate, data, batch_size=batch_size)
    # fit callbacks
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=3)
    csv_logger = CSVLogger('trestnet18_' + data.get_id() + '.csv')
    # tensorboard callback
    tensorboard = TensorBoard(log_dir='./graph', histogram_freq=0,
                              write_graph=True, write_images=True)
    # create folder for weights
    subfolder = os.path.splitext(os.path.basename(data.get_id()))[0]
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # checkpoint
    filepath=subfolder + "/tresnet18-{epoch:02d}-{val_loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

    # train model on dataset
    history = model.fit_generator(generator=training_generator,
                        steps_per_epoch=len(datetime_index_train) // batch_size,
                        validation_data=validation_generator,
                        validation_steps=len(datetime_index_validate) // batch_size,
    #                    use_multiprocessing=True,
    #                    workers=6,
                        epochs=20, verbose=1, max_q_size=10,
                        callbacks = [lr_reducer, early_stopper, csv_logger, checkpoint, tensorboard])
    print(history)
    pass
else:
    # data generator
    predict_generator = data_generator(datetime_index, data, batch_size=batch_size, shuffle=False, train=False)
    # load weights
    model.load_weights(saved_model)
    # predict from dataset
    results = model.predict_generator(generator=predict_generator, steps=int(np.ceil(len(datetime_index) / batch_size)), verbose=1, max_q_size=10)
    print(results)
    dateindex = data.get_index(start_time, end_time)
    resultsall = results[:datetime_index.shape[0]]
    datetime_df = pd.DataFrame(index=dateindex)
    datetime_df.index.name = 'datetime'

    datetime_df['zc'] = resultsall
    datetime_df.to_csv(data.get_id() + ".preds")
    pass
