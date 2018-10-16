import numpy as np
import keras
from sklearn.utils import shuffle

class data_generator(keras.utils.Sequence):
    def __init__(self, date_index, data, dim=(32,32,32), batch_size=32, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data = data
        self.date_index = date_index
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.date_index) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate date_index of the batch
        batch_index = self.date_index[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_index)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            self.date_index = shuffle(self.date_index)

    def __data_generation(self, batch_index):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size))

        # Generate data
        for index, date in enumerate(batch_index):
            x_state, y_state = self.data.get_second(date[0], date[1])

            # Store sample
            X[index,] = x_state.reshape(self.dim)

            # Store class
            y[index] = y_state

        return X, y