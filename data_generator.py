import numpy as np
import keras
from sklearn.utils import shuffle

class data_generator(keras.utils.Sequence):
    def __init__(self, date_index, data, batch_size=32, shuffle=True, train=True):
        'Initialization'
        self.batch_size = batch_size
        self.data = data
        self.date_index = date_index
        self.shuffle = shuffle
        self.on_epoch_end()
        self.train = train

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.date_index) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate date_index of the batch
        batch_index = self.date_index[index*self.batch_size:(index+1)*self.batch_size]
        batch_remain = self.batch_size - len(batch_index)
        if batch_remain is not 0:
            batch_index_a = batch_index
            batch_index = np.empty([self.batch_size, 2], dtype=np.int)
            batch_index[0:len(batch_index_a), ] = batch_index_a
            batch_index[len(batch_index_a):len(batch_index_a) + batch_remain, ] = self.date_index[0:batch_remain]

        # Generate data
        X, y = self.__data_generation(batch_index, self.train)

        if self.train:
            return X, y
        else:
            return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            self.date_index = shuffle(self.date_index)

    def __data_generation(self, batch_index, train):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        if train:
            y = np.empty((self.batch_size))
        else:
            y = None

        # Generate data
        for batch_item, date in enumerate(batch_index):
            x_state, y_state = self.data.get_second(date[0], date[1], train)

            # on first itteration allocate space for all the batch
            if X.__len__() == 0:
                for x_item in x_state:
                    X.append(np.empty((self.batch_size, *x_item.shape)))

            # Store batch item
            for item_index, x_item in enumerate(x_state):
                X[item_index][batch_item,] = x_item

            if train:
                # Store signal
                y[batch_item] = y_state

        return X, y