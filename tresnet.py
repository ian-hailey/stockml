'''
    T-ResNet: Deep Temporal Residual Networks
'''

from __future__ import print_function
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Reshape
)
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils.vis_utils import plot_model


def _shortcut(input, residual):
    return merge([input, residual], mode='sum')


def _bn_relu_conv(nb_filter, size, strides=1, bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution1D(filters=nb_filter, kernel_size=size, strides=strides, padding="same")(activation)
    return f


def _residual_unit(nb_filter, size=3):
    def f(input):
        residual = _bn_relu_conv(nb_filter=nb_filter, size=size)(input)
        residual = _bn_relu_conv(nb_filter=nb_filter, size=size)(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, nb_filter, repetitions=1):
    def f(input):
        for i in range(repetitions):
            input = residual_unit(nb_filter=nb_filter)(input)
        return input
    return f


def tresnet(c_conf=(10, 5), p_conf=(10, 5), t_conf=(10, 5), external_dim=8, n_residual_unit=3):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, n_features)
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, n_features = conf
            input = Input(shape=(len_seq, n_features))
            main_inputs.append(input)
            # Conv1
            conv1 = Convolution1D(filters=64, kernel_size=3, padding="same")(input)
            # [nb_residual_unit] Residual Units
            residual_output = ResUnits(_residual_unit, nb_filter=64, repetations=n_residual_unit)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = Convolution1D(filters=1, kernel_size=3, padding="same")(activation)
            outputs.append(conv2)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        from .ilayer import iLayer
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        main_output = merge(new_outputs, mode='sum')

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(output_dim=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(output_dim=1)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape(1)(activation)
        main_output = merge([main_output, external_output], mode='sum')
    else:
        print('external_dim:', external_dim)

    main_output = Activation('tanh')(main_output)
    model = Model(input=main_inputs, output=main_output)

    return model

if __name__ == '__main__':
    model = tresnet(external_dim=28, nb_residual_unit=12)
    plot_model(model, to_file='ST-ResNet.png', show_shapes=True)
    model.summary()
