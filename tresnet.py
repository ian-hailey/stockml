from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    concatenate,
    add
)
from keras.layers.convolutional import (
    Conv1D, Conv2D,
    MaxPooling1D,
    AveragePooling1D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.utils.vis_utils import plot_model
from contextlib import redirect_stdout


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv1D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    period_length = int(round(input_shape[PERIOD_AXIS] / residual_shape[PERIOD_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 x conv if shape is different. Else identity.
    if period_length > 1 or not equal_channels:
        shortcut = Conv1D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=1,
                          strides=period_length,
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = 1
            if i == 0 and not is_first_layer:
                init_strides = 2
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=1, is_first_block_of_first_layer=False):
    """Basic x 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv1D(filters=filters, kernel_size=3,
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=3,
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=3)(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=1, is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1 = Conv1D(filters=filters, kernel_size=1,
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1 = _bn_relu_conv(filters=filters, kernel_size=1,
                                     strides=init_strides)(input)

        conv_3 = _bn_relu_conv(filters=filters, kernel_size=3)(conv_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=1)(conv_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global PERIOD_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        PERIOD_AXIS = 1
        CHANNEL_AXIS = 2
    else:
        CHANNEL_AXIS = 1
        PERIOD_AXIS = 2


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class TResnetBuilder(object):
    @staticmethod
    def compile(input_shape, num_outputs, block_fn, repetitions):
        """Compiles a custom temporal ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_periods)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            input: The input to the network
            output: The last output layer
        """
        _handle_dim_ordering()
        if len(input_shape) != 2:
            raise Exception("Input shape should be a tuple (nb_channels, nb_periods)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=7, strides=2)(input)
        pool1 = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling1D(pool_size=(block_shape[PERIOD_AXIS]),
                                 strides=1)(block)

        output = Flatten()(pool2)

        return input, output

    @staticmethod
    def build(input_shapes, external_shape, num_outputs, block_fn, repetitions):
        """Compiles a custom temporal ResNet like architecture.

        Args:
            input_shapes: The list input shape in the form (nb_channels, nb_periods)
            external_shape: The size of the external features one hot encoded
            num_outputs: The number of outputs at final softmax layer, 0=Omit final Dense layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """

        # main input
        main_inputs = []
        outputs = []

        for input_shape in input_shapes:
            # Compile the ResNets
            input, output = TResnetBuilder.compile(input_shape, num_outputs, block_fn, repetitions)
            main_inputs.append(input)
            outputs.append(output)

        # concatenate all nets together
        main_output = concatenate(outputs)

        # fuse with external features
        if external_shape != None and external_shape > 0:
            external_input = Input(shape=(external_shape,))
            main_inputs.append(external_input)
            embedding = Dense(units=min(50, external_shape))(external_input)
            embedding = Activation('relu')(embedding)
            h1 = Dense(units=len(input_shapes)*len(repetitions)*64*2)(embedding)
            external_output = Activation('relu')(h1)
            main_output = add([main_output, external_output])

        main_output = Dense(units=num_outputs, kernel_initializer="he_normal",
                            activation="linear")(main_output)


        model = Model(inputs=main_inputs, outputs=main_output)
        plot_model(model, to_file='T_ResNet.png', show_shapes=True)
        with open('T_ResNet_Summary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        return model

    @staticmethod
    def build_tresnet_18(input_shape, external_shape, num_outputs):
        return TResnetBuilder.build(input_shape, external_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_tresnet_34(input_shape, external_shape, num_outputs):
        return TResnetBuilder.build(input_shape, external_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_tresnet_50(input_shape, external_shape, num_outputs):
        return TResnetBuilder.build(input_shape, external_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_tresnet_101(input_shape, external_shape, num_outputs):
        return TResnetBuilder.build(input_shape, external_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_tresnet_152(input_shape, external_shape, num_outputs):
        return TResnetBuilder.build(input_shape, external_shape, num_outputs, bottleneck, [3, 8, 36, 3])

