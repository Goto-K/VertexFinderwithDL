import keras.backend as K
from keras.layers import Dense, Activation, Concatenate, Dot
from keras.layers import Input, Embedding, concatenate, RepeatVector, Dense, Reshape
from keras.layers import (Conv1D, Dense, Activation, MaxPooling1D, Add,
                          Concatenate, Bidirectional, GRU, Dropout,
                          BatchNormalization, Lambda, Dot, Multiply)
import keras.backend as K
import keras.initializers as k_init

import numpy as np
import tensorflow as tf
from keras.models import Model, model_from_json
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import TensorBoard, Callback
from tensorflow import keras


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

from time import gmtime, strftime
import os
import pathlib




def get_attention_context(encoder_output, attention_rnn_output):
    attention_input = Concatenate(axis=-1)([encoder_output,
                                            attention_rnn_output])
    e = Dense(10, activation="tanh")(attention_input)
    energies = Dense(1, activation="relu")(e)
    attention_weights = Activation('softmax')(energies)
    context = Dot(axes=1)([attention_weights,
                           encoder_output])

    return context


def attention_test_model():

    BATCH_SIZE = 32
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)
    #OPTIMIZER = keras.optimizers.SGD(lr=0.001)

    set_dir_name='TEST_VLSTM_ATTENTION_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    pair_input = Input(shape=(44,))
    track_input = Input(shape=(None, INPUT_DIM))

#=========================================================================#
#==== ENCODER ============================================================#
#=========================================================================#
    pair_input = Input(shape=(44,))
    encoder_input = Input(shape=(None, INPUT_DIM)) 

    encoder = Dense(256)(encoder_input)
    encoder = Activation('relu')(encoder)
    encoder = Dropout(0.2)(encoder)
    attention_rnn_input = GRU(256, return_sequences=True)(encoder)

#=========================================================================#
#==== DECODER ============================================================#
#=========================================================================#
    decoder_input = Input(shape=(None, INPUT_DIM)) 

    decoder = Dense(256)(decoder_input)
    decoder = Activation('relu')(decoder)
    decoder = Dropout(0.2)(decoder)
    attention_rnn_output = GRU(256)(decoder)
    attention_rnn_output_repeated = RepeartVector(INPUT_DIM)(attention_rnn_output)
    # (num_samples, features) -> (num_samples, num_samples(tracks), features)
    attention = ATTENTION.get_attention_context(attention_rnn_input, attention_rnn_output_repeated)

    shape1 = int(attention.shape[1])
    shape2 = int(attention.shape[2])

    attention_rnn_output_reshaped = Reshape((shape1, shape2))(attention_rnn_output)

    input_of_decoder_rnn = concatenate([attention, attention_rnn_output_reshaped])
    input_of_decoder_rnn_projected = Dense(256)(input_of_decoder_rnn)

    output_of_decoder_rnn = GRU(256, return_sequences=True)(input_rnn_projected)

