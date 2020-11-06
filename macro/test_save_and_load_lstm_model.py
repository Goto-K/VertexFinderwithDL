from keras.utils.generic_utils import get_custom_objects

import tensorflow as tf 
import numpy as np 
from tensorflow.python.keras import backend as K
from tensorflow import keras
from tensorflow.keras.layers import AbstractRNNCell, Concatenate, Dot
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, LSTMCell, _caching_device, _generate_zero_filled_state_for_cell
from tensorflow.keras.layers import Input, Dense, RNN, LSTM, Activation, TimeDistributed, Masking, BatchNormalization, Reshape, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import activations, constraints, initializers, regularizers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import constant_op, ops
from tensorflow.python.ops import array_ops, math_ops, clip_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.keras import backend_config
from tensorflow.python.training.tracking import data_structures

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")




if __name__ == "__main__":

    NODE_SIZE = 32

    init_data = np.zeros([100, NODE_SIZE])
    lstm_data = np.zeros([100, 10, NODE_SIZE])
    labels = np.zeros([100, 10, 1])

    init_input = Input(shape=(NODE_SIZE,), name='Init_Input')
    lstm_input = Input(shape=(None, NODE_SIZE), name='LSTM_Input')
    x = TimeDistributed(Dense(NODE_SIZE, name='Embedding_Dense'))(lstm_input)

    init_state = Dense(NODE_SIZE, name='Init_Dense')(init_input)
    init_state = BatchNormalization(name='Init_BatchNorm')(init_state)
    init_state = Activation('relu', name='Init_Activation')(init_state)

    #cell = LSTMCell(NODE_SIZE)
    #x = RNN(cell, return_sequences=True, name='LSTM',)(x, initial_state=[init_state, init_state])

    x = LSTM(NODE_SIZE, return_sequences=True, name='LSTM',)(x, initial_state=[init_state, init_state])
    x = TimeDistributed(Dense(1, name='Last_Dense'))(x)

    model = Model(inputs=[init_input, lstm_input], outputs=x)

    model.summary()

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=["accuracy"])


    history = model.fit([init_data, lstm_data], labels,
                        batch_size=1,
                        epochs=1,
                        verbose=1,
                        validation_split=0.2)

    tf.saved_model.save(model, "savetest_as_pb")

    load_model = tf.saved_model.load("savetest_as_pb/")
    #load_model.summary()
    infer = loaded.signatures["serving_default"]
    print(infer.structured_outputs)
    load_model.evaluate([init_data, lstm_data], labels)




