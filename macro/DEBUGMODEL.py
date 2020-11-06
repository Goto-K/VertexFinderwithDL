import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, RNN, Activation, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import TensorBoard, Callback
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
import CUSTOM

def lstm_model(pair, tracks, labels, NB_EPOCHS, NODE_SIZE):

    BATCH_SIZE = 32
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)
    #OPTIMIZER = keras.optimizers.SGD(lr=0.001)

    set_dir_name='LSTM_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    pair_input = Input(shape=(44,))
    track_input = Input(shape=(None, 22))

    init_state1 = Dense(NODE_SIZE, activation='relu')(pair_input)
    init_state2 = Dense(NODE_SIZE, activation='relu')(pair_input)

    cell = CUSTOM.VLSTMCell(NODE_SIZE)

    x = RNN(cell, return_sequences=True)(track_input, initial_state=[init_state1, init_state2])
    x = TimeDistributed(Dense(1))(x)
    x = Activation('sigmoid', name='output')(x)
    
    model = Model(inputs=[pair_input, track_input], outputs=x)

    model.summary()

    # Tensor Board
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=OPTIMIZER, 
                  metrics=['accuracy'])

    callbacks = [TensorBoard(log_dir=log_dir)]
    
    history = model.fit([pair, tracks], labels,
                        batch_size=BATCH_SIZE, 
                        epochs=NB_EPOCHS, 
                        callbacks=callbacks,
                        verbose=VERBOSE, 
                        validation_split=VALIDATION_SPLIT)
    
    return model, history

def lstm_dense_model(pair, tracks, labels, NB_EPOCHS, NODE_SIZE=32):

    BATCH_SIZE = 32
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)
    #OPTIMIZER = keras.optimizers.SGD(lr=0.001)

    set_dir_name='LSTM_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    pair_input = Input(shape=(44,))
    track_input = Input(shape=(None, 22))

    init_state1 = Dense(NODE_SIZE, activation='relu')(pair_input)
    #init_state2 = Dense(1, activation='relu')(pair_input)
    #init_state2 = tf.keras.backend.zeros((1,))

    cell = CUSTOM.VLSTMCell_Dense(NODE_SIZE, 1)

    x = RNN(cell, return_sequences=True)(track_input, initial_state=[init_state1, init_state1])
    
    model = Model(inputs=[pair_input, track_input], outputs=x)

    model.summary()

    # Tensor Board
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=OPTIMIZER, 
                  metrics=['accuracy'])

    callbacks = [TensorBoard(log_dir=log_dir)]
    
    history = model.fit([pair, tracks], labels,
                        batch_size=BATCH_SIZE, 
                        epochs=NB_EPOCHS, 
                        callbacks=callbacks,
                        verbose=VERBOSE, 
                        validation_split=VALIDATION_SPLIT)
    
    return model, history


def lstm_dense_plus_model(pair, tracks, labels, NB_EPOCHS, NODE_SIZE=32):

    BATCH_SIZE = 32
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)
    #OPTIMIZER = keras.optimizers.SGD(lr=0.001)

    set_dir_name='LSTM_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    pair_input = Input(shape=(44,))
    track_input = Input(shape=(None, 32))

    init_state1 = Dense(NODE_SIZE, activation='relu')(pair_input)
    #init_state2 = Dense(1, activation='relu')(pair_input)
    #init_state2 = tf.keras.backend.zeros((1,))

    cell = CUSTOM.VLSTMCell_Dense(NODE_SIZE, 1)

    x = RNN(cell, return_sequences=True)(track_input, initial_state=[init_state1, init_state1])
    
    model = Model(inputs=[pair_input, track_input], outputs=x)

    model.summary()

    # Tensor Board
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=OPTIMIZER, 
                  metrics=['accuracy'])

    callbacks = [TensorBoard(log_dir=log_dir)]
    
    history = model.fit([pair, tracks], labels,
                        batch_size=BATCH_SIZE, 
                        epochs=NB_EPOCHS, 
                        callbacks=callbacks,
                        verbose=VERBOSE, 
                        validation_split=VALIDATION_SPLIT)
    
    return model, history


def stacked_lstm_model(pair, tracks, labels, NB_EPOCHS):

    BATCH_SIZE = 32
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)
    #OPTIMIZER = keras.optimizers.SGD(lr=0.001)

    set_dir_name='LSTM_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    pair_input = Input(shape=(44,))
    track_input = Input(shape=(None, 22))

    init_state1 = Dense(32, activation='relu')(pair_input)
    init_state2 = Dense(32, activation='relu')(pair_input)

    cell = [CUSTOM.VLSTMCell(32), CUSTOM.VLSTMCell(32)]

    x = RNN(cell, return_sequences=True)(track_input, initial_state=[init_state1, init_state2])
    x = TimeDistributed(Dense(1))(x)
    x = Activation('sigmoid', name='output')(x)
    
    model = Model(inputs=[pair_input, track_input], outputs=x)

    #model = Sequential([
    #RNN(CUSTOM.SLSTMCell(22), input_shape=(None, 22), return_sequences=True)
    #])

    model.summary()

    # Tensor Board
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=OPTIMIZER, 
                  metrics=['accuracy'])

    callbacks = [TensorBoard(log_dir=log_dir)]
    
    history = model.fit([pair, tracks], labels,
                        batch_size=BATCH_SIZE, 
                        epochs=NB_EPOCHS, 
                        callbacks=callbacks,
                        verbose=VERBOSE, 
                        validation_split=VALIDATION_SPLIT)
    
    return model, history

