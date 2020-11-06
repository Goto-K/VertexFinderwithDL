import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, concatenate, Dense, Dropout, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten, RNN
from tensorflow.keras.models import Model, Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.vis_utils import model_to_dot 
from keras.utils import np_utils, CustomObjectScope
from keras.callbacks import TensorBoard, Callback
from keras.engine.topology import Network
from tensorflow import keras

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

from time import gmtime, strftime
import pathlib
import os
import TOOLS
import LOSS
import CUSTOM
import GRAPH


"""
'nevent', 'ntr1track', 'ntr2track', # 0 1 2

'tr1d0', 'tr1z0', 'tr1phi', 'tr1omega', 'tr1tanlam', 'tr1charge', 'tr1energy', # 3 4 5 6 7 8 9
'tr1covmatrixd0d0', 'tr1covmatrixd0z0', 'tr1covmatrixd0ph', 'tr1covmatrixd0om', 'tr1covmatrixd0tl', # 10 11 12 13 14
'tr1covmatrixz0z0', 'tr1covmatrixz0ph', 'tr1covmatrixz0om', 'tr1covmatrixz0tl', 'tr1covmatrixphph', # 15 16 17 18 19
'tr1covmatrixphom', 'tr1covmatrixphtl', 'tr1covmatrixomom', 'tr1covmatrixomtl', 'tr1covmatrixtltl', # 20 21 22 23 24

'tr2d0', 'tr2z0', 'tr2phi', 'tr2omega', 'tr2tanlam', 'tr2charge', 'tr2energy', # 25 26 27 28 29 30 31
'tr2covmatrixd0d0', 'tr2covmatrixd0z0', 'tr2covmatrixd0ph', 'tr2covmatrixd0om', 'tr2covmatrixd0tl', # 32 33 34 35 36
'tr2covmatrixz0z0', 'tr2covmatrixz0ph', 'tr2covmatrixz0om', 'tr2covmatrixz0tl', 'tr2covmatrixphph', # 37 38 39 40 41
'tr2covmatrixphom', 'tr2covmatrixphtl', 'tr2covmatrixomom', 'tr2covmatrixomtl', 'tr2covmatrixtltl', # 42 43 44 45 46

'vchi2', 'vposx', 'vposy', 'vposz', 'mass', 'mag', 'vec', 'tr1selection', 'tr2selection', 'v0selection', # 47 48 49 50 51 52 53 54 55 56
'connect', 'lcfiplustag' # 57 58
"""


def load_model(model_name):
    #model = load_model("/home/ilc/gotok/WorkDir/python/model/" + model_name + "_" + date + ".h5", compile=False)

    model_path = "/home/goto/ILC/Deep_Learning/model/" + model_name
    json_path = model_path + ".json"
    h5_path = model_path + ".h5"
    structure = pathlib.Path(json_path).read_text()

    model = model_from_json(structure, custom_objects={'custom_categorical_crossentropy': LOSS.custom_categorical_crossentropy([0.02, 0.08, 1, 0.5, 0,37])})
    model.load_weights(h5_path)
    model.summary()

    return model


def comparison_conv_model(x_train, x_image, y_train, NB_EPOCHS, NB_CLASSES=5, FILTER_SIZE=30, FIRST_CONV=1, SECOND_CONV=2, THIRD_CONV=2,
                          high=True, low=False, noconv=False):

    BATCH_SIZE = 1024
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    #OPTIMIZER = keras.optimizers.Adam(lr=0.001)
    OPTIMIZER = keras.optimizers.SGD(lr=0.001)

    set_dir_name='CONV_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    if noconv:
        if high:
            variable_input = Input(shape=(54,))
        elif low:
            variable_input = Input(shape=(44,))
        cla = Dense(256)(variable_input)
        cla = BatchNormalization()(cla) 
        cla = Activation('relu')(cla)
        cla = Dense(256)(cla)
        cla = BatchNormalization()(cla) 
        cla = Activation('relu')(cla)
        cla = Dense(256)(cla)
        cla = BatchNormalization()(cla) 
        cla = Activation('relu')(cla)
        cla = Dense(NB_CLASSES)(cla)
        vertex_output = Activation('softmax', name='vertex_output')(cla)

        model = Model(inputs=variable_input, outputs=vertex_output)

    else:
        image_input = Input(shape=(2, 200, 3)) # (tracks, t, xyz)
        if high:
            variable_input = Input(shape=(54,))
        elif low:
            variable_input = Input(shape=(44,))

        conv = Conv2D(16, (2, FILTER_SIZE))(image_input)
        conv = BatchNormalization()(conv) 
        conv = Activation('relu')(conv)
        #conv = MaxPooling2D((2, 2))(conv)
        for i in range(FIRST_CONV):
            conv = Conv2D(16, (1, FILTER_SIZE))(conv)
            conv = BatchNormalization()(conv) 
            conv = Activation('relu')(conv)
            #conv = MaxPooling2D((2, 2))(conv)
        for i in range(SECOND_CONV):
            conv = Conv2D(32, (1, FILTER_SIZE))(conv)
            conv = BatchNormalization()(conv) 
            conv = Activation('relu')(conv)
            #conv = MaxPooling2D((2, 2))(conv)
        for i in range(THIRD_CONV):
            conv = Conv2D(64, (1, FILTER_SIZE))(conv)
            conv = BatchNormalization()(conv) 
            conv = Activation('relu')(conv)
            #conv = MaxPooling2D((2, 2))(conv)

        if high or low:
            conv_out = Flatten()(conv)
            combined = concatenate([conv_out, variable_input])
            cla = Dense(256)(combined)
            cla = BatchNormalization()(cla) 
            cla = Activation('relu')(cla)
            cla = Dense(256)(cla)
            cla = BatchNormalization()(cla) 
            cla = Activation('relu')(cla)
            cla = Dense(256)(cla)
            cla = BatchNormalization()(cla) 
            cla = Activation('relu')(cla)
            cla = Dense(NB_CLASSES)(cla)
            vertex_output = Activation('softmax', name='vertex_output')(cla)

            model = Model(inputs=[image_input, variable_input], outputs=vertex_output)

        else:
            conv = Flatten()(conv)
            conv = Dense(128)(conv)
            conv = BatchNormalization()(conv) 
            conv = Activation('relu')(conv)
            conv = Dense(128)(conv)
            conv = BatchNormalization()(conv) 
            conv = Activation('relu')(conv)
            conv = Dense(128)(conv)
            conv = BatchNormalization()(conv) 
            conv = Activation('relu')(conv)
            conv = Dense(NB_CLASSES)(conv)
            vertex_output = Activation('softmax', name='vertex_output')(conv)

            model = Model(inputs=image_input, outputs=vertex_output)

    model.summary()

    # Tensor Board
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with CustomObjectScope({'custom_categorical_crossentropy': LOSS.custom_categorical_crossentropy([0.08, 0.13, 1, 0.5, 0.33])}):
    
        model.compile(loss='custom_categorical_crossentropy', 
                      optimizer=OPTIMIZER, 
                      metrics=['accuracy'])

    callbacks = [TensorBoard(log_dir=log_dir)]
    
    if noconv:
        history = model.fit(x_train, y_train,
                            batch_size=BATCH_SIZE, 
                            epochs=NB_EPOCHS, 
                            callbacks=callbacks,
                            verbose=VERBOSE, 
                            validation_split=VALIDATION_SPLIT)
    elif high or low:
        history = model.fit([x_image, x_train], y_train,
                            batch_size=BATCH_SIZE, 
                            epochs=NB_EPOCHS, 
                            callbacks=callbacks,
                            verbose=VERBOSE, 
                            validation_split=VALIDATION_SPLIT)
    else:
        history = model.fit(x_image, y_train,
                            batch_size=BATCH_SIZE, 
                            epochs=NB_EPOCHS, 
                            callbacks=callbacks,
                            verbose=VERBOSE, 
                            validation_split=VALIDATION_SPLIT)
    
    return model, history


def comparison_model(x_train, x_image, y_train, NB_EPOCHS, NB_CLASSES=5, high=True, low=False, noconv=False):

    BATCH_SIZE = 4096
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)

    set_dir_name='TEST_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    if noconv:
        if high:
            variable_input = Input(shape=(54,))
        elif low:
            variable_input = Input(shape=(44,))
        cla = Dense(256)(variable_input)
        cla = BatchNormalization()(cla) 
        cla = Activation('relu')(cla)
        cla = Dense(256)(cla)
        cla = BatchNormalization()(cla) 
        cla = Activation('relu')(cla)
        cla = Dense(256)(cla)
        cla = BatchNormalization()(cla) 
        cla = Activation('relu')(cla)
        cla = Dense(NB_CLASSES)(cla)
        vertex_output = Activation('softmax', name='vertex_output')(cla)

        model = Model(inputs=variable_input, outputs=vertex_output)

    else:
        image_input = Input(shape=(2, 100, 3)) # (tracks, t, xyz)
        if high:
            variable_input = Input(shape=(54,))
        elif low:
            variable_input = Input(shape=(44,))

        conv = Conv2D(16, (2, 30))(image_input)
        conv = BatchNormalization()(conv) 
        conv = Activation('relu')(conv)
        #conv = MaxPooling2D((2, 2))(conv)
        conv = Conv2D(32, (1, 30))(conv)
        conv = BatchNormalization()(conv) 
        conv = Activation('relu')(conv)
        #conv = MaxPooling2D((2, 2))(conv)
        conv = Conv2D(64, (1, 30))(conv)
        conv = BatchNormalization()(conv) 
        conv = Activation('relu')(conv)
        #conv = MaxPooling2D((2, 2))(conv)

        if high or low:
            conv_out = Flatten()(conv)
            combined = concatenate([conv_out, variable_input])
            cla = Dense(256)(combined)
            cla = BatchNormalization()(cla) 
            cla = Activation('relu')(cla)
            cla = Dense(256)(cla)
            cla = BatchNormalization()(cla) 
            cla = Activation('relu')(cla)
            cla = Dense(256)(cla)
            cla = BatchNormalization()(cla) 
            cla = Activation('relu')(cla)
            cla = Dense(NB_CLASSES)(cla)
            vertex_output = Activation('softmax', name='vertex_output')(cla)

            model = Model(inputs=[image_input, variable_input], outputs=vertex_output)

        else:
            conv = Flatten()(conv)
            conv = Dense(256)(conv)
            conv = BatchNormalization()(conv) 
            conv = Activation('relu')(conv)
            conv = Dense(256)(conv)
            conv = BatchNormalization()(conv) 
            conv = Activation('relu')(conv)
            conv = Dense(256)(conv)
            conv = BatchNormalization()(conv) 
            conv = Activation('relu')(conv)
            conv = Dense(NB_CLASSES)(conv)
            vertex_output = Activation('softmax', name='vertex_output')(conv)

            model = Model(inputs=image_input, outputs=vertex_output)

    model.summary()

    # Tensor Board
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=OPTIMIZER, 
                  metrics=['accuracy'])

    callbacks = [TensorBoard(log_dir=log_dir)]
    
    if noconv:
        history = model.fit(x_train, y_train,
                            batch_size=BATCH_SIZE, 
                            epochs=NB_EPOCHS, 
                            callbacks=callbacks,
                            verbose=VERBOSE, 
                            validation_split=VALIDATION_SPLIT)
    elif high or low:
        history = model.fit([x_image, x_train], y_train,
                            batch_size=BATCH_SIZE, 
                            epochs=NB_EPOCHS, 
                            callbacks=callbacks,
                            verbose=VERBOSE, 
                            validation_split=VALIDATION_SPLIT)
    else:
        history = model.fit(x_image, y_train,
                            batch_size=BATCH_SIZE, 
                            epochs=NB_EPOCHS, 
                            callbacks=callbacks,
                            verbose=VERBOSE, 
                            validation_split=VALIDATION_SPLIT)
    
    return model, history


def test_model(x_train, x_image, y_train, NB_EPOCHS):

    BATCH_SIZE = 4096
    VERBOSE = 1
    NB_CLASSES = 5
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)

    set_dir_name='TEST_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    image_input = Input(shape=(2, 101, 3)) # (tracks, t, xyz)
    #variable_input = Input(shape=(54,))
    variable_input = Input(shape=(44,))

    conv = Conv2D(16, (2, 30))(image_input)
    conv = BatchNormalization()(conv) 
    conv = Activation('relu')(conv)
    #conv = MaxPooling2D((2, 2))(conv)
    conv = Conv2D(32, (1, 30))(conv)
    conv = BatchNormalization()(conv) 
    conv = Activation('relu')(conv)
    #conv = MaxPooling2D((2, 2))(conv)
    conv = Conv2D(64, (1, 30))(conv)
    conv = BatchNormalization()(conv) 
    conv = Activation('relu')(conv)
    #conv = MaxPooling2D((2, 2))(conv)
    conv_out = Flatten()(conv)

    combined = concatenate([conv_out, variable_input])
    cla = Dense(256)(combined)
    cla = BatchNormalization()(cla) 
    cla = Activation('relu')(cla)
    cla = Dense(256)(cla)
    cla = BatchNormalization()(cla) 
    cla = Activation('relu')(cla)
    cla = Dense(256)(cla)
    cla = BatchNormalization()(cla) 
    cla = Activation('relu')(cla)
    cla = Dense(NB_CLASSES)(cla)
    vertex_output = Activation('softmax', name='vertex_output')(cla)

    model = Model(inputs=[image_input, variable_input], outputs=vertex_output)

    model.summary()

    # Tensor Board
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=OPTIMIZER, 
                  metrics=['accuracy'])

    callbacks = [TensorBoard(log_dir=log_dir)]
    
    history = model.fit([x_image, x_train], y_train,
                        batch_size=BATCH_SIZE, 
                        epochs=NB_EPOCHS, 
                        callbacks=callbacks,
                        verbose=VERBOSE, 
                        validation_split=VALIDATION_SPLIT)
    
    return model, history



def lstm_dense_model(pair, tracks, labels, NB_EPOCHS, INPUT_DIM=22, NODE_SIZE=32):

    BATCH_SIZE = 32
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)
    #OPTIMIZER = keras.optimizers.SGD(lr=0.001)

    set_dir_name='VLSTM_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    pair_input = Input(shape=(44,))
    track_input = Input(shape=(None, INPUT_DIM))

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

