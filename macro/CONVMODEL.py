import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Conv2D, AveragePooling2D, Flatten, Concatenate
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras.utils import CustomObjectScope
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
import LOSS
import MODELTOOLS
import MODELBANK
import ADACOS
import AFFINITY


def conv_model(image, x_train, y_train, NB_EPOCHS, NB_CLASSES=5, chi=False):

    BATCH_SIZE = 64
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)

    set_dir_name='CONV_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    image_input = Input(shape=(200, 200, 3))
    variable_input = Input(shape=(44,))

    conv = Conv2D(32, 3, padding='same')(image_input)
    conv = BatchNormalization()(conv)
    conv = AveragePooling2D()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(32, 3, padding='same')(conv)
    conv = BatchNormalization()(conv) 
    conv = AveragePooling2D()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(64, 3, padding='same')(conv)
    conv = BatchNormalization()(conv) 
    conv = AveragePooling2D()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(64, 4, padding='valid')(conv)
    conv = BatchNormalization()(conv) 
    conv = AveragePooling2D()(conv)
    conv = Activation('relu')(conv)

    conv = Flatten()(conv)
    cla = Concatenate()([conv, variable_input])
    cla = Dense(256)(cla)
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

    #(71313., 19564.,  1606.,  3181.,  4336) : full 0.02, 0.08, 1, 0.5, 0.37
    #(2494726., 1526557.,  118644.,  228955.,  278428.) : chi100 0.05, 0.08, 1, 0.52, 0.43
    if chi:
        with CustomObjectScope({'custom_categorical_crossentropy': LOSS.custom_categorical_crossentropy([0.05, 0.08, 1, 0.52, 0.43])}):
            model.compile(loss='custom_categorical_crossentropy', 
                          optimizer=OPTIMIZER, 
                          metrics=['accuracy'])
    else:
        with CustomObjectScope({'custom_categorical_crossentropy': LOSS.custom_categorical_crossentropy([0.02, 0.08, 1, 0.5, 0.37])}):
            model.compile(loss='custom_categorical_crossentropy', 
                          optimizer=OPTIMIZER, 
                          metrics=['accuracy'])

    callbacks = [TensorBoard(log_dir=log_dir)]
    
    history = model.fit([image, x_train], y_train,
                        batch_size=BATCH_SIZE, 
                        epochs=NB_EPOCHS, 
                        callbacks=callbacks,
                        verbose=VERBOSE, 
                        validation_split=VALIDATION_SPLIT)
    
    return model, history


