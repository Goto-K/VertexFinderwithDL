import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
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
import GRAPH


def pair_model(x_train, y_train, NB_EPOCHS, NB_CLASSES=5, chi=False):

    BATCH_SIZE = 1024
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.SGD(lr=0.001)
    Input_DIM = x_train.shape[1]

    set_dir_name='PAIR_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    variable_input = Input(shape=(Input_DIM,))
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
    
    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE, 
                        epochs=NB_EPOCHS, 
                        callbacks=callbacks,
                        verbose=VERBOSE, 
                        validation_split=VALIDATION_SPLIT)
    
    return model, history


def pair_adacos_model(x_train, y_train, NB_EPOCHS, NB_CLASSES=5, chi=False):

    pair_model_name = "Pair_Model_vfdnn04_1Msamples_2500epochs"
    BATCH_SIZE = 1024
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)

    set_dir_name='PAIR_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    pair_model = MODELBANK.load_model(pair_model_name)
    cla = pair_model.output
    _tmp = Input(shape=(NB_CLASSES,), name="y_input")
    """
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
    cla = Activation('softmax')(cla)
    """
    cla = ADACOS.Adacoslayer(NB_CLASSES, np.sqrt(2)*np.log(NB_CLASSES-1), 0.0)([cla, _tmp])
    vertex_output = Activation('softmax', name='adacos_output')(cla)

    model = Model(inputs=[pair_model.input, _tmp], outputs=vertex_output)

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
    
    history = model.fit([x_train, y_train], y_train,
                        batch_size=BATCH_SIZE, 
                        epochs=NB_EPOCHS, 
                        callbacks=callbacks,
                        verbose=VERBOSE, 
                        validation_split=VALIDATION_SPLIT)
    
    return model, history


def pair_adacos_eval(adacos_model):

    OPTIMIZER = keras.optimizers.SGD(lr=0.001)

    model = Model(adacos_model.get_layer(index=0).input, adacos_model.get_layer(index=-4).output)
    model.summary()
    with CustomObjectScope({'custom_categorical_crossentropy': LOSS.custom_categorical_crossentropy([0.02, 0.08, 1, 0.5, 0.37])}):
        model.compile(loss='custom_categorical_crossentropy', 
                      optimizer=OPTIMIZER, 
                      metrics=['accuracy'])

    return model


def pair_affinity_model(x_train, y_train, NB_EPOCHS, NB_CLASSES=5):

    BATCH_SIZE = 1024
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)

    set_dir_name='PAIR_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    variable_input = Input(shape=(44,))
    cla = Dense(256)(variable_input)
    cla = BatchNormalization()(cla) 
    cla = Activation('relu')(cla)
    cla = Dense(256)(cla)
    cla = BatchNormalization()(cla) 
    cla = Activation('relu')(cla)
    cla = Dense(256)(cla)
    cla = BatchNormalization()(cla) 
    vertex_output = AFFINITY.ClusteringAffinity(NB_CLASSES, 1, 10.0)(cla)

    model = Model(inputs=variable_input, outputs=vertex_output)

    model.summary()

    # Tensor Board
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with CustomObjectScope({'affinity_loss': AFFINITY.affinity_loss(0.75), 'affinity_acc' : AFFINITY.acc}):
        model.compile(loss='affinity_loss', 
                      optimizer=OPTIMIZER, 
                      metrics=['affinity_acc'])

    callbacks = [TensorBoard(log_dir=log_dir)]

    dummy_train = np.zeros((x_train.shape[0], 1), dtype=np.float32)
    y_train = np.concatenate([y_train, dummy_train], axis=-1)
    
    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE, 
                        epochs=NB_EPOCHS, 
                        callbacks=callbacks,
                        verbose=VERBOSE, 
                        validation_split=VALIDATION_SPLIT)
    
    return model, history


def pair_pos_model(x_train, y_train, NB_EPOCHS):

    BATCH_SIZE = 1024
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.SGD(lr=0.001)

    set_dir_name='PAIR_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    variable_input = Input(shape=(44,))
    reg = Dense(256)(variable_input)
    reg = BatchNormalization()(reg) 
    reg = Activation('relu')(reg)
    reg = Dense(256)(reg)
    reg = BatchNormalization()(reg) 
    reg = Activation('relu')(reg)
    pos_output = Dense(1, name='pos_output')(reg)

    model = Model(inputs=variable_input, outputs=pos_output)

    model.summary()

    # Tensor Board
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    #(71313., 19564.,  1606.,  3181.,  4336) : full 0.02, 0.08, 1, 0.5, 0.37
    #(2494726., 1526557.,  118644.,  228955.,  278428.) : chi100 0.05, 0.08, 1, 0.52, 0.43
    model.compile(loss=['mean_squared_logarithmic_error'],
                  optimizer=OPTIMIZER, 
                  metrics=['mae'])

    callbacks = [TensorBoard(log_dir=log_dir)]
    
    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE, 
                        epochs=NB_EPOCHS, 
                        callbacks=callbacks,
                        verbose=VERBOSE, 
                        validation_split=VALIDATION_SPLIT)
    
    return model, history


def pair_graph_model(x_train, a_train, y_train, NB_EPOCHS):

    BATCH_SIZE = 32
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.SGD(lr=0.001)

    set_dir_name='GRAPH_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    X_input = Input(shape=(None, 22), name="X_Input")
    A_input = Input(shape=(None, None), name="A_Input")

    graph_first = GRAPH.GraphAdditiveAttentionLayer(feature_units=256,
                                            attn_heads=1,
                                            attn_heads_reduction="concat",  # {"concat", "average"}
                                            activation="relu",
                                            attn_kernel_initializer="glorot_uniform",
                                            attn_kernel_regularizer=None,
                                            attn_kernel_constraint=None,
                                            attention=True,
                                            return_attention=True,
                                            node_level_bias=False,
                                            use_bias=False)
 
    vectors_first, attention_first = graph_first([X_input, A_input])

    graph_second = GRAPH.GraphAdditiveAttentionLayer(feature_units=256,
                                             attn_heads=1,
                                             attn_heads_reduction="concat",  # {"concat", "average"}
                                             activation="relu",
                                             attn_kernel_initializer="glorot_uniform",
                                             attn_kernel_regularizer=None,
                                             attn_kernel_constraint=None,
                                             attention=True,
                                             return_attention=True,
                                             node_level_bias=False,
                                             use_bias=False)
 
    vectors_second, attention_second = graph_second([vectors_first, attention_first[0]])
    #attention = tf.expand_dims(attention_second, axis=-1)

    model = Model(inputs=[X_input, A_input], outputs=attention_second)

    model.summary()

    # Tensor Board
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    print(y_train.shape)
    model.compile(loss=GRAPH.binary_crossentropy(),
                  optimizer=OPTIMIZER, 
                  metrics=[GRAPH.accuracy_all])

    """
    model.compile(loss=["binary_crossentropy"],
                  optimizer=OPTIMIZER, 
                  metrics=["accuracy"])
    """

    callbacks = [TensorBoard(log_dir=log_dir)]
    
    history = model.fit([x_train, a_train], y_train,
                        batch_size=BATCH_SIZE, 
                        epochs=NB_EPOCHS, 
                        callbacks=callbacks,
                        verbose=VERBOSE, 
                        validation_split=VALIDATION_SPLIT)
    
    return model, history


