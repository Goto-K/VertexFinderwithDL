import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, RNN, Activation, TimeDistributed, Masking, BatchNormalization, Reshape, Bidirectional
from tensorflow.keras.models import Model, model_from_json
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
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
import os, h5py
import pathlib
import CUSTOM
import ATTENTION
import MODELTOOLS


def load_model(model_name):
    #model = load_model("/home/ilc/gotok/WorkDir/python/model/" + model_name + "_" + date + ".h5", compile=False)

    model_path = "/home/goto/ILC/Deep_Learning/model/" + model_name
    json_path = model_path + ".json"
    h5_path = model_path + ".h5"
    structure = pathlib.Path(json_path).read_text()

    model = model_from_json(structure, custom_objects={'VLSTMCell':CUSTOM.VLSTMCell, 'VLSTMCell_M':CUSTOM.VLSTMCell_M, 'AttentionVLSTMCell':CUSTOM.AttentionVLSTMCell})
    model.load_weights(h5_path)
    model.summary()

    return model


def load_attention_model(model_name):
    #model = load_model("/home/ilc/gotok/WorkDir/python/model/" + model_name + "_" + date + ".h5", compile=False)

    model_path = "/home/goto/ILC/Deep_Learning/model/" + model_name
    json_path = model_path + ".json"
    h5_path = model_path + ".h5"
    structure = pathlib.Path(json_path).read_text()

    model = model_from_json(structure, custom_objects={'VLSTMCell':CUSTOM.VLSTMCell, 'VLSTMCell_M':CUSTOM.VLSTMCell_M, 'AttentionVLSTMCell':CUSTOM.AttentionVLSTMCell})
    file=h5py.File(h5_path,'r')
    weight = []
    for i in range(len(file.keys())):
        weight.append(file['weight'+str(i)][:])
    model.set_weights(weight)
    model.summary()

    return model


def attentionvlstm_model(pair, tracks, labels, NB_EPOCHS, 
                         ENCODER_INPUT=256, DECODER_UNITS=256, NB_SAMPLES=50000, pair_reinforce=False):

    BATCH_SIZE = 32
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)
    MAX_TRACK_NUM = tracks.shape[1]
    INPUT_DIM = tracks.shape[2]
    DECODER_OUTPUT = 1

    set_dir_name='Attention_VLSTM_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    pair_input = Input(shape=(44,), name='Pair_Input')

    encoder_input = Input(shape=(MAX_TRACK_NUM, INPUT_DIM), name='Encoder_Input')
    encoder_embedd = TimeDistributed(Dense(256, name='Encoder_Embedding_Dense', activation='relu'))(encoder_input)

    decoder_input = Input(shape=(None, INPUT_DIM), name='Decoder_Input')
    decoder_embedd = TimeDistributed(Dense(256, name='Decoder_Embedding_Dense', activation='relu'))(decoder_input)
    #decoder_embedd = TimeDistributed(Dense(512, name='Decoder_Embedding_Dense', activation='relu'))(decoder_input) # scaled dot

    encoder_init_state = Dense(ENCODER_INPUT, name='Encoder_Dense_1')(pair_input)
    ecnoder_init_state = BatchNormalization(name='Encoder_BatchNorm_1')(encoder_init_state)
    encoder_init_state = Activation('relu', name='Encoder_Activation_1')(encoder_init_state)
    encoder_init_state = Dense(ENCODER_INPUT, name='Encoder_Dense_2')(encoder_init_state)
    ecnoder_init_state = BatchNormalization(name='Encoder_BatchNorm_2')(encoder_init_state)
    encoder_init_state = Activation('relu', name='Encoder_Activation_2')(encoder_init_state)

    encoder_init_state_b = Dense(ENCODER_INPUT, name='Encoder_b_Dense_1')(pair_input)
    ecnoder_init_state_b = BatchNormalization(name='Encoder_b_BatchNorm_1')(encoder_init_state_b)
    encoder_init_state_b = Activation('relu', name='Encoder_b_Activation_1')(encoder_init_state_b)
    encoder_init_state_b = Dense(ENCODER_INPUT, name='Encoder_b_Dense_2')(encoder_init_state_b)
    ecnoder_init_state_b = BatchNormalization(name='Encoder_b_BatchNorm_2')(encoder_init_state_b)
    encoder_init_state_b = Activation('relu', name='Encoder_b_Activation_2')(encoder_init_state_b)

    decoder_init_state = Dense(DECODER_UNITS, name='Decoder_Dense_1')(pair_input)
    denoder_init_state = BatchNormalization(name='Decoder_BatchNorm_1')(decoder_init_state)
    decoder_init_state = Activation('relu', name='Decoder_Activation_1')(decoder_init_state)
    decoder_init_state = Dense(DECODER_UNITS, name='Decoder_Dense_2')(decoder_init_state)
    denoder_init_state = BatchNormalization(name='Decoder_BatchNorm_2')(decoder_init_state)
    decoder_init_state = Activation('relu', name='Decoder_Activation_2')(decoder_init_state)

    vlstm_cell = CUSTOM.VLSTMCell_M(ENCODER_INPUT)
    vlstm_cell_b = CUSTOM.VLSTMCell_M(ENCODER_INPUT)
    encoder = RNN(vlstm_cell, return_sequences=True, name="Encoder_VLSTM", go_backwards=False)
    encoder_b = RNN(vlstm_cell_b, return_sequences=True, name="Encoder_VLSTM_b", go_backwards=True)

    with CustomObjectScope({"VLSTMCell_M": CUSTOM.VLSTMCell_M}):
        biencoder = Bidirectional(encoder, backward_layer=encoder_b)(encoder_embedd, initial_state=[encoder_init_state, encoder_init_state, encoder_init_state_b, encoder_init_state_b])

    transfer = TimeDistributed(Dense(1, name='Transfer_Learning'))(biencoder)
    biencoder = Reshape(target_shape=(MAX_TRACK_NUM*ENCODER_INPUT*2,))(biencoder)

    # DCODER_UNITS, ENCODER_INPUTS, DECODER_OUTPUT, MAX_TRACK_NUM
    attentionvlstm_cell = CUSTOM.AttentionVLSTMCell(DECODER_UNITS, ENCODER_INPUT*2, DECODER_OUTPUT, MAX_TRACK_NUM)
    #attentionvlstm_cell = CUSTOM.AttentionVLSTMCell(DECODER_UNITS, ENCODER_INPUT, DECODER_OUTPUT, MAX_TRACK_NUM)
    
    decoder, attention = RNN(attentionvlstm_cell, return_sequences=True, name='Decoder_Attention_VLSTM')(decoder_embedd, initial_state=[biencoder, decoder_init_state])
    
    model = Model(inputs=[pair_input, encoder_input, decoder_input], outputs=decoder)

    model.summary()

    # Tensor Board
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    model.compile(loss=CUSTOM.binary_crossentropy(pair_reinforce=pair_reinforce), 
                  optimizer=OPTIMIZER, 
                  metrics=[CUSTOM.accuracy_all, CUSTOM.accuracy, 
                           CUSTOM.true_positive, CUSTOM.true_negative, CUSTOM.false_positive, CUSTOM.false_negative])

    callbacks = [TensorBoard(log_dir=log_dir)]
    
    for epochs in range(NB_EPOCHS):
        Eindex = np.random.permutation(len(labels))
        Tindex = np.random.permutation(len(labels[0]))
        shuffle_tracks = []
        shuffle_labels = []
        for t, l in zip(tracks[Eindex][:NB_SAMPLES], labels[Eindex][:NB_SAMPLES]):
            shuffle_tracks.append(t[Tindex])
            shuffle_labels.append(l[Tindex])
        shuffle_tracks, shuffle_labels = np.array(shuffle_tracks), np.array(shuffle_labels)
        print("===================== "
              + str(epochs+1) + "/" + str(NB_EPOCHS) + " epochs" +
              " =====================")

        new_history = model.fit([pair[Eindex][:NB_SAMPLES], shuffle_tracks, shuffle_tracks], shuffle_labels,
                            batch_size=BATCH_SIZE, 
                            epochs=1, 
                            callbacks=callbacks,
                            verbose=VERBOSE, 
                            validation_split=VALIDATION_SPLIT)

        if epochs == 0:
            history = {}
        history = MODELTOOLS.appendHist(history, new_history.history)
    
    return model, history

def transfer_model(pair, tracks, labels, NB_EPOCHS, 
                   ENCODER_INPUT=256, DECODER_UNITS=256, NB_SAMPLES=50000, pair_reinforce=False):

    BATCH_SIZE = 32
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)
    MAX_TRACK_NUM = tracks.shape[1]
    INPUT_DIM = tracks.shape[2]
    DECODER_OUTPUT = 1

    set_dir_name='Test_Attention_VLSTM_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    pair_input = Input(shape=(44,), name='Pair_Input')

    encoder_input = Input(shape=(MAX_TRACK_NUM, INPUT_DIM), name='Encoder_Input')
    encoder_embedd = TimeDistributed(Dense(256, name='Encoder_Embedding_Dense'))(encoder_input)

    encoder_init_state = Dense(ENCODER_INPUT, name='Encoder_Dense_1')(pair_input)
    ecnoder_init_state = BatchNormalization(name='Encoder_BatchNorm_1')(encoder_init_state)
    encoder_init_state = Activation('relu', name='Encoder_Activation_1')(encoder_init_state)
    encoder_init_state = Dense(ENCODER_INPUT, name='Encoder_Dense_2')(encoder_init_state)
    ecnoder_init_state = BatchNormalization(name='Encoder_BatchNorm_2')(encoder_init_state)
    encoder_init_state = Activation('relu', name='Encoder_Activation_2')(encoder_init_state)

    encoder_init_state_b = Dense(ENCODER_INPUT, name='Encoder_b_Dense_1')(pair_input)
    ecnoder_init_state_b = BatchNormalization(name='Encoder_b_BatchNorm_1')(encoder_init_state_b)
    encoder_init_state_b = Activation('relu', name='Encoder_b_Activation_1')(encoder_init_state_b)
    encoder_init_state_b = Dense(ENCODER_INPUT, name='Encoder_b_Dense_2')(encoder_init_state_b)
    ecnoder_init_state_b = BatchNormalization(name='Encoder_b_BatchNorm_2')(encoder_init_state_b)
    encoder_init_state_b = Activation('relu', name='Encoder_b_Activation_2')(encoder_init_state_b)

    vlstm_cell = CUSTOM.VLSTMCell_M(ENCODER_INPUT)
    vlstm_cell_b = CUSTOM.VLSTMCell_M(ENCODER_INPUT)
    encoder = RNN(vlstm_cell, return_sequences=True, name="Encoder_VLSTM", go_backwards=False)
    encoder_b = RNN(vlstm_cell_b, return_sequences=True, name="Encoder_VLSTM_b", go_backwards=True)

    with CustomObjectScope({"VLSTMCell_M": CUSTOM.VLSTMCell_M}):
        biencoder = Bidirectional(encoder, backward_layer=encoder_b)(encoder_embedd, initial_state=[encoder_init_state, encoder_init_state, encoder_init_state_b, encoder_init_state_b])

    biencoder = TimeDistributed(Dense(1, name='Transfer_Learning'))(biencoder)

    model = Model(inputs=[pair_input, encoder_input], outputs=biencoder)

    model.summary()

    # Tensor Board
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    model.compile(loss=CUSTOM.binary_crossentropy(pair_reinforce=pair_reinforce), 
                  optimizer=OPTIMIZER, 
                  metrics=[CUSTOM.accuracy_all, CUSTOM.accuracy, 
                           CUSTOM.true_positive, CUSTOM.true_negative, CUSTOM.false_positive, CUSTOM.false_negative])

    callbacks = [TensorBoard(log_dir=log_dir)]
    
    for epochs in range(NB_EPOCHS):
        Eindex = np.random.permutation(len(labels))
        Tindex = np.random.permutation(len(labels[0]))
        shuffle_tracks = []
        shuffle_labels = []
        for t, l in zip(tracks[Eindex][:NB_SAMPLES], labels[Eindex][:NB_SAMPLES]):
            shuffle_tracks.append(t[Tindex])
            shuffle_labels.append(l[Tindex])
        shuffle_tracks, shuffle_labels = np.array(shuffle_tracks), np.array(shuffle_labels)
        print("===================== "
              + str(epochs+1) + "/" + str(NB_EPOCHS) + " epochs" +
              " =====================")

        new_history = model.fit([pair[Eindex][:NB_SAMPLES], shuffle_tracks], shuffle_labels,
                            batch_size=BATCH_SIZE, 
                            epochs=1, 
                            callbacks=callbacks,
                            verbose=VERBOSE, 
                            validation_split=VALIDATION_SPLIT)

        if epochs == 0:
            history = {}
        history = MODELTOOLS.appendHist(history, new_history.history)
    
    return model, history
