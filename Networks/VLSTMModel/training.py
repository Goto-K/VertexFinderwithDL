import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, Callback
from time import gmtime, strftime
import os, gc

from . import loss
from ..Tools import modeltools


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")


def VLSTMModelSimpleTraining(model, model_name, pair, tracks, labels, BATCH_SIZE=32, NB_EPOCHS=100, NB_SAMPLES=50000, VALIDATION_SPLIT=0.2, LR=0.001, pair_reinforce=False):

    full_size = len(labels)
    train_size = int(full_size*(1-VALIDATION_SPLIT))
    Eindex = np.random.permutation(full_size)
    pair_train, tracks_train, labels_train = pair[Eindex][:train_size], tracks[Eindex][:train_size], labels[Eindex][:train_size]
    pair_valid, tracks_valid, labels_valid = pair[Eindex][train_size:], tracks[Eindex][train_size:], labels[Eindex][train_size:]

    del pair, tracks, labels
    gc.collect()

    # Tensor Board
    set_dir_name='VLSTMTensorBoard'
    set_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../log/" + set_dir_name)
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_time = tictoc
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../log/" + set_dir_name + "/" + model_name + directory_time)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    model.compile(loss=loss.binary_crossentropy(pair_reinforce=pair_reinforce),
                  optimizer=Adam(lr=LR),
                  metrics=[loss.accuracy_all, loss.accuracy,
                           loss.true_positive, loss.true_negative, loss.false_positive, loss.false_negative])

    callbacks = [TensorBoard(log_dir=log_dir)]

    for epochs in range(NB_EPOCHS):
        EindexTrain = np.random.permutation(len(labels_train))
        EindexValid = np.random.permutation(len(labels_valid))
        TindexTrain = np.random.permutation(len(labels_train[0]))
        pair_train_use = pair_train[EindexTrain][:NB_SAMPLES]
        pair_valid_use = pair_valid[EindexValid][:int(NB_SAMPLES*VALIDATION_SPLIT)]
        tracks_valid_use = tracks_valid[EindexValid][:int(NB_SAMPLES*VALIDATION_SPLIT)]
        labels_valid_use = labels_valid[EindexValid][:int(NB_SAMPLES*VALIDATION_SPLIT)]

        shuffle_tracks_train = []
        shuffle_labels_train = []
        for t, l in zip(tracks_train[EindexTrain][:NB_SAMPLES], labels_train[EindexTrain][:NB_SAMPLES]):
            shuffle_tracks_train.append(t[TindexTrain])
            shuffle_labels_train.append(l[TindexTrain])
        shuffle_tracks_train, shuffle_labels_train = np.array(shuffle_tracks_train), np.array(shuffle_labels_train)
        print("===================== "
              + str(epochs+1) + "/" + str(NB_EPOCHS) + " epochs" +
              " =====================")

        new_history = model.fit([pair_train_use, shuffle_tracks_train], shuffle_labels_train,
                                batch_size=BATCH_SIZE,
                                epochs=1,
                                callbacks=callbacks,
                                verbose=1,
                                validation_data=([pair_valid_use, tracks_valid_use], labels_valid_use))

        if epochs == 0:
            history = {}
        history = modeltools.appendHist(history, new_history.history)

        del pair_train_use, shuffle_tracks_train, shuffle_labels_train
        del pair_valid_use, tracks_valid_use, labels_valid_use
        gc.collect()

    return model, history


def AttentionVLSTMModelTraining(model, model_name, pair, tracks, labels, BATCH_SIZE=32, NB_EPOCHS=100, NB_SAMPLES=50000, VALIDATION_SPLIT=0.2, LR=0.001, pair_reinforce=False):

    full_size = len(labels)
    train_size = int(full_size*(1-VALIDATION_SPLIT))
    Eindex = np.random.permutation(full_size)
    pair_train, tracks_train, labels_train = pair[Eindex][:train_size], tracks[Eindex][:train_size], labels[Eindex][:train_size]
    pair_valid, tracks_valid, labels_valid = pair[Eindex][train_size:], tracks[Eindex][train_size:], labels[Eindex][train_size:]

    del pair, tracks, labels
    gc.collect()

    # Tensor Board
    set_dir_name='VLSTMTensorBoard'
    set_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../log/" + set_dir_name)
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_time = tictoc
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../log/" + set_dir_name + "/" + model_name + directory_time)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    model.compile(loss=loss.binary_crossentropy(pair_reinforce=pair_reinforce),
                  optimizer=Adam(lr=LR),
                  metrics=[loss.accuracy_all, loss.accuracy,
                           loss.true_positive, loss.true_negative, loss.false_positive, loss.false_negative])

    callbacks = [TensorBoard(log_dir=log_dir)]

    for epochs in range(NB_EPOCHS):
        EindexTrain = np.random.permutation(len(labels_train))
        EindexValid = np.random.permutation(len(labels_valid))
        TindexTrain = np.random.permutation(len(labels_train[0]))
        pair_train_use = pair_train[EindexTrain][:NB_SAMPLES]
        pair_valid_use = pair_valid[EindexValid][:int(NB_SAMPLES*VALIDATION_SPLIT)]
        tracks_valid_use = tracks_valid[EindexValid][:int(NB_SAMPLES*VALIDATION_SPLIT)]
        labels_valid_use = labels_valid[EindexValid][:int(NB_SAMPLES*VALIDATION_SPLIT)]

        shuffle_tracks_train = []
        shuffle_labels_train = []
        for t, l in zip(tracks_train[EindexTrain][:NB_SAMPLES], labels_train[EindexTrain][:NB_SAMPLES]):
            shuffle_tracks_train.append(t[TindexTrain])
            shuffle_labels_train.append(l[TindexTrain])
        shuffle_tracks_train, shuffle_labels_train = np.array(shuffle_tracks_train), np.array(shuffle_labels_train)
        print("===================== "
              + str(epochs+1) + "/" + str(NB_EPOCHS) + " epochs" +
              " =====================")

        new_history = model.fit([pair_train_use, shuffle_tracks_train, shuffle_tracks_train], shuffle_labels_train,
                                batch_size=BATCH_SIZE,
                                epochs=1,
                                callbacks=callbacks,
                                verbose=1,
                                validation_data=([pair_valid_use, tracks_valid_use, tracks_valid_use], labels_valid_use))

        if epochs == 0:
            history = {}
        history = modeltools.appendHist(history, new_history.history)

        del pair_train_use, shuffle_tracks_train, shuffle_labels_train
        del pair_valid_use, tracks_valid_use, labels_valid_use
        gc.collect()

    return model, history

