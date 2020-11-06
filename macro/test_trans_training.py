import MODELTOOLS
import LSTMMODEL
import CUSTOM
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from keras.utils import np_utils
from time import gmtime, strftime
import os, h5py, pathlib



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


if __name__ == "__main__":
    
    encoder_input = 256
    decoder_units = 256
    nb_samples = 50000
    first_epochs = 1
    transfer_model_name = "Transfer_Learning_Model_for_Attention_23Input_30epochs"
    nb_epochs = 1
    pair_reinforce=True
    attention_model_name = "Attention_VLSTM_Model_InitState_ZeroPadding_Masking_Shuffle_Bidirection_Transfer_23Input_" \
                         + str(nb_samples) + "samples_" \
                         + str(nb_epochs) + "epochs"
    
    if pair_reinforce:
        attention_model_name = attention_model_name + "_pair_loss"

    name_track_labels_pv = "/home/goto/ILC/Deep_Learning/data/test_track_labels_pv.npy"
    name_track_labels_sv = "/home/goto/ILC/Deep_Learning/data/test_track_labels_sv.npy"
    name_tracks_pvsv = "/home/goto/ILC/Deep_Learning/data/test_tracks_pvsv.npy"
    name_track_pairs_pv = "/home/goto/ILC/Deep_Learning/data/test_track_pairs_pv.npy"
    name_track_pairs_sv = "/home/goto/ILC/Deep_Learning/data/test_track_pairs_sv.npy"
    name_track_plabels_pv = "/home/goto/ILC/Deep_Learning/data/test_track_plabels_pv.npy"
    name_track_plabels_sv = "/home/goto/ILC/Deep_Learning/data/test_track_plabels_sv.npy"

    data_track_labels_pv = np.load(name_track_labels_pv, allow_pickle=True)
    data_track_labels_sv = np.load(name_track_labels_sv, allow_pickle=True)
    data_tracks_pvsv = np.load(name_tracks_pvsv, allow_pickle=True)
    data_track_pairs_pv = np.load(name_track_pairs_pv, allow_pickle=True)
    data_track_pairs_sv = np.load(name_track_pairs_sv, allow_pickle=True)
    data_track_plabels_pv = np.load(name_track_plabels_pv, allow_pickle=True)
    data_track_plabels_sv = np.load(name_track_plabels_sv, allow_pickle=True)

    data_track_labels = np.concatenate([data_track_labels_pv, data_track_labels_sv], 0)
    data_tracks = np.concatenate([data_tracks_pvsv, data_tracks_pvsv], 0)
    data_track_pairs = np.concatenate([data_track_pairs_pv, data_track_pairs_sv], 0)
    data_track_plabels = np.concatenate([data_track_plabels_pv, data_track_plabels_sv], 0)


    print("file load !")

    print(data_track_labels.shape)
    print(data_tracks.shape)
    print(data_track_pairs.shape)

    teach_track_labels, teach_tracks, teach_track_pairs = MODELTOOLS.data_zero_padding(data_track_labels, 
                                                                                       data_tracks, 
                                                                                       data_track_pairs, 
                                                                                       data_track_plabels)

    targets = []
    for labels, tracks, pairs in zip(teach_track_labels, teach_tracks, teach_track_pairs):
        target = []
        for label, track in zip(labels, tracks):
            label = np.atleast_1d(label)
            zero_padding = np.atleast_1d(track[0])
            pair1 = np.atleast_1d(pairs[0])
            pair2 = np.atleast_1d(pairs[22])
            track = np.atleast_1d(track[1])
            t = np.concatenate([label, zero_padding, pair1, pair2, track])
            target.append(t)
        targets.append(target)
    targets = np.array(targets)

    print(targets.shape)

    # lstm_dense_model(pair, track, label, epochs, encoder_input, decoder_units, nb_samples, pair_reinforce)
    model, history = LSTMMODEL.attentionvlstm_model(teach_track_pairs, teach_tracks[:, :, :23], targets, 
                                                    NB_EPOCHS=1, 
                                                    ENCODER_INPUT=encoder_input,
                                                    DECODER_UNITS=decoder_units, 
                                                    NB_SAMPLES=nb_samples,
                                                    pair_reinforce=pair_reinforce)
           
    transfer_model =  Model(inputs=model.input, outputs=model.get_layer("Transfer_Learning").output)   
    transfer_model.summary()

    # Tensor Board
    set_dir_name_t = "Transfer"
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name_t + "/" + set_dir_name_t + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    callbacks = [TensorBoard(log_dir=log_dir)]

    transfer_model.compile(loss=CUSTOM.binary_crossentropy(pair_reinforce=pair_reinforce),
                           optimizer=keras.optimizers.Adam(lr=0.001),
                           metrics=[CUSTOM.accuracy_all, CUSTOM.accuracy,
                           CUSTOM.true_positive, CUSTOM.true_negative, CUSTOM.false_positive, CUSTOM.false_negative])
 
    for epochs in range(first_epochs):
        Eindex = np.random.permutation(len(targets))
        Tindex = np.random.permutation(len(targets[0]))
        shuffle_tracks = []
        shuffle_labels = []
        for t, l in zip(teach_tracks[Eindex][:50000], targets[Eindex][:50000]):
            shuffle_tracks.append(t[Tindex])
            shuffle_labels.append(l[Tindex])
        shuffle_tracks, shuffle_labels = np.array(shuffle_tracks), np.array(shuffle_labels)
        print("===================== "
              + str(epochs+1) + "/" + str(NB_EPOCHS) + " epochs" +
              " =====================")

        new_history = transfer_model.fit([teach_track_pairs[Eindex][:50000], shuffle_tracks, shuffle_tracks], shuffle_labels,
                                          batch_size=32, 
                                          epochs=1, 
                                          callbacks=callbacks,
                                          verbose=1, 
                                          validation_split=0.2)

        if epochs == 0:
            transfer_history = {}
        transfer_history = MODELTOOLS.appendHist(transfer_history, new_history.history)
    
    MODELTOOLS.save_history(transfer_history, transfer_model_name)
    MODELTOOLS.save_attention_model(transfer_model, transfer_model_name)

    attention_model =  Model(inputs=transfer_model.input, outputs=transfer_model.get_layer("Decoder_Attention_VLSTM").output[0])   
    attention_model.summary()

    # Tensor Board
    set_dir_name = "Attention_VLSTM_MODEL"
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    callbacks = [TensorBoard(log_dir=log_dir)]

    attention_model.compile(loss=CUSTOM.binary_crossentropy(pair_reinforce=pair_reinforce),
                            optimizer=keras.optimizers.Adam(lr=0.001),
                            metrics=[CUSTOM.accuracy_all, CUSTOM.accuracy,
                            CUSTOM.true_positive, CUSTOM.true_negative, CUSTOM.false_positive, CUSTOM.false_negative])
 
    for epochs in range(nb_epochs):
        Eindex = np.random.permutation(len(targets))
        Tindex = np.random.permutation(len(targets[0]))
        shuffle_tracks = []
        shuffle_labels = []
        for t, l in zip(teach_tracks[Eindex][:50000], targets[Eindex][:50000]):
            shuffle_tracks.append(t[Tindex])
            shuffle_labels.append(l[Tindex])
        shuffle_tracks, shuffle_labels = np.array(shuffle_tracks), np.array(shuffle_labels)
        print("===================== "
              + str(epochs+1) + "/" + str(NB_EPOCHS) + " epochs" +
              " =====================")

        new_history = attention_model.fit([teach_track_pairs[Eindex][:50000], shuffle_tracks, shuffle_tracks], shuffle_labels,
                                           batch_size=32, 
                                           epochs=1, 
                                           callbacks=callbacks,
                                           verbose=1, 
                                           validation_split=0.2)

        if epochs == 0:
            attention_history = {}
        attention_history = MODELTOOLS.appendHist(attention_history, new_history.history)
    
    MODELTOOLS.save_history(attention_history, attention_model_name)
    MODELTOOLS.save_attention_model(attention_model, attention_model_name)

