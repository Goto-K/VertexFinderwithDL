import MODELTOOLS
import LSTMMODEL
import CUSTOM
import numpy as np
from keras.utils import np_utils



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
    
    init_dense = 2
    nb_samples = 50000
    node_size = 256
    nb_epochs = 100
    pair_reinforce=True
    model_name = "VLSTM_Model_InitState_ZeroPadding_Masking_Shuffle_Input23_" \
                 + str(init_dense) + "denseBNact_" \
                 + str(nb_samples) + "samples_" \
                 + str(node_size) + "nodes_" \
                 + str(nb_epochs) + "epochs_1LSTM"
    
    if pair_reinforce:
        model_name = model_name + "_pair_loss"

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

    # lstm_dense_model(pair, track, label, epochs, init_dense, input_din, node_size, nb_samples, pair_reinforce)
    model, history = LSTMMODEL.vlstm_model(teach_track_pairs, teach_tracks[:, :, :23], targets, 
                                           NB_EPOCHS=nb_epochs, 
                                           INIT_DENSE=init_dense,
                                           INPUT_DIM=23, 
                                           NODE_SIZE=node_size, 
                                           NB_SAMPLES=nb_samples,
                                           pair_reinforce=pair_reinforce)
           
            
    MODELTOOLS.save_history(history, model_name)
    MODELTOOLS.save_model(model, model_name)
