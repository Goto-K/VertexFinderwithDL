import EVALTOOLS
import MODELTOOLS
import LSTMMODEL
import numpy as np
from keras.utils import np_utils
from tensorflow import keras



if __name__ == "__main__":

    init_dense = 2
    nb_samples = 50000
    node_size = 32
    nb_epochs = 50
    pair_reinforce=True
    model_name = "VLSTM_Model_InitState_ZeroPadding_Masking_Shuffle_" \
                 + str(init_dense) + "dense_" \
                 + str(nb_samples) + "samples_" \
                 + str(node_size) + "nodes_" \
                 + str(nb_epochs) + "epochs_2LSTM"
    
    if pair_reinforce:
        model_name = model_name + "_pair_loss"

    model_name = "VLSTM_Model_vfdnn06_50000samples_100epochs"

    name_track_labels_pv = "/home/goto/ILC/Deep_Learning/data/vfdnn_08_track_labels_pv.npy"
    name_track_labels_sv = "/home/goto/ILC/Deep_Learning/data/vfdnn_08_track_labels_sv.npy"
    name_tracks_pvsv = "/home/goto/ILC/Deep_Learning/data/vfdnn_08_tracks_pvsvnc.npy"
    name_track_pairs_pv = "/home/goto/ILC/Deep_Learning/data/vfdnn_08_track_pairs_pv.npy"
    name_track_pairs_sv = "/home/goto/ILC/Deep_Learning/data/vfdnn_08_track_pairs_sv.npy"
    name_track_plabels_pv = "/home/goto/ILC/Deep_Learning/data/vfdnn_08_track_plabels_pv.npy"
    name_track_plabels_sv = "/home/goto/ILC/Deep_Learning/data/vfdnn_08_track_plabels_sv.npy"

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


    model = LSTMMODEL.load_model(model_name)
        
    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    
    predict = model.predict([teach_track_pairs, teach_tracks[:, :, :23]])
    predict = np.array(predict, dtype=float)

    # Evaluation of Vertex Finder
    EVALTOOLS.cut_curve_binary(predict, teach_track_labels, teach_tracks[:, 0], model_name)

