import EVALTOOLS
import MODELTOOLS
import LSTMMODEL
import CUSTOM
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, Callback
from keras.utils import np_utils
from tensorflow import keras
from time import gmtime, strftime
import os



if __name__ == "__main__":

    nb_samples = 50000
    nb_epochs = 100
    pair_reinforce=False

    model_name = "Attention_Bidirectional_VLSTM_Model_vfdnn06_50000samples_100epochs"
    new_model_name = "Attention_Bidirectional_VLSTM_Model_vfdnn06_50000samples_100epochs_ps_100epochs_s"

    """
    name_track_labels_pv = "/home/goto/ILC/Deep_Learning/data/vfdnn_06_track_labels_pv.npy"
    name_track_labels_sv = "/home/goto/ILC/Deep_Learning/data/vfdnn_06_track_labels_sv.npy"
    name_tracks_pvsv = "/home/goto/ILC/Deep_Learning/data/vfdnn_06_tracks_pvsvnc.npy"
    name_track_pairs_pv = "/home/goto/ILC/Deep_Learning/data/vfdnn_06_track_pairs_pv.npy"
    name_track_pairs_sv = "/home/goto/ILC/Deep_Learning/data/vfdnn_06_track_pairs_sv.npy"
    name_track_plabels_pv = "/home/goto/ILC/Deep_Learning/data/vfdnn_06_track_plabels_pv.npy"
    name_track_plabels_sv = "/home/goto/ILC/Deep_Learning/data/vfdnn_06_track_plabels_sv.npy"

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
    """

    name_track_labels_sv = "/home/goto/ILC/Deep_Learning/data/vfdnn_06_track_labels_sv.npy"
    name_tracks_pvsv = "/home/goto/ILC/Deep_Learning/data/vfdnn_06_tracks_pvsvnc.npy"
    name_track_pairs_sv = "/home/goto/ILC/Deep_Learning/data/vfdnn_06_track_pairs_sv.npy"
    name_track_plabels_sv = "/home/goto/ILC/Deep_Learning/data/vfdnn_06_track_plabels_sv.npy"

    data_track_labels = np.load(name_track_labels_sv, allow_pickle=True)
    data_tracks = np.load(name_tracks_pvsv, allow_pickle=True)
    data_track_pairs = np.load(name_track_pairs_sv, allow_pickle=True)
    data_track_plabels = np.load(name_track_plabels_sv, allow_pickle=True)

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

    model = LSTMMODEL.load_attention_model(model_name)
        
    # Tensor Board
    set_dir_name='Test_Attention_VLSTM_MODEL'
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    callbacks = [TensorBoard(log_dir=log_dir)]

    model.compile(loss=CUSTOM.binary_crossentropy(pair_reinforce=pair_reinforce),
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=[CUSTOM.accuracy_all, CUSTOM.accuracy,
                           CUSTOM.true_positive, CUSTOM.true_negative, CUSTOM.false_positive, CUSTOM.false_negative])

    print("Model Compile")
    for epochs in range(nb_epochs):
        Eindex = np.random.permutation(len(targets))
        Tindex = np.random.permutation(len(targets[0]))
        shuffle_tracks = []
        shuffle_labels = []
        for t, l in zip(teach_tracks[Eindex][:nb_samples], targets[Eindex][:nb_samples]):
            shuffle_tracks.append(t[Tindex])
            shuffle_labels.append(l[Tindex])
        shuffle_tracks, shuffle_labels = np.array(shuffle_tracks), np.array(shuffle_labels)
        print("===================== "
              + str(epochs+1) + "/" + str(nb_epochs) + " epochs" +
              " =====================")

        new_history = model.fit([teach_track_pairs[Eindex][:nb_samples], shuffle_tracks[:, :53, :23], shuffle_tracks[:, :, :23]], shuffle_labels,
                            batch_size=32,
                            epochs=1,
                            callbacks=callbacks,
                            verbose=1,
                            validation_split=0.2)

        if epochs == 0:
            history = {}
        history = MODELTOOLS.appendHist(history, new_history.history)

    
    MODELTOOLS.save_history(history, new_model_name)
    MODELTOOLS.save_attention_model(model, new_model_name)
