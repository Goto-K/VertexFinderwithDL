import EVALTOOLS
import MODELTOOLS
import LSTMMODEL
import numpy as np
from keras.utils import np_utils
from tensorflow import keras
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt



if __name__ == "__main__":

    encoder_input = 256
    decoder_units = 256
    nb_samples = 50000
    nb_epochs = 100
    pair_reinforce=False
    model_name = "Attention_VLSTM_Model_InitState_ZeroPadding_Masking_Shuffle_Bidirection_" \
                 + str(nb_samples) + "samples_" \
                 + str(nb_epochs) + "epochs"

    if pair_reinforce:
        model_name = model_name + "_pair_loss"

    model_name = "Attention_Bidirectional_VLSTM_Model_vfdnn06_50000samples_100epochs"

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

    Eindex = np.random.permutation(len(data_track_labels))

    print("file load !")

    print(data_track_labels.shape)
    print(data_tracks.shape)
    print(data_track_pairs.shape)

    teach_track_labels, teach_tracks, teach_track_pairs = MODELTOOLS.data_zero_padding(data_track_labels[Eindex][:10000], 
                                                                                       data_tracks[Eindex][:10000], 
                                                                                       data_track_pairs[Eindex][:10000], 
                                                                                       data_track_plabels[Eindex][:10000])

    Eindex = np.random.permutation(len(teach_track_labels))

    model = LSTMMODEL.load_attention_model(model_name)
        
    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    

    get_layer_output_model = Model(inputs=model.input,
                                   outputs=model.get_layer("Decoder_Attention_VLSTM").output)   
    predict, attention = get_layer_output_model.predict([teach_track_pairs[Eindex][:100], teach_tracks[Eindex][:100, :53, :23], teach_tracks[Eindex][:100, :, :23]])
    attention = attention.reshape([100, 53, 53])

    labels = teach_track_labels[Eindex][:100]
    for aw, l in zip(attention, labels):
        ititle = [i for i, ll in enumerate(l) if ll==1]
        title = "Connected track index\n" + str(ititle)
        plt.imshow(aw, cmap = 'gray', interpolation = 'none')
        plt.title(title)
        plt.xticks(np.arange(0, 53+1, 5.0)
        plt.rcParams["xtick.minor.visible"] = True
        plt.yticks(np.arange(0, 53+1, 5.0))
        plt.rcParams["ytick.minor.visible"] = True
        plt.show()
        plt.cla()


