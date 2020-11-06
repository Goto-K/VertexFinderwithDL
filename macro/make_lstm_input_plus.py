import EVALTOOLS
import MODELBANK
import numpy as np
from keras.utils import np_utils
from tensorflow import keras


if __name__ == "__main__":
    # The track list is not sorted
    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03.npy"
    model_name = "Comparison_Conv_OnlyLow_CustomLoss_Model_1500epochs"


    data = np.load(variable_name)
    print("File load successfully")

    event_nums = list(range(40000))
    print("Making event number list")


    variables = data[:, 3:47]
    model = MODELBANK.load_model(model_name)
    print("Model load successfully")
        
    model.compile(loss='categorical_crossentropy', 
                  optimizer=keras.optimizers.Adam(lr=0.001), 
                  metrics=['accuracy'])

    predict_vertex_finder = model.predict([variables])
    predict_vertex_finder = np.array(predict_vertex_finder, dtype=float)

    data = np.concatenate([data, predict_vertex_finder], 1)
    # data[:, -5], data[:, -4], data[:, -3], data[:, -2], data[:, -1]

    tracks = []
    track_labels_pv = []
    track_plabels_pv = []
    track_pairs_pv = []
    track_labels_sv = []
    track_plabels_sv = []
    track_pairs_sv = []

    for event_num in event_nums:
        event_data = [datum for datum in data if datum[0]==event_num]

        track_num = (1 + np.sqrt(1 + 8*len(event_data)))/2 # 1,2,3,4,5,...
        print("The number of tracks in event {} is {}".format(event_num, track_num))
        vertex_mat_tpv = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
        vertex_mat_tsv = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks

        vertex_mat_pnc = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
        vertex_mat_ppv = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
        vertex_mat_pcc = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
        vertex_mat_pbb = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
        vertex_mat_pbc = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
        
        for t in range(int(track_num)):
            vertex_mat_tpv[t][t] = 1
            vertex_mat_tsv[t][t] = 1
            
            vertex_mat_pnc[t][t] = 1
            vertex_mat_ppv[t][t] = 1
            vertex_mat_pcc[t][t] = 1
            vertex_mat_pbb[t][t] = 1
            vertex_mat_pbc[t][t] = 1

        track = []
        track_label_sv = []
        track_plabel_sv = []
        track_pair_sv = []
        track_label_pv = []
        track_plabel_pv = []
        track_pair_pv = []

        for event_datum in event_data:
            pri_vtx = 1 if event_datum[57] == 1 else 0
            sec_vtx = 1 if event_datum[57] == 2 or event_datum[57] == 3 else 0
            vertex_mat_tpv[int(event_datum[1])][int(event_datum[2])] = pri_vtx
            vertex_mat_tpv[int(event_datum[2])][int(event_datum[1])] = pri_vtx
            vertex_mat_tsv[int(event_datum[1])][int(event_datum[2])] = sec_vtx
            vertex_mat_tsv[int(event_datum[2])][int(event_datum[1])] = sec_vtx

            vertex_mat_pnc[int(event_datum[1])][int(event_datum[2])] = event_datum[-5]
            vertex_mat_pnc[int(event_datum[2])][int(event_datum[1])] = event_datum[-5]
            vertex_mat_ppv[int(event_datum[1])][int(event_datum[2])] = event_datum[-4]
            vertex_mat_ppv[int(event_datum[2])][int(event_datum[1])] = event_datum[-4]
            vertex_mat_pcc[int(event_datum[1])][int(event_datum[2])] = event_datum[-3]
            vertex_mat_pcc[int(event_datum[2])][int(event_datum[1])] = event_datum[-3]
            vertex_mat_pbb[int(event_datum[1])][int(event_datum[2])] = event_datum[-2]
            vertex_mat_pbb[int(event_datum[2])][int(event_datum[1])] = event_datum[-2]
            vertex_mat_pbc[int(event_datum[1])][int(event_datum[2])] = event_datum[-1]
            vertex_mat_pbc[int(event_datum[2])][int(event_datum[1])] = event_datum[-1]

            if int(event_datum[1]) == int(track_num)-1:
                #track.append([event_num, event_datum[2], event_datum[25:47]])
                track.append(event_datum[25:47])
                if int(event_datum[2]) == int(track_num)-2:
                    track.append(event_datum[3:25])

        # Primay Vertex 
        for t in range(int(track_num)):
            tlist_pv = vertex_mat_tpv[t]
            index_pv = [i for i, x in enumerate(tlist_pv) if x == 1]
            if len(index_pv) == 1:
                continue
            else:
                track_plabel_tmp = []
                track_label_pv.append(vertex_mat_tpv[t])

                track_plabel_tmp.append(vertex_mat_pnc[t])
                track_plabel_tmp.append(vertex_mat_pnc[index_pv[0]])
                track_plabel_tmp.append(vertex_mat_ppv[t])
                track_plabel_tmp.append(vertex_mat_ppv[index_pv[0]])
                track_plabel_tmp.append(vertex_mat_pcc[t])
                track_plabel_tmp.append(vertex_mat_pcc[index_pv[0]])
                track_plabel_tmp.append(vertex_mat_pbb[t])
                track_plabel_tmp.append(vertex_mat_pbb[index_pv[0]])
                track_plabel_tmp.append(vertex_mat_pbc[t])
                track_plabel_tmp.append(vertex_mat_pbc[index_pv[0]])

                track_plabel_pv.append(track_plabel_tmp)

                track_pair_pv.append([track[t], track[index_pv[0]]])

        # Secondary Vertex 
        for t in range(int(track_num)):
            tlist_sv = vertex_mat_tsv[t]
            index_sv = [i for i, x in enumerate(tlist_sv) if x == 1]
            if len(index_sv) == 1:
                continue
            else:
                track_plabel_tmp = []
                track_label_sv.append(vertex_mat_tsv[t])

                track_plabel_tmp.append(vertex_mat_pnc[t])
                track_plabel_tmp.append(vertex_mat_pnc[index_sv[0]])
                track_plabel_tmp.append(vertex_mat_ppv[t])
                track_plabel_tmp.append(vertex_mat_ppv[index_sv[0]])
                track_plabel_tmp.append(vertex_mat_pcc[t])
                track_plabel_tmp.append(vertex_mat_pcc[index_sv[0]])
                track_plabel_tmp.append(vertex_mat_pbb[t])
                track_plabel_tmp.append(vertex_mat_pbb[index_sv[0]])
                track_plabel_tmp.append(vertex_mat_pbc[t])
                track_plabel_tmp.append(vertex_mat_pbc[index_sv[0]])

                track_plabel_sv.append(track_plabel_tmp)

                track_pair_sv.append([track[t], track[index_sv[0]]])

        tracks.append(track)
        track_labels_pv.append(track_label_pv)
        track_plabels_pv.append(track_plabel_pv)
        track_pairs_pv.append(track_pair_pv)
        track_labels_sv.append(track_label_sv)
        track_plabels_sv.append(track_plabel_sv)
        track_pairs_sv.append(track_pair_sv)

    tracks = np.array(tracks)
    track_labels_pv = np.array(track_labels_pv)
    track_plabels_pv = np.array(track_plabels_pv)
    track_pairs_pv = np.array(track_pairs_pv)
    track_labels_sv = np.array(track_labels_sv)
    track_plabels_sv = np.array(track_plabels_sv)
    track_pairs_sv = np.array(track_pairs_sv)

    np.save("vfdnn_03_tracks_pvsv.npy", tracks, fix_imports=True)
    np.save("vfdnn_03_track_labels_pv.npy", track_labels_pv, fix_imports=True)
    np.save("vfdnn_03_track_plabels_pv.npy", track_plabels_pv, fix_imports=True)
    np.save("vfdnn_03_track_pairs_pv.npy", track_pairs_pv, fix_imports=True)
    np.save("vfdnn_03_track_labels_sv.npy", track_labels_sv, fix_imports=True)
    np.save("vfdnn_03_track_plabels_sv.npy", track_plabels_sv, fix_imports=True)
    np.save("vfdnn_03_track_pairs_sv.npy", track_pairs_sv, fix_imports=True)
