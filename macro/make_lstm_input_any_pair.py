import EVALTOOLS
import MODELBANK
import numpy as np
from keras.utils import np_utils
from tensorflow import keras


if __name__ == "__main__":
    # The track list is not sorted
    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_08_shaped.npy"
    model_name = "Pair_Model_vfdnn04_1Msamples_2500epochs"


    data = np.load(variable_name)
    print("File load successfully")

    event_nums = list(range(20000))
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
    track_labels_nc = []
    track_plabels_nc = []
    track_pairs_nc = []

    for event_num in event_nums:
        event_data = [datum for datum in data if datum[0]==event_num]

        track_num = (1 + np.sqrt(1 + 8*len(event_data)))/2 # 1,2,3,4,5,...
        print("The number of tracks in event {} is {}".format(event_num, track_num))
        vertex_mat_tpv = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
        vertex_mat_tsv = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
        vertex_mat_tnc = [0 for i in range(int(track_num))] # Max number of tracks

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
        track_label_nc = []
        track_plabel_nc = []
        track_pair_nc = []

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

        for event_datum in event_data:
            track_plabel_tmp = []
                
            track_plabel_tmp.append(vertex_mat_pnc[int(event_datum[1])])
            track_plabel_tmp.append(vertex_mat_pnc[int(event_datum[2])])
            track_plabel_tmp.append(vertex_mat_ppv[int(event_datum[1])])
            track_plabel_tmp.append(vertex_mat_ppv[int(event_datum[2])])
            track_plabel_tmp.append(vertex_mat_pcc[int(event_datum[1])])
            track_plabel_tmp.append(vertex_mat_pcc[int(event_datum[2])])
            track_plabel_tmp.append(vertex_mat_pbb[int(event_datum[1])])
            track_plabel_tmp.append(vertex_mat_pbb[int(event_datum[2])])
            track_plabel_tmp.append(vertex_mat_pbc[int(event_datum[1])])
            track_plabel_tmp.append(vertex_mat_pbc[int(event_datum[2])])

            # Primay Vertex 
            if event_datum[57] == 1:

                track_label_pv.append(vertex_mat_tpv[int(event_datum[1])])

                track_plabel_pv.append(track_plabel_tmp)

                track_pair_pv.append([track[int(event_datum[1])], track[int(event_datum[2])]])

            # Secondary Vertex 
            elif event_datum[57] == 2 or event_datum[57] == 3:

                track_label_sv.append(vertex_mat_tsv[int(event_datum[1])])

                track_plabel_sv.append(track_plabel_tmp)

                track_pair_sv.append([track[int(event_datum[1])], track[int(event_datum[2])]])

            # Not connected and secondary bc
            else:

                track_label_nc.append(vertex_mat_tnc)

                track_plabel_nc.append(track_plabel_tmp)

                track_pair_nc.append([track[int(event_datum[1])], track[int(event_datum[2])]])

        tracks.append(track)
        track_labels_pv.append(track_label_pv)
        track_plabels_pv.append(track_plabel_pv)
        track_pairs_pv.append(track_pair_pv)
        track_labels_sv.append(track_label_sv)
        track_plabels_sv.append(track_plabel_sv)
        track_pairs_sv.append(track_pair_sv)
        track_labels_nc.append(track_label_nc)
        track_plabels_nc.append(track_plabel_nc)
        track_pairs_nc.append(track_pair_nc)

    tracks = np.array(tracks)
    track_labels_pv = np.array(track_labels_pv)
    track_plabels_pv = np.array(track_plabels_pv)
    track_pairs_pv = np.array(track_pairs_pv)
    track_labels_sv = np.array(track_labels_sv)
    track_plabels_sv = np.array(track_plabels_sv)
    track_pairs_sv = np.array(track_pairs_sv)
    track_labels_nc = np.array(track_labels_nc)
    track_plabels_nc = np.array(track_plabels_nc)
    track_pairs_nc = np.array(track_pairs_nc)

    np.save("data/vfdnn_08_tracks_pvsvnc.npy", tracks, fix_imports=True)
    np.save("data/vfdnn_08_track_labels_pv.npy", track_labels_pv, fix_imports=True)
    np.save("data/vfdnn_08_track_plabels_pv.npy", track_plabels_pv, fix_imports=True)
    np.save("data/vfdnn_08_track_pairs_pv.npy", track_pairs_pv, fix_imports=True)
    np.save("data/vfdnn_08_track_labels_sv.npy", track_labels_sv, fix_imports=True)
    np.save("data/vfdnn_08_track_plabels_sv.npy", track_plabels_sv, fix_imports=True)
    np.save("data/vfdnn_08_track_pairs_sv.npy", track_pairs_sv, fix_imports=True)
    np.save("data/vfdnn_08_track_labels_nc.npy", track_labels_nc, fix_imports=True)
    np.save("data/vfdnn_08_track_plabels_nc.npy", track_plabels_nc, fix_imports=True)
    np.save("data/vfdnn_08_track_pairs_nc.npy", track_pairs_nc, fix_imports=True)
