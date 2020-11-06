import EVALTOOLS
import MODELTOOLS
import LSTMMODEL
import MODELBANK
import numpy as np
import itertools
from keras.utils import np_utils
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RETURN = '\033[07m' #反転
    ACCENT = '\033[01m' #強調
    FLASH = '\033[05m' #点滅
    RED_FLASH = '\033[05;41m' #赤背景+点滅
    END = '\033[0m'


if __name__ == "__main__":

    max_event = 100
    event_counter = 0
    vth = 0.5
    pth = 0.5
    primary_loop = 3
    Seed_accuracy = 0
    Efficiency = 0
    Purity = 0

    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_08_shaped.npy"
    helpdata_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_08.npy"
    pair_model_name = "Pair_Model_vfdnn04_1Msamples_2500epochs"
    
    #lstm_model_name = "VLSTM_Model_InitState_ZeroPadding_Masking_Shuffle_2denseBNact_50000samples_32nodes_50epochs_1LSTM_any_pairs_pair_loss"
    lstm_model_name = "Attention_Bidirectional_VLSTM_Model_vfdnn06_50000samples_100epochs"


    event_nums = list(range(max_event))
    print("Making " + str(event_nums) + " event number list")


    #data load
    data = np.load(variable_name)
    helpdata = np.load(helpdata_name)
    print("File load successfully")


    variables = data[:100000, 3:47]
    pair_model = MODELBANK.load_model(pair_model_name)
    print("Pair Model load successfully")
        
    pair_model.compile(loss='categorical_crossentropy', 
                       optimizer=keras.optimizers.Adam(lr=0.001), 
                       metrics=['accuracy'])

    lstm_model = LSTMMODEL.load_attention_model(lstm_model_name)
    print("VLSTM Model load successfully")
        
    lstm_model.compile(loss='binary_crossentropy', 
                       optimizer=keras.optimizers.Adam(lr=0.001),
                       metrics=['accuracy'])
    

    print("Pair model prediction ...")
    predict_vertex_finder = pair_model.predict([variables])
    predict_vertex_finder = np.array(predict_vertex_finder, dtype=float)

    data = np.concatenate([data[:100000], predict_vertex_finder], 1)
    # data[:, -5], data[:, -4], data[:, -3], data[:, -2], data[:, -1]

    predict_vertex_labels = np.argmax(predict_vertex_finder, axis=1)
    # score -> label (one hot)


    vertices_list = []
    for event_num in event_nums:
        print(pycolor.ACCENT + "=================================================================================================" + pycolor.END)
        print(pycolor.ACCENT + "EVENT NUMBER " + str(event_num) + pycolor.END)
        print(pycolor.ACCENT + "=================================================================================================" + pycolor.END)
        event_data = [datum for datum in data if int(datum[0])==int(event_num)]
        event_helpdata = [datum for datum in helpdata if int(datum[0])==int(event_num)]
        pred_labels = [pred_label for datum, pred_label in zip(data, predict_vertex_labels) if datum[0]==event_num]

        track_num = (1 + np.sqrt(1 + 8*len(event_data)))/2 # 1,2,3,4,5,...
        print("The number of tracks in event {} is {}".format(event_num, track_num))

        if track_num < 15:
            print("The number of Track is very small ==> Skip !! Next Event !!")
            continue
        event_counter = event_counter + 1

        """
        # Zero matrix
        vertex_mat_pnc = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
        vertex_mat_ppv = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
        vertex_mat_pcc = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
        vertex_mat_pbb = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
        vertex_mat_pbc = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
         
        # same track = 1
        for t in range(int(track_num)):
            vertex_mat_pnc[t][t] = 1
            vertex_mat_ppv[t][t] = 1
            vertex_mat_pcc[t][t] = 1
            vertex_mat_pbb[t][t] = 1
            vertex_mat_pbc[t][t] = 1
        """

        # Event track list & Predicted score matrix
        track_list = []
        true_pvertex_list_index = []
        true_vertex_list_index = []
        for event_datum in event_data:

            """
            # predicted score matrix
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
            # score matrix[track_num, track_num]
            """

            if int(event_datum[1]) == int(track_num)-1:
                #track.append([event_num, event_datum[2], event_datum[25:47]])
                track_list.append(np.concatenate([[1], event_datum[25:47]]))
                if int(event_datum[2]) == int(track_num)-2:
                    track_list.append(np.concatenate([[1], event_datum[3:25]]))
            # track_list[track_num, 23]

            if event_datum[57] == 1:
                true_pvertex_list_index.append([int(event_datum[1]), int(event_datum[2])])
        
            if event_datum[57] == 2 or event_datum[57] == 3:
                true_vertex_list_index.append([int(event_datum[1]), int(event_datum[2])])
        
        true_pvertex_list_index = list(itertools.chain.from_iterable(true_pvertex_list_index))
        true_pvertex_list_index = list(set(true_pvertex_list_index))
        true_vertex_list_index = list(itertools.chain.from_iterable(true_vertex_list_index))
        true_vertex_list_index = list(set(true_vertex_list_index))

        # ===================================================================================================== #
        # Secondary vertex third track ======================================================================== #
        # ===================================================================================================== #
        # Pair track list
        pairs = []
        tracks = []
        """
        vertex_mat_pnc = np.array(vertex_mat_pnc)
        vertex_mat_ppv = np.array(vertex_mat_ppv)
        vertex_mat_pcc = np.array(vertex_mat_pcc)
        vertex_mat_pbb = np.array(vertex_mat_pbb)
        vertex_mat_pbc = np.array(vertex_mat_pbc)
        """
        # ===================================================================================================== #
        # Primary vertex first check ========================================================================== #
        # ===================================================================================================== #
        event_data = np.array(event_data)
        event_helpdata = np.array(event_helpdata)
        primary_score_index = event_data[np.argsort(event_data[:, -4])]
        sorted_event_helpdata = event_helpdata[np.argsort(event_data[:, -4])]
                
        pvertex_list_index = []
        predict_pvlstm_list = []
        for i in range(primary_loop):
            pevent_datum = primary_score_index[int(-(i+1))]
            pevent_helpdatum = sorted_event_helpdata[int(-(i+1))]
            if pevent_helpdatum[47] > 100: continue
            print("Seed " + str(int(pevent_datum[1])) + " and " + str(int(pevent_datum[2])))
            if pevent_datum[1] in true_pvertex_list_index and pevent_datum[2] in true_pvertex_list_index:
                print(pycolor.CYAN + "Seed is existence in True Primary List" + pycolor.END)
                Seed_accuracy = Seed_accuracy + 1
            ppair = [track_list[int(pevent_datum[1])][1:], track_list[int(pevent_datum[2])][1:]]
            ppair = np.array(ppair).reshape([1, 44])
                
            """
            ptrack = np.concatenate([track_list, 
                                     vertex_mat_pnc[int(pevent_datum[1]), :].reshape([int(track_num), 1]),
                                     vertex_mat_pnc[int(pevent_datum[2]), :].reshape([int(track_num), 1]), 
                                     vertex_mat_ppv[int(pevent_datum[1]), :].reshape([int(track_num), 1]),
                                     vertex_mat_ppv[int(pevent_datum[2]), :].reshape([int(track_num), 1]), 
                                     vertex_mat_pcc[int(pevent_datum[1]), :].reshape([int(track_num), 1]),
                                     vertex_mat_pcc[int(pevent_datum[2]), :].reshape([int(track_num), 1]), 
                                     vertex_mat_pbb[int(pevent_datum[1]), :].reshape([int(track_num), 1]),
                                     vertex_mat_pbb[int(pevent_datum[2]), :].reshape([int(track_num), 1]), 
                                     vertex_mat_pbc[int(pevent_datum[1]), :].reshape([int(track_num), 1]),
                                     vertex_mat_pbc[int(pevent_datum[2]), :].reshape([int(track_num), 1])], 1)
            """
            
            ptrack = np.array(track_list)
            attention_ptrack = np.pad(ptrack, [(0, 53-int(track_num)), (0, 0)])
            ptrack = ptrack.reshape([1, int(track_num), 23])
            attention_ptrack = attention_ptrack.reshape([1, 53, 23])

            predict_pvlstm = lstm_model.predict([ppair, attention_ptrack, ptrack])
            predict_pvlstm_list.append(predict_pvlstm)
            predict_pvlstmnp = np.array(predict_pvlstm).reshape([-1])
            print(list(predict_pvlstmnp))
            predict_pvlstmnp[predict_pvlstmnp > pth] = 1
            pvertex_list_index_ = [i for i, s in enumerate(predict_pvlstmnp) if s == 1] # predicted tracks in primary vertex
            print(pycolor.YELLOW + str(pvertex_list_index_) + pycolor.END)
            pvertex_list_index.append(pvertex_list_index_)
        
        predict_pvlstm_ = []
        try:
            predict_pvlstm_list = np.array(predict_pvlstm_list).reshape([primary_loop, int(track_num)])
        except ValueError: 
            predict_pvlstm_list = np.array(predict_pvlstm_list).reshape([-1, int(track_num)])
        for pred_pvlstm_list in predict_pvlstm_list.T: # num_track
            pred_p = 0
            for pred_vlstm in pred_pvlstm_list: # 10 times
                pred_p = pred_vlstm if pred_vlstm > pred_p else pred_p
                #pred_p = pred_p + pred_vlstm
            predict_pvlstm_.append(pred_p)
            #predict_pvlstm_.append(pred_p/primary_loop)
        
        predict_pvlstm_bigger = np.array(predict_pvlstm_).reshape([-1])

        pvertex_list_index = list(itertools.chain.from_iterable(pvertex_list_index))
        pvertex_list_index = list(set(pvertex_list_index))
        pvertex_list_index = np.array(pvertex_list_index, dtype=int)
        
        # ===================================================================================================== #
        # Finish ============================================================================================== #
        # ===================================================================================================== #
        print("Finish !!")
        print("True Primary Vertex")
        true_pvertex_list_index = list(np.sort(true_pvertex_list_index))
        print(true_pvertex_list_index)
        print("Primary Vertex")
        pvertex_list_index = list(np.sort(pvertex_list_index))
        print(pvertex_list_index)
        E = 0
        P = 0
        for true in true_pvertex_list_index:
            if true in pvertex_list_index: E = E + 1
        for pred in pvertex_list_index:
            if pred not in true_pvertex_list_index: P = P + 1
        if len(pvertex_list_index) == 0 and len(true_pvertex_list_index) == 0:
            Efficiency = Efficiency + 1
            Purity = Purity + 1
        elif len(pvertex_list_index) != 0 and len(true_pvertex_list_index) != 0:
            Efficiency = Efficiency + E / len(true_pvertex_list_index)
            Purity = Purity + P / len(pvertex_list_index)
        elif len(pvertex_list_index) == 0:
            Efficiency = Efficiency + E / len(true_pvertex_list_index)
            Purity = Purity + 1
        elif len(true_pvertex_list_index) == 0:
            Efficiency = Efficiency + 1
            Purity = Purity + P / len(pvertex_list_index)

    print("Seed Check : S")
    print(Seed_accuracy / (primary_loop * event_counter))
    print("Evaluate Coincident : <C>")
    print("Efficiency : " + str(Efficiency / event_counter))
    print("Purity : " + str(Purity / event_counter))
