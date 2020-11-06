import EVALTOOLS
import MODELTOOLS
import LSTMMODEL
import MODELBANK
import numpy as np
import itertools
from keras.utils import np_utils
from tensorflow import keras
import os
import copy
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

    vth = 0.5
    pth = 0.7
    primary_loop = 3

    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_08_shaped.npy"
    nonvariable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_08.npy"

    pair_model_name = "Pair_Model_vfdnn04_1Msamples_2500epochs"

    lstm_model_name = "Attention_Bidirectional_VLSTM_Model_vfdnn06_50000samples_100epochs"
    slstm_model_name = "Attention_Bidirectional_VLSTM_Model_vfdnn06_50000samples_100epochs"


    event_nums = list(range(10))
    print("Making " + str(event_nums) + " event number list")


    #data load
    data = np.load(variable_name)
    nondata = np.load(nonvariable_name)
    print("File load successfully")


    variables = data[:10000, 3:47]
    pair_model = MODELBANK.load_model(pair_model_name)
    print("Pair Model load successfully")
        
    pair_model.compile(loss='categorical_crossentropy', 
                       optimizer=keras.optimizers.Adam(lr=0.001), 
                       metrics=['accuracy'])

    lstm_model = LSTMMODEL.load_attention_model(lstm_model_name)
    slstm_model = LSTMMODEL.load_attention_model(slstm_model_name)
    print("VLSTM Model load successfully")
        
    lstm_model.compile(loss='binary_crossentropy', 
                       optimizer=keras.optimizers.Adam(lr=0.001),
                       metrics=['accuracy'])
    slstm_model.compile(loss='binary_crossentropy', 
                       optimizer=keras.optimizers.Adam(lr=0.001),
                       metrics=['accuracy'])
    

    print("Pair model prediction ...")
    predict_vertex_finder = pair_model.predict([variables])
    predict_vertex_finder = np.array(predict_vertex_finder, dtype=float)

    data = np.concatenate([data[:10000], predict_vertex_finder], 1)
    # data[:, -5], data[:, -4], data[:, -3], data[:, -2], data[:, -1]

    predict_vertex_labels = np.argmax(predict_vertex_finder, axis=1)
    # score -> label (one hot)


    vertices_list = []
    for event_num in event_nums:
        print(pycolor.ACCENT + "=================================================================================================" + pycolor.END)
        print(pycolor.ACCENT + "EVENT NUMBER " + str(event_num) + pycolor.END)
        print(pycolor.ACCENT + "=================================================================================================" + pycolor.END)
        event_data = [datum for datum in data if datum[0]==event_num]
        event_nondata = [datum for datum in nondata if datum[0]==event_num]
        pred_labels = [pred_label for datum, pred_label in zip(data, predict_vertex_labels) if datum[0]==event_num]

        track_num = (1 + np.sqrt(1 + 8*len(event_data)))/2 # 1,2,3,4,5,...
        print("The number of tracks in event {} is {}".format(event_num, track_num))

        if track_num < 15:
            print("The number of Track is very small ==> Skip !! Next Event !!")
            continue

        # Zero matrix ture bc
        vertex_mat_tbc = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks

        # Zero matrix ture cc bb bc
        ppccbb_list = ["o"  for i in range(int(track_num))] # Max number of tracks
        chain_list = [0  for i in range(int(track_num))] # Max number of tracks
        
        # same track = 1
        for t in range(int(track_num)):
            vertex_mat_tbc[t][t] = 1

        # Event track list & Predicted score matrix
        track_list = []
        true_pvertex_list_index = []
        true_ccvertex_list_index = []
        true_bbvertex_list_index = []
        for event_datum in event_data:

            if int(event_datum[1]) == int(track_num)-1:
                #track.append([event_num, event_datum[2], event_datum[25:47]])
                track_list.append(np.concatenate([[1], event_datum[25:47]]))
                if int(event_datum[2]) == int(track_num)-2:
                    track_list.append(np.concatenate([[1], event_datum[3:25]]))
            # track_list[track_num, 23]

        # ================================================================================================= #
        # Making True vertex ============================================================================== #
        # ================================================================================================= #
            # True primary vertex ver1
            if event_datum[57] == 1:
                true_pvertex_list_index.append([int(event_datum[1]), int(event_datum[2])])
            # True secondary vertex ver3
            elif event_datum[57] == 2:
                true_ccvertex_list_index.append([int(event_datum[1]), int(event_datum[2])])
            elif event_datum[57] == 3:
                true_bbvertex_list_index.append([int(event_datum[1]), int(event_datum[2])])
            elif event_datum[57] == 4:
                vertex_mat_tbc[int(event_datum[1])][int(event_datum[2])] = 1
                vertex_mat_tbc[int(event_datum[2])][int(event_datum[1])] = 1
        
        vertex_mat_tbc = np.array(vertex_mat_tbc)

        print("Calclation vertex chain lists ...")
        true_bc_lists = []
        for t in range(int(track_num)):
            true_bc_list1 = [i for i, x in enumerate(vertex_mat_tbc[t, :]) if x == 1]
            if len(true_bc_list1)>1:
                true_bc_list_tmp = copy.deepcopy(true_bc_list1)
                for u in range(int(track_num)):
                    if u == t: continue
                    true_bc_list2 = [i for i, x in enumerate(vertex_mat_tbc[u, :]) if x == 1]
                    if len(true_bc_list2)>1:
                        check = copy.deepcopy(true_bc_list1)
                        check.extend(true_bc_list2)
                        seen = []
                        unique_list = [x for x in check if x not in seen and not seen.append(x)]
                        if len(check) != len(unique_list):
                            true_bc_list_tmp.extend(true_bc_list2)
                            true_bc_list_tmp = list(set(true_bc_list_tmp))
                true_bc_list_tmp = sorted(list(set(true_bc_list_tmp)))
                if true_bc_list_tmp not in true_bc_lists:
                    true_bc_lists.append(true_bc_list_tmp)

        true_pvertex_list_index = list(set(list(itertools.chain.from_iterable(true_pvertex_list_index))))
        for true_pvertex in true_pvertex_list_index:
            ppccbb_list[int(true_pvertex)] = "p"
        true_ccvertex_list_index = list(set(list(itertools.chain.from_iterable(true_ccvertex_list_index))))
        for true_ccvertex in true_ccvertex_list_index:
            ppccbb_list[int(true_ccvertex)] = "c"
        true_bbvertex_list_index = list(set(list(itertools.chain.from_iterable(true_bbvertex_list_index))))
        for true_bbvertex in true_bbvertex_list_index:
            ppccbb_list[int(true_bbvertex)] = "b"
        for i, true_bc_list in enumerate(true_bc_lists):
            for true_bc in true_bc_list:
                chain_list[int(true_bc)] = i+1


        # ===================================================================================================== #
        # Secondary vertex third track ======================================================================== #
        # ===================================================================================================== #
        # Pair track list
        pairs = []
        tracks = []
        attention_tracks = []
        num_pair = 0
        secondary_index = []
        for event_datum, pred_label, event_nondatum in zip(event_data, pred_labels, event_nondata):
            if (pred_label != 2 and pred_label != 3 and pred_label != 4) or event_nondatum[47]>100: # this event datum is not "predicted secondary vertex"
                continue
            else:
                pairs.append([track_list[int(event_datum[1])][1:], track_list[int(event_datum[2])][1:]])
                
                # score matrix[track_num, track_num]

                track = np.array(track_list)
                attention_track = np.pad(track, [(0, 53-int(track_num)), (0, 0)])
                track = track.reshape([1, int(track_num), 23])
                attention_track = attention_track.reshape([1, 53, 23])
                tracks.append(track)
                attention_tracks.append(attention_track)

                num_pair = num_pair + 1 # faster than +=1
                secondary_index.append([int(event_datum[1]), int(event_datum[2])])
        
        pairs = np.array(pairs).reshape([-1, 44])
        tracks = np.array(tracks).reshape([-1, int(track_num), 23])
        attention_tracks = np.array(attention_tracks).reshape([-1, 53, 23])
        predict_vlstm = slstm_model.predict([pairs, attention_tracks, tracks])
        # predict_vlstm[num_pair, num_track]
        predict_vlstm = np.array(predict_vlstm).reshape([-1, int(track_num)])

        three_probabilities = []
        for pred_vlstm in predict_vlstm:
            pred_vlstm_sort = np.sort(pred_vlstm)
            # ascending order
            three_probability = pred_vlstm[-1] + pred_vlstm[-2] + pred_vlstm[-3]
            # pair + one track
            three_probabilities.append(three_probability)

        three_probabilities = np.array(three_probabilities).reshape([-1, 1])

        three_probabilities_and_secondary_index = np.concatenate([three_probabilities, secondary_index], 1)
        #three_probabilities_and_secondary_index[secondary_num][prob, pair1, pair2]

        #three_probability_index = np.argsort(three_probabilities)
        three_probability_index = np.argsort(three_probabilities_and_secondary_index[:, 0])
        three_probability_sort = three_probabilities_and_secondary_index[three_probability_index]
        # ascending order
        
        track_list_index = np.arange(0, track_num, 1)

        secondary_index = np.array(secondary_index)
        secondary_index_sort = np.array(three_probability_sort[:, 1:])

        # ===================================================================================================== #
        # Primary vertex first check ========================================================================== #
        # ===================================================================================================== #
        event_data = np.array(event_data)
        primary_score_index = event_data[np.argsort(event_data[:, -4])]
                
        pvertex_list_index = []
        predict_pvlstm_list = []
        for i in range(primary_loop):
            pevent_datum = primary_score_index[int(-(i+1))]
            ppair = [track_list[int(pevent_datum[1])][1:], track_list[int(pevent_datum[2])][1:]]
            ppair = np.array(ppair).reshape([1, 44])
                
            ptrack = np.array(track_list)
            attention_ptrack = np.pad(ptrack, [(0, 53-int(track_num)), (0, 0)])
            ptrack = ptrack.reshape([1, int(track_num), 23])
            attention_ptrack = attention_ptrack.reshape([1, 53, 23])
            
            predict_pvlstm = lstm_model.predict([ppair, attention_ptrack, ptrack])
            predict_pvlstm_list.append(predict_pvlstm)
            predict_pvlstmnp = np.array(predict_pvlstm).reshape([-1])
            predict_pvlstmnp[predict_pvlstmnp > pth] = 1
            pvertex_list_index_ = [i for i, s in enumerate(predict_pvlstmnp) if s == 1] # predicted tracks in primary vertex
            pvertex_list_index.append(pvertex_list_index_)
        
        predict_pvlstm_ = []
        predict_pvlstm_list = np.array(predict_pvlstm_list).reshape([primary_loop, int(track_num)])
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
        print("Primary Vertex first check")
        print(pvertex_list_index)
        pvertex_list_index = np.array(pvertex_list_index, dtype=int)

        
        # ===================================================================================================== #
        # Secondary vertex build up =========================================================================== #
        # ===================================================================================================== #
        vertices = []
        seed_list = []
        next_pair = 1
        track_list_index = np.array(track_list_index, dtype=int)
        pairs = pairs[three_probability_index]
        tracks = tracks[three_probability_index]
        attention_tracks = attention_tracks[three_probability_index]
        while next_pair < len(secondary_index_sort):
            new_pairs = pairs[-next_pair].reshape([-1, 44])
            _tracks = tracks[-next_pair]
            new_attention_tracks = attention_tracks[-next_pair].reshape([-1, 53, 23])
            if secondary_index_sort[-next_pair][0] in pvertex_list_index or secondary_index_sort[-next_pair][1] in pvertex_list_index:
                next_pair = next_pair + 1
                continue
            new_tracks = _tracks[track_list_index].reshape([-1, len(track_list_index), 23])
            predict_vlstm_ite = slstm_model.predict([new_pairs, new_attention_tracks, new_tracks])

            seed_list.append(next_pair)

            # > threshold = 1
            predict_vlstm_itenp = np.array(predict_vlstm_ite).reshape([-1])
            predict_vlstm_itenp[predict_vlstm_itenp > vth] = 1
            
            # track list index update
            vertex_list_index = [i for i, s in zip(track_list_index, predict_vlstm_itenp) if s == 1]

            # check whether the track is in primary vertex
            #print(vertex_list_index)
            vertex_list_index = np.array(vertex_list_index, dtype=int).reshape([-1])
            for vli in vertex_list_index:
                #print(vli)
                if vli in pvertex_list_index:
                    if predict_pvlstm_bigger[vli] > predict_vlstm_itenp[np.where(vertex_list_index==vli)]:
                        #print(vli)
                        vertex_list_index = np.delete(vertex_list_index, np.where(vertex_list_index==vli))
                    else:
                        pvertex_list_index = np.delete(pvertex_list_index, np.where(pvertex_list_index==vli))

            vertices.append(vertex_list_index) # track list associate to secondary vertex
            print("Secondary Vertex seed " + str(secondary_index_sort[-next_pair][0]) + " and " + str(secondary_index_sort[-next_pair][1]))
            print("Secondary Vertex check")
            print(vertex_list_index)

            #track_list_index = list(track_list_index)
            for i in vertex_list_index:
                track_list_index = np.delete(track_list_index, np.where(track_list_index==i))
                #track_list_index.pop(i) # removing track from track list

            print("Remaining Track list")
            print(track_list_index)
            # decision of the next seed

            for j, secondary in enumerate(secondary_index_sort[::-1]):
                # both tracks are not removed from track list
                next_pair = j + 1
                if (int(secondary[0]) in track_list_index and int(secondary[1]) in track_list_index) and \
                   (int(secondary[0]) not in pvertex_list_index and int(secondary[1]) not in pvertex_list_index):
                    if next_pair in seed_list:
                        continue
                    else:
                        break
        
        #print(track_list_index)
        # ===================================================================================================== #
        # Primary vertex removing ============================================================================= #
        # ===================================================================================================== #
        #print("Primary Vertex final check")
        #print(pvertex_list_index)
        track_list_index = list(track_list_index)
        for i in pvertex_list_index:
            track_list_index = np.delete(track_list_index, np.where(track_list_index==i))
            #track_list_index.pop(i) # removing track from track list

        """
        # ===================================================================================================== #
        # Remaining tracks ==================================================================================== #
        # ===================================================================================================== #
        print("Remaining track list")
        if len(track_list_index) != 0:
            print(track_list_index)
            predict_vlstm_remain_tracks = []
            for i in track_list_index:
                remain_score = predict_pvlstm_bigger[i]
                vertex_remain = 0
                for seed in seed_list:
                    if predict_vlstm[three_probability_index[-seed]][i] > remain_score:
                        vertex_remain = seed
                if vertex_remain == 0:
                    pvertex_list_index = np.append(pvertex_list_index, i)
                else:
                    vertices[seed_list.index(vertex_remain)] = np.append(vertices[seed_list.index(vertex_remain)], i)
        """
        print("Remaining track list")
        if len(track_list_index) != 0:
            print(track_list_index)

        # ===================================================================================================== #
        # Finish ============================================================================================== #
        # ===================================================================================================== #
        print("Finish !!")
        #print("Test True Vertex List")
        #print(ppccbb_list)
        #print(chain_list)
        print(pycolor.YELLOW + "True Primary Vertex" + pycolor.END)
        tp = [i for i, x in enumerate(ppccbb_list) if x == "p"]
        print(pycolor.YELLOW + str(tp) + pycolor.END)
        print("Predict Primary Vertex")
        pvertex_list_index = np.sort(pvertex_list_index)
        print(list(pvertex_list_index))
        for bc in range(len(true_bc_lists)):
            tcc = [i for i, (x, c) in enumerate(zip(ppccbb_list, chain_list)) if x == "c" and c == bc+1]
            tbb = [i for i, (x, c) in enumerate(zip(ppccbb_list, chain_list)) if x == "b" and c == bc+1]
            one = [i for i, (x, c) in enumerate(zip(ppccbb_list, chain_list)) if x == "o" and c == bc+1]
            print(pycolor.CYAN + "True Secondary Vertex Chain " + str(bc+1) + pycolor.END)
            print(pycolor.CYAN + "cc : " + str(tcc) + pycolor.END)
            print(pycolor.CYAN + "bb : " + str(tbb) + pycolor.END)
            print(pycolor.CYAN + "one track : " + str(one) + pycolor.END)
        seco_num = 0
        recos = []
        for vertex in vertices:
            if len(vertex) > 0:
                print("Predict Secondary Vertex " + str(seco_num))
                vertex = np.sort(vertex)
                print(list(vertex))
                seco_num = seco_num + 1
                recos.extend(vertex)

        mcb = [i for i, x in enumerate(ppccbb_list) if x == "b"]
        mcc = [i for i, x in enumerate(ppccbb_list) if x == "c"]

        mcbrcs = 0
        mcbrcsc = 0
        mcbrcsp = 0
        mccrcs = 0
        mccrcsc = 0
        mccrcsp = 0

        for _mcb in mcb:
            if _mcb in recos: mcbrcs = mcbrcs + 1

        if 

