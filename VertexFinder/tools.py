from Networks.Tools import modeltools
from copy import deepcopy
import numpy as np

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")



class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RETURN = '\033[07m'
    ACCENT = '\033[01m'
    FLASH = '\033[05m'
    RED_FLASH = '\033[05;41m'
    END = '\033[0m'


def ModelsLoad(pair_model_name, lstm_model_name, slstm_model_name):

    pair_model = modeltools.LoadPairModel(pair_model_name)
    lstm_model = modeltools.LoadAttentionVLSTMModel(lstm_model_name)
    slstm_model = modeltools.LoadAttentionVLSTMModel(slstm_model_name)
    pair_model.compile(loss={'Vertex_Output': 'categorical_crossentropy', 'Position_Output': 'mean_squared_logarithmic_error'},
                       optimizer='SGD',
                       metrics=['accuracy', 'mae'])
    lstm_model.compile(loss='binary_crossentropy',
                       optimizer="Adam",
                       metrics=['accuracy'])
    slstm_model.compile(loss='binary_crossentropy',
                        optimizer="Adam",
                        metrics=['accuracy'])

    return pair_model, lstm_model, slstm_model


def PairInference(pair_model, variables):

    predict_vertex, predict_position = pair_model.predict([variables], verbose=1)

    return predict_vertex, predict_position


def GetEncoderDecoderTracksandTrue(debug, event_data, NTrack, MaxTrack):

    tracks = []
    true_label = []
    chain_lists = []
    particle_lists = []
    chain_label = [0 for i in range(int(NTrack))]
    vertex_mat_tbbcc = [[0 for j in range(int(NTrack))] for i in range(int(NTrack))]
    vertex_mat_tbc = [[0 for j in range(int(NTrack))] for i in range(int(NTrack))]
    
    for event_datum in event_data:
        if event_datum[57] == 2 or event_datum[57] == 3 or event_datum[57] == 4:
            vertex_mat_tbbcc[int(event_datum[1])][int(event_datum[2])] = 1
            vertex_mat_tbbcc[int(event_datum[2])][int(event_datum[1])] = 1
        if event_datum[57] == 3 or event_datum[57] == 4 or event_datum[57] == 5:
            vertex_mat_tbc[int(event_datum[1])][int(event_datum[2])] = 1
            vertex_mat_tbc[int(event_datum[2])][int(event_datum[1])] = 1
        
        if int(event_datum[1]) == int(NTrack)-1:
            
            if(event_datum[71]==1): true_label.append("c")
            elif(event_datum[72]==1): true_label.append("b")
            elif(event_datum[73]==1): true_label.append("o")
            elif(event_datum[74]==1): true_label.append("p")
            else: true_label.append("u")
            
            tracks.append(np.concatenate([[1], event_datum[25:47]]))
            if int(event_datum[2]) == int(NTrack)-2:
             
                if(event_datum[63]==1): true_label.append("c")
                elif(event_datum[64]==1): true_label.append("b")
                elif(event_datum[65]==1): true_label.append("o")
                elif(event_datum[66]==1): true_label.append("p")
                else: true_label.append("u")
            
                tracks.append(np.concatenate([[1], event_datum[3:25]]))

    decoder_tracks = np.array(deepcopy(tracks))
    encoder_tracks = np.pad(np.array(deepcopy(tracks)), [(0, int(MaxTrack-NTrack)), (0, 0)])

    vertex_mat_tbbcc = np.array(vertex_mat_tbbcc)
    vertex_mat_tbc = np.array(vertex_mat_tbc)

    for t in range(int(NTrack)):
        vertex_mat_tbbcc[t][t] = 1
        vertex_mat_tbc[t][t] = 1
        tmp_particle_lists= [part for particle in particle_lists for part in particle]
        tmp_chain_lists= [ch for chain in chain_lists for ch in chain]
        particle_list = [i for i, x in enumerate(vertex_mat_tbbcc[t, :]) if x == 1]
        chain_list = [i for i, x in enumerate(vertex_mat_tbc[t, :]) if x == 1]
        if len(particle_list) > 1 and (t not in tmp_particle_lists): particle_lists.append(particle_list)
        if len(chain_list) > 1 and (t not in tmp_chain_lists): chain_lists.append(chain_list)

    for i, particle_list in enumerate(particle_lists):
        for particle in particle_list:
            chain_label[int(particle)] = -(i+1)
    
    for i, chain_list in enumerate(chain_lists):
        c = i + 1
        for chain in chain_list:
            if chain_label[int(chain)] > 0: c = chain_label[int(chain)]
        for chain in chain_list:
            chain_label[int(chain)] = c

    if debug==True: 
        print("Encoder Track Shape " + str(encoder_tracks.shape))
        print("Decoder Track Shape " + str(decoder_tracks.shape))
        print("True Label" + str(true_label))
        print("Chain Label" + str(chain_label))
        print(list(particle_lists))
        print(list(chain_lists))

    return encoder_tracks, decoder_tracks, true_label, chain_label


def SecondarySeedSelectionOne(debug, event_data, ThresholdPairSecondaryScore, ThresholdPairPosScore):

    predict_vertex_labels = np.argmax(event_data[:, -8:-1], axis=1)
    secondary_event_data = []
    tmp_secondary_scores = []
    for event_datum, predict_vertex_label in zip(event_data, predict_vertex_labels):
        tmp_secondary_score = event_datum[-6] + event_datum[-5] + event_datum[-4] + event_datum[-3]
        if predict_vertex_label==0 or predict_vertex_label==1 or  predict_vertex_label==6: continue
        if tmp_secondary_score < ThresholdPairSecondaryScore: continue
        if event_datum[-1] > ThresholdPairPosScore: continue
	
        secondary_event_data.append(event_datum)
        tmp_secondary_scores.append(tmp_secondary_score)
        
    tmp_secondary_scores = np.array(tmp_secondary_scores)
    secondary_event_data = np.array(secondary_event_data)
    index = np.argsort(-tmp_secondary_scores)

    if debug==True: 
        for i, secondary_event_datum in enumerate(secondary_event_data[index]):
            print("Secondary Seeds " + str(i) + " Track 1: " + str(secondary_event_datum[1]) + " Track 2: " + str(secondary_event_datum[2]) \
                  + " SV Score: " + str(secondary_event_datum[-6] + secondary_event_datum[-5] + secondary_event_datum[-4] + secondary_event_datum[-3]))
    
    return secondary_event_data[index]


#def SecondarySeedSelectionTwo(debug, event_data, ThresholdPairSecondaryScore, ThresholdPairPosScore):


def PrimaryVertexFinder(debug, MaxPrimaryVertexLoop, ThresholdPrimaryScore, event_data, 
                        encoder_tracks, decoder_tracks, lstm_model):

    primary_track_list = []
    bigger_primary_scores = []
    primary_pairs = []
    for event_datum in event_data[np.argsort(-event_data[:, -7])][:MaxPrimaryVertexLoop]:
        primary_pairs.append(event_datum[3:47])
    primary_encoder_tracks = np.tile(encoder_tracks, (MaxPrimaryVertexLoop, 1, 1)) # tracks.shape = (MaxPrimaryVertexLoop, MaxTrack, 23)
    primary_decoder_tracks = np.tile(decoder_tracks, (MaxPrimaryVertexLoop, 1, 1)) # tracks.shape = (MaxPrimaryVertexLoop, NTrack, 23)
    if debug==True: print("Primary Encoder Track Shape " + str(primary_encoder_tracks.shape))
    if debug==True: print("Primary Decoder Track Shape " + str(primary_decoder_tracks.shape))
    primary_scores = lstm_model.predict([primary_pairs, primary_encoder_tracks, primary_decoder_tracks])

    for i in range(len(primary_scores[0])): 
        tmpbigger_primary_scores = 0
        for j in range(len(primary_scores)):
            score = primary_scores[j][i]
            if tmpbigger_primary_scores < score: tmpbigger_primary_scores = score
        
        bigger_primary_scores.append(tmpbigger_primary_scores)
        if tmpbigger_primary_scores < ThresholdPrimaryScore: continue
        if debug==True: print("Track " + str(i) + " Primary Score: " + str(tmpbigger_primary_scores))
        primary_track_list.append(i)

    return primary_track_list, bigger_primary_scores


def SecondaryVertexFinder(debug, ThresholdSecondaryScore, bigger_primary_scores, primary_track_list, secondary_event_data, 
                          encoder_tracks, decoder_tracks, slstm_model):

    track_list = np.arange(decoder_tracks.shape[0])
    secondary_track_lists = []
    for secondary_event_datum in secondary_event_data:
        track1, track2 = secondary_event_datum[1], secondary_event_datum[2]
        if (track1 in primary_track_list) or (track2 in primary_track_list): continue
        if (track1 not in list(track_list)) or (track2 not in list(track_list)): continue

        #if debug==True: print("Track List " + str(track_list))
        
        remain_decoder_tracks = decoder_tracks[track_list]
        secondary_pair = np.tile(secondary_event_datum[3:47], (1, 1)) # pair.shape = (1, 44)
        secondary_encoder_tracks = np.tile(encoder_tracks, (1, 1, 1)) # tracks.shape = (1, MaxTrack, 23)
        secondary_decoder_tracks = np.tile(remain_decoder_tracks, (1, 1, 1)) # tracks.shape = (1, RemainNTrack, 23)
        secondary_scores = slstm_model.predict([secondary_pair, secondary_encoder_tracks, secondary_decoder_tracks])
        secondary_scores = np.array(secondary_scores).reshape((-1, 1))

        tmptrack_list = np.copy(track_list)
        primary_track_list = np.array(primary_track_list, dtype=int)
        tmpsecondary_track_list = []
        for i, t in enumerate(track_list):
            if secondary_scores[i] > ThresholdSecondaryScore:
                if t not in primary_track_list:
                    tmpsecondary_track_list.append(t)
                    tmptrack_list = tmptrack_list[~(tmptrack_list == t)]
                elif (t in primary_track_list) and (secondary_scores[i] > bigger_primary_scores[t]):
                    tmpsecondary_track_list.append(t)
                    if debug==True:
                        print("Scramble Track Number " + str(t) + " SV Score " + str(secondary_scores[i]) 
                              + " PV Score " + str(bigger_primary_scores[t]))
                    primary_track_list = primary_track_list[~(primary_track_list == t)]
                    tmptrack_list = tmptrack_list[~(tmptrack_list == t)]
        if len(tmpsecondary_track_list)!=0: secondary_track_lists.append(tmpsecondary_track_list)
        track_list = np.copy(tmptrack_list)

    return primary_track_list, secondary_track_lists


def CountPrintTrueTrackLists(debug, true_label, chain_label):

    ccbbvtx = 0
    bcvtx = 0
    true_secondary_bb_track_lists = []
    true_secondary_cc_track_lists = []
    true_secondary_same_chain_track_lists = []

    print(pycolor.YELLOW + "True Primary Vertex" + pycolor.END)
    true_primary_track_list = [i for i, x in enumerate(true_label) if x == "p"]
    print(pycolor.YELLOW + str(true_primary_track_list) + pycolor.END)

    for vtx in list(set(chain_label)):
        tcc = [i for i, (x, c) in enumerate(zip(true_label, chain_label)) if x == "c" and c == vtx]
        tbb = [i for i, (x, c) in enumerate(zip(true_label, chain_label)) if x == "b" and c == vtx]
        if vtx < 0:
            ccbbvtx = ccbbvtx + 1
            if len(tcc) != 0:
                print(pycolor.CYAN + "True Secondary Vertex Alone " + str(ccbbvtx) + pycolor.END)
                print(pycolor.CYAN + "cc : " + str(tcc) + pycolor.END)
                true_secondary_cc_track_lists.append(tcc)
                
            if len(tbb) != 0:
                print(pycolor.CYAN + "True Secondary Vertex Alone " + str(ccbbvtx) + pycolor.END)
                print(pycolor.CYAN + "bb : " + str(tbb) + pycolor.END)
                true_secondary_bb_track_lists.append(tbb)

        elif vtx > 0:
            bcvtx = ccbbvtx + 1
            print(pycolor.CYAN + "True Secondary Vertex Chain " + str(bcvtx) + pycolor.END)
            print(pycolor.CYAN + "cc : " + str(tcc) + pycolor.END)
            print(pycolor.CYAN + "bb : " + str(tbb) + pycolor.END)
            true_secondary_cc_track_lists.append(tcc)
            true_secondary_bb_track_lists.append(tbb)
            true_secondary_same_chain_track_lists.append([i for i, c in enumerate(chain_label) if c == vtx])

    print(pycolor.YELLOW + "True Other Tracks" + pycolor.END)
    true_other_track_list = [i for i, x in enumerate(true_label) if x == "o"]
    print(pycolor.YELLOW + str(true_other_track_list) + pycolor.END)

    return true_primary_track_list, true_secondary_bb_track_lists, true_secondary_cc_track_lists, true_secondary_same_chain_track_lists, true_other_track_list

        
def PrintPredTrackLists(primary_track_list, secondary_track_lists):

    print("Predict Primary Vertex")
    print(list(primary_track_list))

    for i, secondary_track_list in enumerate(secondary_track_lists):
        print("Predict Secondary Vertex " + str(i))
        print(list(secondary_track_list))


def yinx(y, x):
    for _y in y:
        if _y not in x: return False
    return True


def listremove(y, x):
    for _y in y:
        x.remove(_y)
    return x


def EvalPrintResults(debug, secondary_track_lists, true_primary_tracks, true_secondary_bb_track_lists, true_secondary_cc_track_lists, true_secondary_same_chain_track_lists, true_other_tracks,
		     MCPrimaryRecoSV, MCOthersRecoSV, MCBottomRecoSV, MCBottomRecoSVSameChain, MCBottomRecoSVSameParticle, MCCharmRecoSV, MCCharmRecoSVSameChain, MCCharmRecoSVSameParticle,
	             NumPVEvent, NumOthersEvent, NumBBEvent, NumCCEvent,
                     MCPrimaryRecoSVTrack, MCOthersRecoSVTrack, MCBottomRecoSVTrack, MCBottomRecoSVSameChainTrack, MCBottomRecoSVSameParticleTrack,
                     MCCharmRecoSVTrack, MCCharmRecoSVSameChainTrack, MCCharmRecoSVSameParticleTrack,
                     NumPVTrack, NumOthersTrack, NumBBTrack, NumCCTrack):

    true_secondary_bb_tracks = [track for tracks in true_secondary_bb_track_lists for track in tracks]
    true_secondary_cc_tracks = [track for tracks in true_secondary_cc_track_lists for track in tracks]
    true_secondary_same_particle_track_lists = true_secondary_bb_track_lists + true_secondary_cc_track_lists
    secondary_tracks = [track for tracks in secondary_track_lists for track in tracks]


    tmpMCPrimaryRecoSV = 0
    tmpMCOthersRecoSV = 0
    tmpMCBottomRecoSV = 0
    tmpMCBottomRecoSVSameChain = 0
    tmpMCBottomRecoSVSameParticle = 0
    tmpMCCharmRecoSV = 0
    tmpMCCharmRecoSVSameChain = 0
    tmpMCCharmRecoSVSameParticle = 0

    chains = deepcopy(secondary_tracks)
    particles = deepcopy(secondary_tracks)
    for secondary_track_list in secondary_track_lists:
        if len(secondary_track_list) == 0: continue
        chain_TrueorFalse = []
        particle_TrueorFalse = []
        for true_secondary_same_chain_track_list in true_secondary_same_chain_track_lists:
            chain_TrueorFalse.append(yinx(secondary_track_list, true_secondary_same_chain_track_list))
        if not any(chain_TrueorFalse): chains = listremove(secondary_track_list, chains)
        for true_secondary_same_particle_track_list in true_secondary_same_particle_track_lists:
            particle_TrueorFalse.append(yinx(secondary_track_list, true_secondary_same_particle_track_list))
        if not any(particle_TrueorFalse): particles = listremove(secondary_track_list, particles)

    if len(true_primary_tracks) != 0:
        for true_primary_track in true_primary_tracks:
            if true_primary_track in secondary_tracks: tmpMCPrimaryRecoSV = tmpMCPrimaryRecoSV + 1
    if len(true_other_tracks) != 0:
        for true_other_track in true_other_tracks:
            if true_other_track in secondary_tracks: tmpMCOthersRecoSV = tmpMCOthersRecoSV + 1
    if len(true_secondary_bb_tracks) != 0:
        for true_secondary_bb_track in true_secondary_bb_tracks:
            if true_secondary_bb_track in secondary_tracks: tmpMCBottomRecoSV = tmpMCBottomRecoSV + 1
            if true_secondary_bb_track in chains: tmpMCBottomRecoSVSameChain = tmpMCBottomRecoSVSameChain + 1
            if true_secondary_bb_track in particles: tmpMCBottomRecoSVSameParticle = tmpMCBottomRecoSVSameParticle + 1
    if len(true_secondary_cc_tracks) != 0:
        for true_secondary_cc_track in true_secondary_cc_tracks:
            if true_secondary_cc_track in secondary_tracks: tmpMCCharmRecoSV = tmpMCCharmRecoSV + 1
            if true_secondary_cc_track in chains: tmpMCCharmRecoSVSameChain = tmpMCCharmRecoSVSameChain + 1
            if true_secondary_cc_track in particles: tmpMCCharmRecoSVSameParticle = tmpMCCharmRecoSVSameParticle + 1

    print(pycolor.ACCENT + "-------------------------------------------------------------------------------------------------" + pycolor.END)
    if len(true_primary_tracks) != 0:
        tmp_score = tmpMCPrimaryRecoSV/len(true_primary_tracks)
        print(pycolor.RED + "MC Primary / Reco SV : " + str(tmp_score) + pycolor.END)
        NumPVEvent = NumPVEvent + 1
        MCPrimaryRecoSV = MCPrimaryRecoSV + tmp_score
        MCPrimaryRecoSVTrack = MCPrimaryRecoSVTrack + tmpMCPrimaryRecoSV
        NumPVTrack = NumPVTrack + len(true_primary_tracks)
    else: print(pycolor.RED + "MC Primary / Reco SV : [not exists]" + pycolor.END)

    if len(true_other_tracks) != 0:
        tmp_score = tmpMCOthersRecoSV/len(true_other_tracks)
        print(pycolor.RED + "MC Others / Reco SV : " + str(tmp_score) + pycolor.END)
        NumOthersEvent = NumOthersEvent + 1
        MCOthersRecoSV = MCOthersRecoSV + tmp_score
        MCOthersRecoSVTrack = MCOthersRecoSVTrack + tmpMCOthersRecoSV
        NumOthersTrack = NumOthersTrack + len(true_other_tracks)
    else: print(pycolor.RED + "MC Others / Reco SV : [not exists]" + pycolor.END)

    if len(true_secondary_bb_tracks) != 0:
        tmp_score, tmp_chain, tmp_particle\
        = tmpMCBottomRecoSV/len(true_secondary_bb_tracks), tmpMCBottomRecoSVSameChain/len(true_secondary_bb_tracks), tmpMCBottomRecoSVSameParticle/len(true_secondary_bb_tracks)
        print(pycolor.RED + "MC Bottom  / Reco SV : " + str(tmp_score) \
                          + " Same Chain : " + str(tmp_chain) + " Same Particle : " + str(tmp_particle) + pycolor.END)
        NumBBEvent = NumBBEvent + 1
        MCBottomRecoSV = MCBottomRecoSV + tmp_score
        MCBottomRecoSVSameChain = MCBottomRecoSVSameChain + tmp_chain
        MCBottomRecoSVSameParticle = MCBottomRecoSVSameParticle + tmp_particle
        MCBottomRecoSVTrack = MCBottomRecoSVTrack + tmpMCBottomRecoSV
        MCBottomRecoSVSameChainTrack = MCBottomRecoSVSameChainTrack + tmpMCBottomRecoSVSameChain
        MCBottomRecoSVSameParticleTrack = MCBottomRecoSVSameParticleTrack + tmpMCBottomRecoSVSameParticle
        NumBBTrack = NumBBTrack + len(true_secondary_bb_tracks)
    else: print(pycolor.RED + "MC Bottom  / Reco SV : [not exists] Same Chain : [not exists] Same Particle : [not exists]" + pycolor.END)
        
    if len(true_secondary_cc_tracks) != 0:
        tmp_score, tmp_chain, tmp_particle\
        = tmpMCCharmRecoSV/len(true_secondary_cc_tracks), tmpMCCharmRecoSVSameChain/len(true_secondary_cc_tracks), tmpMCCharmRecoSVSameParticle/len(true_secondary_cc_tracks)
        print(pycolor.RED + "MC Charm   / Reco SV : " + str(tmp_score) \
                          + " Same Chain : " + str(tmp_chain) + " Same Particle : " + str(tmp_particle) + pycolor.END)
        NumCCEvent = NumCCEvent + 1
        MCCharmRecoSV = MCCharmRecoSV + tmp_score
        MCCharmRecoSVSameChain = MCCharmRecoSVSameChain + tmp_chain
        MCCharmRecoSVSameParticle = MCCharmRecoSVSameParticle + tmp_particle
        MCCharmRecoSVTrack = MCCharmRecoSVTrack + tmpMCCharmRecoSV
        MCCharmRecoSVSameChainTrack = MCCharmRecoSVSameChainTrack + tmpMCCharmRecoSVSameChain
        MCCharmRecoSVSameParticleTrack = MCCharmRecoSVSameParticleTrack + tmpMCCharmRecoSVSameParticle
        NumCCTrack = NumCCTrack + len(true_secondary_cc_tracks)
    else: print(pycolor.RED + "MC Charm   / Reco SV : [not exists] Same Chain : [not exists] Same Particle : [not exists]" + pycolor.END)
    print(pycolor.ACCENT + "-------------------------------------------------------------------------------------------------" + pycolor.END)
		
    return MCPrimaryRecoSV, MCOthersRecoSV, MCBottomRecoSV, MCBottomRecoSVSameChain, MCBottomRecoSVSameParticle, MCCharmRecoSV, MCCharmRecoSVSameChain, MCCharmRecoSVSameParticle,\
           NumPVEvent, NumOthersEvent, NumBBEvent, NumCCEvent,\
           MCPrimaryRecoSVTrack, MCOthersRecoSVTrack, MCBottomRecoSVTrack, MCBottomRecoSVSameChainTrack, MCBottomRecoSVSameParticleTrack,\
           MCCharmRecoSVTrack, MCCharmRecoSVSameChainTrack, MCCharmRecoSVSameParticleTrack,\
           NumPVTrack, NumOthersTrack, NumBBTrack, NumCCTrack


def PrintFinish(MCPrimaryRecoSV, MCOthersRecoSV, MCBottomRecoSV, MCBottomRecoSVSameChain, MCBottomRecoSVSameParticle, MCCharmRecoSV, MCCharmRecoSVSameChain, MCCharmRecoSVSameParticle,
                NumPVEvent, NumCCEvent, NumBBEvent, NumOthersEvent):

    if(NumPVEvent>0): print(pycolor.RED + "MC Primary / Reco SV : " + str(MCPrimaryRecoSV/NumPVEvent) + pycolor.END)
    else: print(pycolor.RED + "MC Primary / Reco SV : [not exist]" + pycolor.END)

    if(NumOthersEvent>0): print(pycolor.RED + "MC Others  / Reco SV : " + str(MCOthersRecoSV/NumOthersEvent) + pycolor.END)
    else: print(pycolor.RED + "MC Others / Reco SV : [not exist]" + pycolor.END)

    if(NumBBEvent>0):
        print(pycolor.RED + "MC Bottom  / Reco SV : " + str(MCBottomRecoSV/NumBBEvent) \
                          + " Same Chain : " + str(MCBottomRecoSVSameChain/NumBBEvent) + " Same Particle : " + str(MCBottomRecoSVSameParticle/NumBBEvent) + pycolor.END)
    else: print(pycolor.RED + "MC Bottom  / Reco SV : [not exists] Same Chain : [not exists] Same Particle : [not exists]" + pycolor.END)

    if(NumCCEvent>0):
        print(pycolor.RED + "MC Charm   / Reco SV : " + str(MCCharmRecoSV/NumCCEvent) \
                          + " Same Chain : " + str(MCCharmRecoSVSameChain/NumCCEvent) + " Same Particle : " + str(MCCharmRecoSVSameParticle/NumCCEvent) + pycolor.END)
    else: print(pycolor.RED + "MC Charm   / Reco SV : [not exists] Same Chain : [not exists] Same Particle : [not exists]" + pycolor.END)


def PrintFinishTrackBase(MCPrimaryRecoSVTrack, MCOthersRecoSVTrack, MCBottomRecoSVTrack, MCBottomRecoSVSameChainTrack, MCBottomRecoSVSameParticleTrack, 
                         MCCharmRecoSVTrack, MCCharmRecoSVSameChainTrack, MCCharmRecoSVSameParticleTrack,
                         NumPVTrack, NumCCTrack, NumBBTrack, NumOthersTrack):

    if(NumPVTrack>0): print(pycolor.RED + "MC " + str(NumPVTrack) + " MC Primary / Reco SV : " + str(MCPrimaryRecoSVTrack/NumPVTrack) + pycolor.END)
    else: print(pycolor.RED + "MC Primary / Reco SV : [not exist]" + pycolor.END)

    if(NumOthersTrack>0): print(pycolor.RED + "MC " + str(NumOthersTrack) + " MC Others  / Reco SV : " + str(MCOthersRecoSVTrack/NumOthersTrack) + pycolor.END)
    else: print(pycolor.RED + "MC Others / Reco SV : [not exist]" + pycolor.END)

    if(NumBBTrack>0):
        print(pycolor.RED + "MC " + str(NumBBTrack) + " MC Bottom  / Reco SV : " + str(MCBottomRecoSVTrack/NumBBTrack) \
                          + " Same Chain : " + str(MCBottomRecoSVSameChainTrack/NumBBTrack) + " Same Particle : " + str(MCBottomRecoSVSameParticleTrack/NumBBTrack) + pycolor.END)
    else: print(pycolor.RED + "MC Bottom  / Reco SV : [not exists] Same Chain : [not exists] Same Particle : [not exists]" + pycolor.END)

    if(NumCCTrack>0):
        print(pycolor.RED + "MC " + str(NumCCTrack) + " MC Charm   / Reco SV : " + str(MCCharmRecoSVTrack/NumCCTrack) \
                          + " Same Chain : " + str(MCCharmRecoSVSameChainTrack/NumCCTrack) + " Same Particle : " + str(MCCharmRecoSVSameParticleTrack/NumCCTrack) + pycolor.END)
    else: print(pycolor.RED + "MC Charm   / Reco SV : [not exists] Same Chain : [not exists] Same Particle : [not exists]" + pycolor.END)


