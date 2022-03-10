from Networks.Tools import modeltools
from copy import deepcopy
import numpy as np

def CountTrueTrackLists(debug, true_label, chain_label):

    ccbbvtx = 0
    bcvtx = 0
    true_secondary_bb_track_lists = []
    true_secondary_cc_track_lists = []
    true_secondary_same_chain_track_lists = []

    true_primary_track_list = [i for i, x in enumerate(true_label) if x == "p"]

    for vtx in list(set(chain_label)):
        tcc = [i for i, (x, c) in enumerate(zip(true_label, chain_label)) if x == "c" and c == vtx]
        tbb = [i for i, (x, c) in enumerate(zip(true_label, chain_label)) if x == "b" and c == vtx]
        if vtx < 0:
            ccbbvtx = ccbbvtx + 1
            if len(tcc) != 0:
                true_secondary_cc_track_lists.append(tcc)
                
            if len(tbb) != 0:
                true_secondary_bb_track_lists.append(tbb)

        elif vtx > 0:
            bcvtx = ccbbvtx + 1
            true_secondary_cc_track_lists.append(tcc)
            true_secondary_bb_track_lists.append(tbb)
            true_secondary_same_chain_track_lists.append([i for i, c in enumerate(chain_label) if c == vtx])

    true_other_track_list = [i for i, x in enumerate(true_label) if x == "o"]

    return true_primary_track_list, true_secondary_bb_track_lists, true_secondary_cc_track_lists, true_secondary_same_chain_track_lists, true_other_track_list


def SecondaryVertexFinder(debug, ThresholdSecondaryScore, bigger_primary_scores, primary_track_list, secondary_event_data, 
                          encoder_tracks, decoder_tracks, slstm_model):

    track_list = np.arange(decoder_tracks.shape[0])
    secondary_track_lists = []
    all_secondary_seeds = len(secondary_event_data)
    used_secondary_seeds = 0
    used_true_secondary_seeds = 0
    for secondary_event_datum in secondary_event_data:
        track1, track2 = secondary_event_datum[1], secondary_event_datum[2]
        if (track1 in primary_track_list) or (track2 in primary_track_list): continue
        if (track1 not in list(track_list)) or (track2 not in list(track_list)): continue
        used_secondary_seeds = used_secondary_seeds + 1
        if secondary_event_datum[57]!=0 and secondary_event_datum[57]!=1 and secondary_event_datum[57]!=6: used_true_secondary_seeds = used_true_secondary_seeds + 1

        #if debug==True: print("Track List " + str(track_list))
        
        remain_decoder_tracks = decoder_tracks[track_list]
        secondary_pair = np.tile(secondary_event_datum[3:47], (1, 1)) # pair.shape = (1, 44)
        secondary_encoder_tracks = np.tile(encoder_tracks, (1, 1, 1)) # tracks.shape = (1, MaxTrack, 23)
        secondary_decoder_tracks = np.tile(remain_decoder_tracks, (1, 1, 1)) # tracks.shape = (1, RemainNTrack, 23)
        secondary_scores = slstm_model.predict([secondary_pair, secondary_encoder_tracks, secondary_decoder_tracks])
        secondary_scores = np.array(secondary_scores).reshape((-1, 1))

        tmptrack_list = np.copy(track_list)
        tmpsecondary_track_list = []
        primary_track_list = np.array(primary_track_list, dtype=int)
        for i, t in enumerate(track_list):
            if secondary_scores[i] > ThresholdSecondaryScore:
                if t not in primary_track_list:
                    tmpsecondary_track_list.append(t)
                    tmptrack_list = tmptrack_list[~(tmptrack_list == t)]
                elif (t in primary_track_list) and (secondary_scores[i] > bigger_primary_scores[t]):
                    tmpsecondary_track_list.append(t)
                    if debug==True:
                        print("Scramble Track Number " + str(t) + " SV Score " + str(secondary_score[i]) 
                              + " PV Score " + str(bigger_primary_scores[t]))
                    primary_track_list = primary_track_list[~(primary_track_list == t)]
                    tmptrack_list = tmptrack_list[~(tmptrack_list == t)]
        if len(tmpsecondary_track_list)!=0: secondary_track_lists.append(tmpsecondary_track_list)
        track_list = np.copy(tmptrack_list)

    return primary_track_list, secondary_track_lists, all_secondary_seeds, used_secondary_seeds, used_true_secondary_seeds


def yinx(y, x):
    for _y in y:
        if _y not in x: return False
    return True


def listremove(y, x):
    for _y in y:
        x.remove(_y)
    return x


def EvalResults(debug, secondary_track_lists, true_primary_tracks, true_secondary_bb_track_lists, true_secondary_cc_track_lists, true_secondary_same_chain_track_lists, true_other_tracks,
  	        MCPrimaryRecoSV, MCOthersRecoSV, MCBottomRecoSV, MCBottomRecoSVSameChain, MCBottomRecoSVSameParticle, MCCharmRecoSV, MCCharmRecoSVSameChain, MCCharmRecoSVSameParticle,
	        NumPVEvent, NumCCEvent, NumBBEvent, NumOthersEvent):

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

    if len(true_primary_tracks) != 0:
        tmp_score = tmpMCPrimaryRecoSV/len(true_primary_tracks)
        NumPVEvent = NumPVEvent + 1
        MCPrimaryRecoSV = MCPrimaryRecoSV + tmp_score

    if len(true_other_tracks) != 0:
        tmp_score = tmpMCOthersRecoSV/len(true_other_tracks)
        NumOthersEvent = NumOthersEvent + 1
        MCOthersRecoSV = MCOthersRecoSV + tmp_score

    if len(true_secondary_bb_tracks) != 0:
        tmp_score, tmp_chain, tmp_particle\
        = tmpMCBottomRecoSV/len(true_secondary_bb_tracks), tmpMCBottomRecoSVSameChain/len(true_secondary_bb_tracks), tmpMCBottomRecoSVSameParticle/len(true_secondary_bb_tracks)
        NumBBEvent = NumBBEvent + 1
        MCBottomRecoSV = MCBottomRecoSV + tmp_score
        MCBottomRecoSVSameChain = MCBottomRecoSVSameChain + tmp_chain
        MCBottomRecoSVSameParticle = MCBottomRecoSVSameParticle + tmp_particle
        
    if len(true_secondary_cc_tracks) != 0:
        tmp_score, tmp_chain, tmp_particle\
        = tmpMCCharmRecoSV/len(true_secondary_cc_tracks), tmpMCCharmRecoSVSameChain/len(true_secondary_cc_tracks), tmpMCCharmRecoSVSameParticle/len(true_secondary_cc_tracks)
        NumCCEvent = NumCCEvent + 1
        MCCharmRecoSV = MCCharmRecoSV + tmp_score
        MCCharmRecoSVSameChain = MCCharmRecoSVSameChain + tmp_chain
        MCCharmRecoSVSameParticle = MCCharmRecoSVSameParticle + tmp_particle
		
    return MCPrimaryRecoSV, MCOthersRecoSV, MCBottomRecoSV, MCBottomRecoSVSameChain, MCBottomRecoSVSameParticle, MCCharmRecoSV, MCCharmRecoSVSameChain, MCCharmRecoSVSameParticle,\
           NumPVEvent, NumCCEvent, NumBBEvent, NumOthersEvent

def EvalResultsTrackBase(debug, secondary_track_lists, true_primary_tracks, true_secondary_bb_track_lists, true_secondary_cc_track_lists, true_secondary_same_chain_track_lists, true_other_tracks,
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

    if len(true_primary_tracks) != 0:
        tmp_score = tmpMCPrimaryRecoSV/len(true_primary_tracks)
        NumPVEvent = NumPVEvent + 1
        MCPrimaryRecoSV = MCPrimaryRecoSV + tmp_score
        MCPrimaryRecoSVTrack = MCPrimaryRecoSVTrack + tmpMCPrimaryRecoSV
        NumPVTrack = NumPVTrack + len(true_primary_tracks)

    if len(true_other_tracks) != 0:
        tmp_score = tmpMCOthersRecoSV/len(true_other_tracks)
        NumOthersEvent = NumOthersEvent + 1
        MCOthersRecoSV = MCOthersRecoSV + tmp_score
        MCOthersRecoSVTrack = MCOthersRecoSVTrack + tmpMCOthersRecoSV
        NumOthersTrack = NumOthersTrack + len(true_other_tracks)

    if len(true_secondary_bb_tracks) != 0:
        tmp_score, tmp_chain, tmp_particle\
        = tmpMCBottomRecoSV/len(true_secondary_bb_tracks), tmpMCBottomRecoSVSameChain/len(true_secondary_bb_tracks), tmpMCBottomRecoSVSameParticle/len(true_secondary_bb_tracks)
        NumBBEvent = NumBBEvent + 1
        MCBottomRecoSV = MCBottomRecoSV + tmp_score
        MCBottomRecoSVSameChain = MCBottomRecoSVSameChain + tmp_chain
        MCBottomRecoSVSameParticle = MCBottomRecoSVSameParticle + tmp_particle
        MCBottomRecoSVTrack = MCBottomRecoSVTrack + tmpMCBottomRecoSV
        MCBottomRecoSVSameChainTrack = MCBottomRecoSVSameChainTrack + tmpMCBottomRecoSVSameChain
        MCBottomRecoSVSameParticleTrack = MCBottomRecoSVSameParticleTrack + tmpMCBottomRecoSVSameParticle
        NumBBTrack = NumBBTrack + len(true_secondary_bb_tracks)
        
    if len(true_secondary_cc_tracks) != 0:
        tmp_score, tmp_chain, tmp_particle\
        = tmpMCCharmRecoSV/len(true_secondary_cc_tracks), tmpMCCharmRecoSVSameChain/len(true_secondary_cc_tracks), tmpMCCharmRecoSVSameParticle/len(true_secondary_cc_tracks)
        NumCCEvent = NumCCEvent + 1
        MCCharmRecoSV = MCCharmRecoSV + tmp_score
        MCCharmRecoSVSameChain = MCCharmRecoSVSameChain + tmp_chain
        MCCharmRecoSVSameParticle = MCCharmRecoSVSameParticle + tmp_particle
        MCCharmRecoSVTrack = MCCharmRecoSVTrack + tmpMCCharmRecoSV
        MCCharmRecoSVSameChainTrack = MCCharmRecoSVSameChainTrack + tmpMCCharmRecoSVSameChain
        MCCharmRecoSVSameParticleTrack = MCCharmRecoSVSameParticleTrack + tmpMCCharmRecoSVSameParticle
        NumCCTrack = NumCCTrack + len(true_secondary_cc_tracks)
                
    return MCPrimaryRecoSV, MCOthersRecoSV, MCBottomRecoSV, MCBottomRecoSVSameChain, MCBottomRecoSVSameParticle, MCCharmRecoSV, MCCharmRecoSVSameChain, MCCharmRecoSVSameParticle,\
           NumPVEvent, NumOthersEvent, NumBBEvent, NumCCEvent,\
           MCPrimaryRecoSVTrack, MCOthersRecoSVTrack, MCBottomRecoSVTrack, MCBottomRecoSVSameChainTrack, MCBottomRecoSVSameParticleTrack,\
           MCCharmRecoSVTrack, MCCharmRecoSVSameChainTrack, MCCharmRecoSVSameParticleTrack,\
           NumPVTrack, NumOthersTrack, NumBBTrack, NumCCTrack


