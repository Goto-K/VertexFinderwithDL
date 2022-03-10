from VertexFinder import tools
import numpy as np


if __name__ == "__main__":

    #MaxEvent = 100
    MaxSample = -1
    MaxTrack = 60
    MaxPrimaryVertexLoop = 3
    ThresholdPairSecondaryScoreBBCC = 0.6
    ThresholdPairSecondaryScore = 0.88
    ThresholdPairPosScore = 30
    ThresholdPrimaryScore = 0.50
    ThresholdSecondaryScore = 0.75
    debug = True

    NumPVEvent = 0
    NumOthersEvent = 0
    NumBBEvent = 0
    NumCCEvent = 0

    MCPrimaryRecoSV = 0
    MCOthersRecoSV = 0

    MCBottomRecoSV = 0
    MCBottomRecoSVSameChain = 0
    MCBottomRecoSVSameParticle = 0
    MCCharmRecoSV = 0
    MCCharmRecoSVSameChain = 0
    MCCharmRecoSVSameParticle = 0

    MCPrimaryRecoSVTrack = 0
    MCOthersRecoSVTrack = 0
    MCBottomRecoSVTrack = 0
    MCBottomRecoSVSameChainTrack = 0
    MCBottomRecoSVSameParticleTrack = 0
    MCCharmRecoSVTrack = 0
    MCCharmRecoSVSameChainTrack = 0
    MCCharmRecoSVSameParticleTrack = 0
    
    NumPVTrack = 0
    NumOthersTrack = 0 
    NumBBTrack = 0
    NumCCTrack = 0

    data_path = "data/numpy/Vertex_Finder_bb_lcfiplus_reshaped.npy"

    pair_model_name = "Pair_Model_Standard"

    lstm_model_name = "VLSTM_Model_Standard_PV"
    slstm_model_name = "VLSTM_Model_Standard_SV"

    print("Data Loading ...")
    data = np.load(data_path)
    variables = data[:MaxSample, 3:47]
    
    MaxEvent = int(data[-1, 0])

    print("Model Loading ...")
    pair_model, lstm_model, slstm_model = tools.ModelsLoad(pair_model_name, lstm_model_name, slstm_model_name)

    pred_vertex, pred_position = tools.PairInference(pair_model, variables)
    data = np.concatenate([data[:MaxSample], pred_vertex], 1)
    data = np.concatenate([data, pred_position], 1) # -8:NC, -7:PV, -6:SVCC, -5:SVBB, -4:TVCC, -3:SVBC, -2:Others, -1:Position

    vertices_list = []
    for ievent in range(MaxEvent):
        print("=================================================================================================")
        print("EVENT NUMBER " + str(ievent))
        print("=================================================================================================")
        event_data = [datum for datum in data if datum[0]==ievent]
        event_data = np.array(event_data)
        NTrack = (1 + np.sqrt(1 + 8*event_data.shape[0]))/2 
        
        if debug==True: print("The Number of Tracks in this event is " + str(NTrack))
        if NTrack<3: continue

        # ================================================================================================= #
        # Making Tracks / True Labels ===================================================================== #
        # ================================================================================================= #
        encoder_tracks, decoder_tracks, true_label, chain_label = tools.GetEncoderDecoderTracksandTrue(debug, event_data, NTrack, MaxTrack)

        # ================================================================================================= #
        # Secondary Seed Selection========================================================================= #
        # ================================================================================================= #
        print("Secondary Seed Selection ...")
        secondary_event_data = tools.SecondarySeedSelectionOne(debug, event_data, ThresholdPairSecondaryScore, ThresholdPairPosScore)
        #secondary_seed_data = SecondarySeedSelectionTwo(debug, secondary_event_data, )

        # ================================================================================================= #
        # Primary Vertex Finder =========================================================================== #
        # ================================================================================================= #
        print("Primary Vertex Prediction ...")
        primary_track_list, bigger_primary_scores = tools.PrimaryVertexFinder(debug, MaxPrimaryVertexLoop, ThresholdPrimaryScore, event_data,
		                                                              encoder_tracks, decoder_tracks, lstm_model)

        # ================================================================================================= #
        # Secondary Vertex Finder ========================================================================= #
        # ================================================================================================= #
        print("Secondary Vertex Prediction ...")
        primary_track_list, secondary_track_lists = tools.SecondaryVertexFinder(debug, ThresholdSecondaryScore, bigger_primary_scores, primary_track_list, secondary_event_data, 
                                                                                encoder_tracks, decoder_tracks, slstm_model);

        # ================================================================================================= #
        # Result ========================================================================================== #
        # ================================================================================================= #
        print("Finish !!")
        true_primary_tracks, true_secondary_bb_track_lists, true_secondary_cc_track_lists, true_secondary_same_chain_track_lists, true_other_tracks\
                = tools.CountPrintTrueTrackLists(debug, true_label, chain_label)
        
        tools.PrintPredTrackLists(primary_track_list, secondary_track_lists)

        MCPrimaryRecoSV, MCOthersRecoSV, MCBottomRecoSV, MCBottomRecoSVSameChain, MCBottomRecoSVSameParticle, MCCharmRecoSV, MCCharmRecoSVSameChain, MCCharmRecoSVSameParticle,\
        NumPVEvent, NumOthersEvent, NumBBEvent, NumCCEvent,\
        MCPrimaryRecoSVTrack, MCOthersRecoSVTrack, MCBottomRecoSVTrack, MCBottomRecoSVSameChainTrack, MCBottomRecoSVSameParticleTrack,\
        MCCharmRecoSVTrack, MCCharmRecoSVSameChainTrack, MCCharmRecoSVSameParticleTrack,\
        NumPVTrack, NumOthersTrack, NumBBTrack, NumCCTrack\
	= tools.EvalPrintResults(debug, 
                                 secondary_track_lists, true_primary_tracks, true_secondary_bb_track_lists, true_secondary_cc_track_lists, true_secondary_same_chain_track_lists, true_other_tracks, 
			         MCPrimaryRecoSV, MCOthersRecoSV, MCBottomRecoSV, MCBottomRecoSVSameChain, MCBottomRecoSVSameParticle, MCCharmRecoSV, MCCharmRecoSVSameChain, MCCharmRecoSVSameParticle,
		                 NumPVEvent, NumOthersEvent, NumBBEvent, NumCCEvent,
                                 MCPrimaryRecoSVTrack, MCOthersRecoSVTrack, MCBottomRecoSVTrack, MCBottomRecoSVSameChainTrack, MCBottomRecoSVSameParticleTrack,
                                 MCCharmRecoSVTrack, MCCharmRecoSVSameChainTrack, MCCharmRecoSVSameParticleTrack,
                                 NumPVTrack, NumOthersTrack, NumBBTrack, NumCCTrack)


    tools.PrintFinish(MCPrimaryRecoSV, MCOthersRecoSV, MCBottomRecoSV, MCBottomRecoSVSameChain, MCBottomRecoSVSameParticle, MCCharmRecoSV, MCCharmRecoSVSameChain, MCCharmRecoSVSameParticle,
                      NumPVEvent, NumCCEvent, NumBBEvent, NumOthersEvent)
    tools.PrintFinishTrackBase(MCPrimaryRecoSVTrack, MCOthersRecoSVTrack, MCBottomRecoSVTrack, MCBottomRecoSVSameChainTrack, MCBottomRecoSVSameParticleTrack, 
                               MCCharmRecoSVTrack, MCCharmRecoSVSameChainTrack, MCCharmRecoSVSameParticleTrack,
                               NumPVTrack, NumCCTrack, NumBBTrack, NumOthersTrack)
