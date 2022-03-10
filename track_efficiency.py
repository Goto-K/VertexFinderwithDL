from VertexFinder import tools, tunetools
import numpy as np
from tqdm import tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


if __name__ == "__main__":

    MaxEvent = 100
    MaxSample = 1000000
    MaxTrack = 60
    ThresholdPairSecondaryScoreBBCC = 0.6
    ThresholdPairSecondaryScore = 0.88
    ThresholdPairPosScore = 30
    debug = False

    data_path = "data/numpy/Vertex_Finder_bb08_reshaped.npy"

    pair_model_name = "Pair_Model_Standard_201129_Standard_loss_weights_vertex0.1_position0.9_1000epochs_plus_loss_weights_vertex0.9_position0.1_1500epochs_plus_loss_weights_vertex0.95_position0.05_500epochs_lr_0.0001"

    lstm_model_name = "VLSTM_Model_Standard_PV"
    slstm_model_name = "VLSTM_Model_Standard_SV"

    print("Data Loading ...")
    data = np.load(data_path)
    variables = data[:MaxSample, 3:47]

    print("Model Loading ...")
    pair_model, lstm_model, slstm_model = tools.ModelsLoad(pair_model_name, lstm_model_name, slstm_model_name)

    pred_vertex, pred_position = tools.PairInference(pair_model, variables)
    data = np.concatenate([data[:MaxSample], pred_vertex], 1)
    data = np.concatenate([data, pred_position], 1) # -8:NC, -7:PV, -6:SVCC, -5:SVBB, -4:TVCC, -3:SVBC, -2:Others, -1:Position

    MCPIDRecoSV = []
    MCPIDRecoSVTrack = []
    for MaxPrimaryVertexLoop in tqdm(range(1, 4)):
        for ThresholdPrimaryScore in tqdm(np.arange(0.5, 1.0, 0.05)):
            for ThresholdSecondaryScore in tqdm(np.arange(0.5, 1.0, 0.05)):
                NumPVEvent = 0
                NumCCEvent = 0
                NumBBEvent = 0
                NumOthersEvent = 0

                MCPrimaryRecoSV = 0
                MCOthersRecoSV = 0

                MCBottomRecoSV = 0
                MCBottomRecoSVSameChain = 0
                MCBottomRecoSVSameParticle = 0
                MCCharmRecoSV = 0
                MCCharmRecoSVSameChain = 0
                MCCharmRecoSVSameParticle = 0

                NumPVTrack = 0
                NumCCTrack = 0
                NumBBTrack = 0
                NumOthersTrack = 0

                MCPrimaryRecoSVTrack = 0
                MCOthersRecoSVTrack = 0

                MCBottomRecoSVTrack = 0
                MCBottomRecoSVSameChainTrack = 0
                MCBottomRecoSVSameParticleTrack = 0
                MCCharmRecoSVTrack = 0
                MCCharmRecoSVSameChainTrack = 0
                MCCharmRecoSVSameParticleTrack = 0

                all_true_secondary_seeds, all_secondary_seeds, used_secondary_seeds, used_true_secondary_seeds = 0, 0, 0, 0

                for ievent in range(MaxEvent):
                    event_data = [datum for datum in data if datum[0]==ievent]
                    event_data = np.array(event_data)
                    all_true_secondary_seeds = all_true_secondary_seeds + len([d for d in event_data if d[57]!=0 and d[57]!=1 and d[57]!=6])
                    NTrack = (1 + np.sqrt(1 + 8*event_data.shape[0]))/2 
        
                    encoder_tracks, decoder_tracks, true_label, chain_label = tools.GetEncoderDecoderTracksandTrue(debug, event_data, NTrack, MaxTrack)
                    secondary_event_data = tools.SecondarySeedSelectionOne(debug, event_data, ThresholdPairSecondaryScore, ThresholdPairPosScore)
                    primary_track_list, bigger_primary_scores = tools.PrimaryVertexFinder(debug, MaxPrimaryVertexLoop, ThresholdPrimaryScore, event_data,
		                                                              encoder_tracks, decoder_tracks, lstm_model)
                    primary_track_list, secondary_track_lists, tmpall_secondary_seeds, tmpused_secondary_seeds, tmpused_true_secondary_seeds\
                     = tunetools.SecondaryVertexFinder(debug, ThresholdSecondaryScore, bigger_primary_scores, primary_track_list, secondary_event_data, 
                                                                                encoder_tracks, decoder_tracks, slstm_model)

                    all_secondary_seeds = all_secondary_seeds + tmpall_secondary_seeds 
                    used_secondary_seeds = used_secondary_seeds + tmpused_secondary_seeds
                    used_true_secondary_seeds = used_true_secondary_seeds + tmpused_true_secondary_seeds
                    
                    true_primary_tracks, true_secondary_bb_track_lists, true_secondary_cc_track_lists, true_secondary_same_chain_track_lists, true_other_tracks\
                     = tunetools.CountTrueTrackLists(debug, true_label, chain_label)

                    MCPrimaryRecoSV, MCOthersRecoSV, MCBottomRecoSV, MCBottomRecoSVSameChain, MCBottomRecoSVSameParticle, MCCharmRecoSV, MCCharmRecoSVSameChain, MCCharmRecoSVSameParticle,\
                    NumPVEvent, NumOthersEvent, NumBBEvent, NumCCEvent,\
                    MCPrimaryRecoSVTrack, MCOthersRecoSVTrack, MCBottomRecoSVTrack, MCBottomRecoSVSameChainTrack, MCBottomRecoSVSameParticleTrack,\
                    MCCharmRecoSVTrack, MCCharmRecoSVSameChainTrack, MCCharmRecoSVSameParticleTrack,\
                    NumPVTrack, NumOthersTrack, NumBBTrack, NumCCTrack\
                     = tunetools.EvalResultsTrackBase(debug, secondary_track_lists, true_primary_tracks, 
                         true_secondary_bb_track_lists, true_secondary_cc_track_lists, true_secondary_same_chain_track_lists, true_other_tracks,
                         MCPrimaryRecoSV, MCOthersRecoSV, MCBottomRecoSV, MCBottomRecoSVSameChain, MCBottomRecoSVSameParticle, MCCharmRecoSV, MCCharmRecoSVSameChain, MCCharmRecoSVSameParticle,
                         NumPVEvent, NumOthersEvent, NumBBEvent, NumCCEvent,
                         MCPrimaryRecoSVTrack, MCOthersRecoSVTrack, MCBottomRecoSVTrack, MCBottomRecoSVSameChainTrack, MCBottomRecoSVSameParticleTrack,
                         MCCharmRecoSVTrack, MCCharmRecoSVSameChainTrack, MCCharmRecoSVSameParticleTrack,
                         NumPVTrack, NumOthersTrack, NumBBTrack, NumCCTrack)

                MCPIDRecoSV.append([ThresholdPrimaryScore, ThresholdSecondaryScore, MaxPrimaryVertexLoop, 
                                    MCPrimaryRecoSV, NumPVEvent, 
                                    MCOthersRecoSV, NumOthersEvent, 
                                    MCBottomRecoSV, MCBottomRecoSVSameChain, MCBottomRecoSVSameParticle, NumBBEvent, 
                                    MCCharmRecoSV, MCCharmRecoSVSameChain, MCCharmRecoSVSameParticle, NumCCEvent, 
                                    all_true_secondary_seeds, all_secondary_seeds, used_secondary_seeds, used_true_secondary_seeds])
    
                MCPIDRecoSVTrack.append([ThresholdPrimaryScore, ThresholdSecondaryScore, MaxPrimaryVertexLoop, 
                                    MCPrimaryRecoSVTrack, NumPVTrack, 
                                    MCOthersRecoSVTrack, NumOthersTrack, 
                                    MCBottomRecoSVTrack, MCBottomRecoSVSameChainTrack, MCBottomRecoSVSameParticleTrack, NumBBTrack, 
                                    MCCharmRecoSVTrack, MCCharmRecoSVSameChainTrack, MCCharmRecoSVSameParticleTrack, NumCCTrack, 
                                    all_true_secondary_seeds, all_secondary_seeds, used_secondary_seeds, used_true_secondary_seeds])
    
    np.save("data/numpy/seed/Track_Efficiency_pth_sth_loop_mcpv_npv_mcoth_noth_mcbb_nbb_mccc_ncc_altss_alss_uss_utss.npy", MCPIDRecoSV)
    np.save("data/numpy/seed/Track_Efficiency_TrackBase_pth_sth_loop_mcpv_npv_mcoth_noth_mcbb_nbb_mccc_ncc_altss_alss_uss_utss.npy", MCPIDRecoSVTrack)
