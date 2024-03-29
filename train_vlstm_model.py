from Networks.Tools import datatools, modeltools
from Networks.VLSTMModel import models, training
import numpy as np
import gc


if __name__ == "__main__":
    print("VLSTM Model Training...")
    
    #model_name = "Attention_VLSTM_Model_Re_3"
    model_name = "VLSTM_Simple_Model_Re_2"

    cc05pv_pairs, cc05pv_tracks, cc05pv_targets, cc05sv_pairs, cc05sv_tracks, cc05sv_targets = np.load("data/numpy/cc/cc_05_pair_pv_pairs_small.npy"), np.load("data/numpy/cc/cc_05_pair_pv_tracks_small.npy"), np.load("data/numpy/cc/cc_05_pair_pv_targets_small.npy"), np.load("data/numpy/cc/cc_05_pair_sv_pairs.npy"), np.load("data/numpy/cc/cc_05_pair_sv_tracks.npy"), np.load("data/numpy/cc/cc_05_pair_sv_targets.npy")
    bb06pv_pairs, bb06pv_tracks, bb06pv_targets, bb06sv_pairs, bb06sv_tracks, bb06sv_targets = np.load("data/numpy/bb/bb_06_pair_pv_pairs_small.npy"), np.load("data/numpy/bb/bb_06_pair_pv_tracks_small.npy"), np.load("data/numpy/bb/bb_06_pair_pv_targets_small.npy"), np.load("data/numpy/bb/bb_06_pair_sv_pairs.npy"), np.load("data/numpy/bb/bb_06_pair_sv_tracks.npy"), np.load("data/numpy/bb/bb_06_pair_sv_targets.npy")

    MaxCC = cc05sv_pairs.shape[0]
    MaxBB = bb06sv_pairs.shape[0]

    pair = np.concatenate([cc05pv_pairs[:MaxCC], cc05sv_pairs, bb06pv_pairs[:MaxBB], bb06sv_pairs], 0)
    targets = np.concatenate([cc05pv_targets[:MaxCC], cc05sv_targets, bb06pv_targets[:MaxBB], bb06sv_targets], 0)
    del cc05pv_pairs, cc05sv_pairs, bb06pv_pairs, bb06sv_pairs
    del cc05pv_targets, cc05sv_targets, bb06pv_targets, bb06sv_targets
    gc.collect()
    tracks = np.concatenate([cc05pv_tracks[:MaxCC], cc05sv_tracks, bb06pv_tracks[:MaxBB], bb06sv_tracks], 0)
    del cc05pv_tracks, cc05sv_tracks, bb06pv_tracks, bb06sv_tracks
    gc.collect()

    encoder_units = 256
    decoder_units= 256
    batch_size = 32
    nb_epochs = 100
    nb_samples = 50000
    validation_split = 0.2
    lr = 0.001

    #reinforce loss of tracks without pair
    pair_reinforce = False

    #model = models.AttentionVLSTMModel(pair, tracks, ENCODER_UNITS=encoder_units, DECODER_UNITS=decoder_units)
    model = models.VLSTMModelSimple(pair, tracks, UNITS=256)

    #model, history = training.AttentionVLSTMModelTraining(model, model_name, pair, tracks, targets, 
    #                                                      BATCH_SIZE=batch_size, NB_EPOCHS=nb_epochs, NB_SAMPLES=nb_samples, VALIDATION_SPLIT=validation_split, LR=lr,
    #                                                      pair_reinforce=pair_reinforce)
    
    model, history = training.VLSTMModelSimpleTraining(model, model_name, pair, tracks, targets, 
                                                       BATCH_SIZE=batch_size, NB_EPOCHS=nb_epochs, NB_SAMPLES=nb_samples, VALIDATION_SPLIT=validation_split, LR=lr,
                                                       pair_reinforce=pair_reinforce)
    
    modeltools.SaveVLSTMHistory(history, model_name)
    modeltools.SaveAttentionVLSTMModel(model, model_name)

