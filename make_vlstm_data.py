from Networks.Tools import datatools
import numpy as np


if __name__ == "__main__":
    print("VLSTM Data Making")
    
    cc05_data_path = "data/numpy/cc/cc_05_pair.npy"
    cc06_data_path = "data/numpy/cc/cc_08_pair.npy"
    bb06_data_path = "data/numpy/bb/bb_06_pair.npy"
    bb07_data_path = "data/numpy/bb/bb_07_pair.npy"
    
    pv_pairs_save_data_path = "data/numpy/Test_VLSTM_training_pv_pairs.npy"
    pv_tracks_save_data_path = "data/numpy/Test_VLSTM_training_pv_tracks.npy"
    pv_targets_save_data_path = "data/numpy/Test_VLSTM_training_pv_targets.npy"
    sv_pairs_save_data_path = "data/numpy/Test_VLSTM_training_sv_pairs.npy"
    sv_tracks_save_data_path = "data/numpy/Test_VLSTM_training_sv_tracks.npy"
    sv_targets_save_data_path = "data/numpy/Test_VLSTM_training_sv_targets.npy"

    #cc05pv_pairs, cc05pv_tracks, cc05pv_targets, cc05sv_pairs, cc05sv_tracks, cc05sv_targets = datatools.LoadVLSTMData(cc05_data_path, MaxTrack=60)
    #cc06pv_pairs, cc06pv_tracks, cc06pv_targets, cc06sv_pairs, cc06sv_tracks, cc06sv_targets = datatools.LoadVLSTMData(cc06_data_path, MaxTrack=60)
    #bb06pv_pairs, bb06pv_tracks, bb06pv_targets, bb06sv_pairs, bb06sv_tracks, bb06sv_targets = datatools.LoadVLSTMData(bb06_data_path, MaxTrack=60)
    bb07pv_pairs, bb07pv_tracks, bb07pv_targets, bb07sv_pairs, bb07sv_tracks, bb07sv_targets = datatools.LoadVLSTMData(bb07_data_path, MaxTrack=60)

    """
    pv_pairs = np.concatenate([cc05pv_pairs, cc06pv_pairs, bb06pv_pairs, bb07pv_pairs], 0)
    pv_tracks = np.concatenate([cc05pv_tracks, cc06pv_tracks, bb06pv_tracks, bb07pv_tracks], 0)
    pv_targets = np.concatenate([cc05pv_targets, cc06pv_targets, bb06pv_targets, bb07pv_targets], 0)

    sv_pairs = np.concatenate([cc05sv_pairs, cc06sv_pairs, bb06sv_pairs, bb07sv_pairs], 0)
    sv_tracks = np.concatenate([cc05sv_tracks, cc06sv_tracks, bb06sv_tracks, bb07sv_tracks], 0)
    sv_targets = np.concatenate([cc05sv_targets, cc06sv_targets, bb06sv_targets, bb07sv_targets], 0)

    np.save(pv_pairs_save_data_path, pv_pairs)
    np.save(pv_tracks_save_data_path, pv_tracks)
    np.save(pv_targets_save_data_path, pv_targets)
    np.save(sv_pairs_save_data_path, sv_pairs)
    np.save(sv_tracks_save_data_path, sv_tracks)
    np.save(sv_targets_save_data_path, sv_targets)
    """
