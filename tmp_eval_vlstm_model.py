from Networks.Tools import datatools, modeltools, evaltools
from Networks.VLSTMModel import models, training
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc


if __name__ == "__main__":
    print("VLSTM Model Evaluation...")
    
    model_name = "VLSTM_Model_Standard_SV"
    history_path = "History_VLSTM_Model_Standard_SV_20201202"
    cc06pv_pairs, cc06pv_tracks, cc06pv_targets, cc06sv_pairs, cc06sv_tracks, cc06sv_targets = np.load("/work/goto/data/numpy/cc/cc_06_pair_pv_pairs.npy"), np.load("/work/goto/data/numpy/cc/cc_06_pair_pv_tracks.npy"), np.load("/work/goto/data/numpy/cc/cc_06_pair_pv_targets.npy"), np.load("/work/goto/data/numpy/cc/cc_06_pair_sv_pairs.npy"), np.load("/work/goto/data/numpy/cc/cc_06_pair_sv_tracks.npy"), np.load("/work/goto/data/numpy/cc/cc_06_pair_sv_targets.npy")
    bb07pv_pairs, bb07pv_tracks, bb07pv_targets, bb07sv_pairs, bb07sv_tracks, bb07sv_targets = np.load("/work/goto/data/numpy/bb/bb_07_pair_pv_pairs.npy"), np.load("/work/goto/data/numpy/bb/bb_07_pair_pv_tracks.npy"), np.load("/work/goto/data/numpy/bb/bb_07_pair_pv_targets.npy"), np.load("/work/goto/data/numpy/bb/bb_07_pair_sv_pairs.npy"), np.load("/work/goto/data/numpy/bb/bb_07_pair_sv_tracks.npy"), np.load("/work/goto/data/numpy/bb/bb_07_pair_sv_targets.npy")

    history = evaltools.LoadVLSTMHistory(history_path)

    MaxCC = cc06sv_pairs.shape[0]
    MaxBB = bb07sv_pairs.shape[0]

    """
    pair = np.concatenate([cc06pv_pairs[:MaxCC], cc06sv_pairs, bb07pv_pairs[:MaxBB], bb07sv_pairs], 0)
    targets = np.concatenate([cc06pv_targets[:MaxCC], cc06sv_targets, bb07pv_targets[:MaxBB], bb07sv_targets], 0)
    del cc06pv_pairs, cc06sv_pairs, bb07pv_pairs, bb07sv_pairs
    del cc06pv_targets, cc06sv_targets, bb07pv_targets, bb07sv_targets
    gc.collect()
    tracks = np.concatenate([cc06pv_tracks[:MaxCC], cc06sv_tracks, bb07pv_tracks[:MaxBB], bb07sv_tracks], 0)
    del cc06pv_tracks, cc06sv_tracks, bb07pv_tracks, bb07sv_tracks
    gc.collect()
    """
    pair = np.concatenate([cc06sv_pairs, bb07sv_pairs], 0)
    targets = np.concatenate([cc06sv_targets, bb07sv_targets], 0)
    del cc06pv_pairs, cc06sv_pairs, bb07pv_pairs, bb07sv_pairs
    del cc06pv_targets, cc06sv_targets, bb07pv_targets, bb07sv_targets
    gc.collect()
    tracks = np.concatenate([cc06sv_tracks, bb07sv_tracks], 0)
    del cc06pv_tracks, cc06sv_tracks, bb07pv_tracks, bb07sv_tracks
    gc.collect()

    #loss, accuracy_all, accuracy, true_positive, true_negative, false_positive, false_negative
    
    loss = history['loss']
    val_loss = history['val_loss']
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    true_positive = history['true_positive']
    val_true_positive = history['val_true_positive']
    true_negative = history['true_negative']
    val_true_negative = history['val_true_negative']
    
    plt.rcParams["font.size"] = 15
    plt.plot(loss, color="black", label="Loss - Train")
    plt.plot(val_loss, color="black", label="Loss - Validation", linestyle="dashed")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("data/figure/loss/Loss_" + model_name + ".png")
    plt.clf()
    
    plt.rcParams["font.size"] = 15
    plt.plot(accuracy, color="black", label="Accuracy - Train")
    plt.plot(val_accuracy, color="black", label="Accuracy - Validation", linestyle="dashed")
    plt.plot(true_positive, color="red", label="True Positive - Train")
    plt.plot(val_true_positive, color="red", label="True_Positive - Validation", linestyle="dashed")
    plt.plot(true_negative, color="blue", label="True Negative - Train")
    plt.plot(val_true_negative, color="blue", label="True Negative - Validation", linestyle="dashed")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("data/figure/loss/Accuracy_" + model_name + ".png")
    plt.clf()
    
    
    model = modeltools.LoadAttentionVLSTMModel(model_name)

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    predict_labels = model.predict([pair, tracks, tracks], verbose=1)
    predict_labels = np.array(predict_labels, dtype=float).reshape((-1, 1))
    targets = np.array(targets, dtype=int).reshape((-1, 5))

    zptargets = []
    zplabels = []
    for label, target in zip(predict_labels, targets): # events
        if target[1]==0: continue
        zptargets.append(target[0])
        zplabels.append(label)
    zptargets = np.array(zptargets, dtype=float).reshape((-1, 1))
    zplabels = np.array(zplabels, dtype=float).reshape((-1, 1))

    save_path = "data/numpy/roc/Efficiency_Curve_" + model_name + "_thre_allsig_allbg_sig_bg.npy"
    true_pred = np.concatenate([zptargets, zplabels], 1)

    cuts = np.arange(0.00, 1.01, 0.01)
    efficiency_curve = []
    all_signal = len([datum for datum in true_pred if datum[0]==1])
    all_background = len([datum for datum in true_pred if datum[0]==0])
    for cut in tqdm(cuts):
        signal = len([datum for datum in true_pred if datum[1] > cut and datum[0]==1])
        background = len([datum for datum in true_pred if datum[1] > cut and datum[0]==0])
        efficiency_curve.append([cut, all_signal, all_background, signal, background])
    efficiency_curve = np.array(efficiency_curve, dtype=float)
    np.save(save_path, efficiency_curve, allow_pickle=True)

    model_name2 = "VLSTM_Model_Standard_CC_BB_PV_SV"

    model = modeltools.LoadAttentionVLSTMModel(model_name2)

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    predict_labels = model.predict([pair, tracks, tracks], verbose=1)
    predict_labels = np.array(predict_labels, dtype=float).reshape((-1, 1))
    targets = np.array(targets, dtype=int).reshape((-1, 5))

    zptargets = []
    zplabels = []
    for label, target in zip(predict_labels, targets): # events
        if target[1]==0: continue
        zptargets.append(target[0])
        zplabels.append(label)
    zptargets = np.array(zptargets, dtype=float).reshape((-1, 1))
    zplabels = np.array(zplabels, dtype=float).reshape((-1, 1))

    save_path = "data/numpy/roc/Efficiency_Curve_" + model_name + "_" + model_name2 + "_thre_allsig_allbg_sig_bg.npy"
    true_pred = np.concatenate([zptargets, zplabels], 1)

    cuts = np.arange(0.00, 1.01, 0.01)
    efficiency_curve = []
    all_signal = len([datum for datum in true_pred if datum[0]==1])
    all_background = len([datum for datum in true_pred if datum[0]==0])
    for cut in tqdm(cuts):
        signal = len([datum for datum in true_pred if datum[1] > cut and datum[0]==1])
        background = len([datum for datum in true_pred if datum[1] > cut and datum[0]==0])
        efficiency_curve.append([cut, all_signal, all_background, signal, background])
    efficiency_curve = np.array(efficiency_curve, dtype=float)
    np.save(save_path, efficiency_curve, allow_pickle=True)


