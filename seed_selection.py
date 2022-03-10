from VertexFinder import tools
from Networks.Tools import modeltools
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    MaxEvent = 100
    MaxSample = 1000000
    MaxTrack = 60
    MaxPrimaryVertexLoop = 2
    ThresholdPairSecondaryScoreBBCC = 0.6
    ThresholdPairSecondaryScore = 0.8
    ThresholdPairPosScore = 5
    ThresholdPrimaryScore = 0.8
    ThresholdSecondaryScore = 0.7
    debug = False

    data_path = "data/numpy/Vertex_Finder_bb08_reshaped.npy"

    pair_model_name = "Pair_Model_Standard_201129_Standard_loss_weights_vertex0.1_position0.9_1000epochs_plus_loss_weights_vertex0.9_position0.1_1500epochs_plus_loss_weights_vertex0.95_position0.05_500epochs_lr_0.0001"

    lstm_model_name = "VLSTM_Model_Standard_PV"
    slstm_model_name = "VLSTM_Model_Standard_SV"

    print("Data Loading ...")
    data = np.load(data_path)
    variables = data[:MaxSample, 3:47]

    print("Model Loading ...")

    pair_model = modeltools.LoadPairModel(pair_model_name)
    pair_model.compile(loss={'Vertex_Output': 'categorical_crossentropy', 'Position_Output': 'mean_squared_logarithmic_error'},
                       optimizer='SGD',
                       metrics=['accuracy', 'mae'])

    pred_vertex, pred_position = tools.PairInference(pair_model, variables)
    data = np.concatenate([data[:MaxSample], pred_vertex], 1)
    data = np.concatenate([data, pred_position], 1) # -8:NC, -7:PV, -6:SVCC, -5:SVBB, -4:TVCC, -3:SVBC, -2:Others, -1:Position

    """
    primary_eff = []
    for loop in range(1, 5):
        all_seeds = 0
        primary_seeds = 0
        for datum in data[np.argsort(-data[:, -7])][:loop]:
            all_seeds = all_seeds + 1
            if datum[57] == 1: primary_seeds = primary_seeds + 1
        primary_eff.append([loop, all_seeds, primary_seeds])

    np.save("data/numpy/seed/Primary_Seeds_loop_all_pv.npy", np.array(primary_eff)) 
    """

    labels = np.argmax(data[:, -8:-1], axis=1)
    vposes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
              1, 2, 3, 4, 5, 6, 7, 8, 9, 
              10, 20, 30, 40, 50, 60, 70, 80, 90, 
              100, 200, 300, 400, 500, 600, 700, 800, 900]
    vposes = np.array(vposes)

    secondary_eff = []
    all_seeds = len([datum for datum in data if datum[57]!=0 and datum[57]!=1 and datum[57]!=6])
    for sth in tqdm(range(50, 100, 1)):
        sth = sth*0.01
        tmp = []
        for vpos in vposes:
            secondary_seeds = len([datum for datum, label in zip(data, labels) if datum[-6] + datum[-5] + datum[-4] + datum[-3] >= sth and datum[-1] <= vpos and label!=0 and label!=1 and label!=6])
            true_seeds = len([datum for datum, label in zip(data, labels) if datum[-6] + datum[-5] + datum[-4] + datum[-3] >= sth and datum[-1] <= vpos \
                              and label!=0 and label!=1 and label!=6 \
                              and datum[57]!=0 and datum[57]!=1 and datum[57]!=6])
            tmp.append([sth, vpos, all_seeds, secondary_seeds, true_seeds])
        secondary_eff.append(tmp)

    np.save("data/numpy/seed/SV_Seeds_th_vpos_all_sv_true.npy", np.array(secondary_eff))



