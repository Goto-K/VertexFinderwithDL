import EVALTOOLS
import MODELBANK
import numpy as np
from keras.utils import np_utils
from tensorflow import keras


if __name__ == "__main__":
    model_name = "Pair_Model_vfdnn04_1Msamples_1500epochs_2"

    #image_name = "/home/goto/ILC/Deep_Learning/data/test/all_test_complete_track_image.npy"
    #variable_name = "/home/goto/ILC/Deep_Learning/data/test/all_test_complete_shaped.npy"
    #image_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03_chi100_track_image_v2_05.npy"
    #variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03_chi100_shaped.npy"
    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_08_shaped.npy"

    BATCH_SIZE = 4096
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)

    #images = np.load(image_name)
    data = np.load(variable_name)

    variables = data[:100000, 3:47]
    #variables = data[:, 3:47]
    true_vertex_finder = np_utils.to_categorical(data[:100000, 57], 5)

    model = MODELBANK.load_model(model_name)
        
    model.compile(loss='categorical_crossentropy', 
                  optimizer=OPTIMIZER, 
                  metrics=['accuracy'])
    
    #predict_vertex_finder = model.predict([images, variables])
    predict_vertex_finder = model.predict([variables])
    predict_vertex_finder = np.array(predict_vertex_finder, dtype=float)

true_nc_pred_nc = [pred[0] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[0]==1]
true_nc_pred_pv = [pred[1] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[0]==1]
true_nc_pred_cc = [pred[2] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[0]==1]
true_nc_pred_bb = [pred[3] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[0]==1]
true_nc_pred_bc = [pred[4] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[0]==1]
true_pv_pred_nc = [pred[0] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[1]==1]
true_pv_pred_pv = [pred[1] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[1]==1]
true_pv_pred_cc = [pred[2] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[1]==1]
true_pv_pred_bb = [pred[3] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[1]==1]
true_pv_pred_bc = [pred[4] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[1]==1]
true_cc_pred_nc = [pred[0] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[2]==1]
true_cc_pred_pv = [pred[1] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[2]==1]
true_cc_pred_cc = [pred[2] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[2]==1]
true_cc_pred_bb = [pred[3] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[2]==1]
true_cc_pred_bc = [pred[4] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[2]==1]
true_bb_pred_nc = [pred[0] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[3]==1]
true_bb_pred_pv = [pred[1] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[3]==1]
true_bb_pred_cc = [pred[2] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[3]==1]
true_bb_pred_bb = [pred[3] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[3]==1]
true_bb_pred_bc = [pred[4] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[3]==1]
true_bc_pred_nc = [pred[0] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[4]==1]
true_bc_pred_pv = [pred[1] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[4]==1]
true_bc_pred_cc = [pred[2] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[4]==1]
true_bc_pred_bb = [pred[3] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[4]==1]
true_bc_pred_bc = [pred[4] for true, pred in zip(true_vertex_finder, predict_vertex_finder) if true[4]==1]
import matplotlib.pyplot as plt
plt.hist(true_nc_pred_pv, bins=100, range=(0, 1), histtype="step", label="true not connected")
plt.hist(true_pv_pred_pv, bins=100, range=(0, 1), histtype="step", label="true primary vertex")
plt.hist(true_cc_pred_pv, bins=100, range=(0, 1), histtype="step", label="true secondary vertex cc")
plt.hist(true_bb_pred_pv, bins=100, range=(0, 1), histtype="step", label="true secondary vertex bb")
plt.hist(true_bc_pred_pv, bins=100, range=(0, 1), histtype="step", label="true secondary vertex bc")
plt.legend()
plt.yscale('log')
plt.show()
%history -f history_drow_pair_pd.py
