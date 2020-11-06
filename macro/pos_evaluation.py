import EVALTOOLS
import TOOLS
import MODELBANK
import numpy as np
from keras.utils import np_utils
from tensorflow import keras


if __name__ == "__main__":
    model_name = "Pair_Pos_Model_vfdnn04_1Msamples_2500epochs"

    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_08_shaped.npy"
    ori_variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_08.npy"

    BATCH_SIZE = 4096
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)

    #images = np.load(image_name)
    data = np.load(variable_name)
    ori_data = np.load(ori_variable_name)

    variables = data[:100000, 3:47]
    #variables = data[:, 3:47]

    ori_data[:, 48:51] = TOOLS.cartesian2polar(ori_data[:, 48:51])
    true_pos = ori_data[:100000, 48]

    model = MODELBANK.load_model(model_name)
        
    model.compile(loss='mean_squared_logarithmic_error', 
                  optimizer=OPTIMIZER, 
                  metrics=['mae'])
    
    #predict_vertex_finder = model.predict([images, variables])
    predict_pos = model.predict([variables])
    predict_pos = np.array(predict_pos, dtype=float)

    # Evaluation of Vertex Finder
    classes = ["not connected", "primary", "secondary cc", "secondary bb", "secondary bc"]
    l = 1
    labels = ["R_position"]
    predict_pos = np.reshape(predict_pos, [-1, l])
    true_pos = np.reshape(true_pos, [-1, l])
    EVALTOOLS.ana_regression_heat(predict_pos, true_pos, model_name, labels, l, log=True)

