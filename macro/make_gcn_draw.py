import EVALTOOLS
import MODELBANK
import numpy as np
from keras.utils import np_utils
from tensorflow import keras


if __name__ == "__main__":
    model_name = "Comparison_Conv_OnlyLow_CustomLoss_Model_1500epochs"

    #image_name = "/home/goto/ILC/Deep_Learning/data/test/all_test_complete_track_image.npy"
    #variable_name = "/home/goto/ILC/Deep_Learning/data/test/all_test_complete_shaped.npy"

    # can't use the chi100 dataset
    
    #image_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03_track_image_v2_05.npy"
    variable_name = "/home/goto/ILC/Deep_Learning/data/gcn_vfdnn_03_shaped.npy"

    BATCH_SIZE = 4096
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)

    #images = np.load(image_name)
    data = np.load(variable_name)

    variables = data[1200000:1500000, 3:47]
    variables_full = data[1200000:1500000]
    #variables = data[:, 3:47]
    true_vertex_finder = np_utils.to_categorical(data[1200000:1500000, 57], 5)

    model = MODELBANK.load_model(model_name)
        
    model.compile(loss='categorical_crossentropy', 
                  optimizer=OPTIMIZER, 
                  metrics=['accuracy'])
    
    #predict_vertex_finder = model.predict([images, variables])
    predict_vertex_finder = model.predict([variables])
    predict_vertex_finder = np.array(predict_vertex_finder, dtype=float)


    save_data = np.concatenate([variables_full, true_vertex_finder], 1)
    save_data = np.concatenate([save_data, predict_vertex_finder], 1)
    print(variables_full.shape)
    print(save_data.shape)
    np.save("Full_plus_Predict_vfdnn_03_1200000_1500000.npy", save_data)

