import EVALTOOLS
import MODELBANK
import numpy as np
from keras.utils import np_utils
from tensorflow import keras


if __name__ == "__main__":
    model_name = "Pair_Model_vfdnn04_1Msamples_chi100_2500epochs"

    #image_name = "/home/goto/ILC/Deep_Learning/data/test/all_test_complete_track_image.npy"
    #variable_name = "/home/goto/ILC/Deep_Learning/data/test/all_test_complete_shaped.npy"
    #image_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03_chi100_track_image_v2_05.npy"
    #variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03_chi100_shaped.npy"
    #variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_08_shaped.npy"
    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_08_chi100_shaped.npy"

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

    # Evaluation of Vertex Finder
    classes = ["not connected", "primary", "secondary cc", "secondary bb", "secondary bc"]
    EVALTOOLS.ana_confusion_matrix(predict_vertex_finder, true_vertex_finder, model_name, classes, Norm=False)
    EVALTOOLS.ana_confusion_matrix(predict_vertex_finder, true_vertex_finder, model_name, classes, Norm=True)
    #EVALTOOLS.cut_curve(predict_vertex_finder, true_vertex_finder, model_name)

