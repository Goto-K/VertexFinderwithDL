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
    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03.npy"

    BATCH_SIZE = 4096
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)

    images = np.load(image_name)
    data = np.load(variable_name)

    variables = data[1200000:1500000, 3:57]
    #variables = data[:, 3:47]
    true_vertex_finder = np_utils.to_categorical(data[1200000:1500000, 57], 5)

    model = MODELBANK.load_model(model_name)
        
    model.compile(loss='categorical_crossentropy', 
                  optimizer=OPTIMIZER, 
                  metrics=['accuracy'])
    
    #predict_vertex_finder = model.predict([images, variables])
    predict_vertex_finder = model.predict([variables])
    predict_vertex_finder = np.array(predict_vertex_finder, dtype=float)

    # Evaluation of Vertex Finder
    #classes = ["not connected", "primary", "secondary cc", "secondary bb", "secondary bc"]
    #EVALTOOLS.ana_confusion_matrix(predict_vertex_finder, true_vertex_finder, model_name, classes, Norm=False)
    #EVALTOOLS.ana_confusion_matrix(predict_vertex_finder, true_vertex_finder, model_name, classes, Norm=True)
    #EVALTOOLS.cut_curve(predict_vertex_finder, true_vertex_finder, model_name)

    # event check
    for i, ev in enumerate(data[:, 0]):
        if i>0 && data[i-1, 0]>ev:
            sys.exit("event numbers are not in ascending order")

    empty_track = [0 for i in range(22)] # empty dummy track

    EVENT_NUMBER = [i for i in range(data[-1, 0])]
    A = []
    X = []

    for event in EVENT_NUMBER:
        adjacency_matrix_0 = [[0 for j in range(55)] for i in range(55)] # Max number of tracks (padding)
        adjacency_matrix_1 = [[0 for j in range(55)] for i in range(55)] # Max number of tracks (padding)
        adjacency_matrix_2 = [[0 for j in range(55)] for i in range(55)] # Max number of tracks (padding)
        feature_matrix = []

        edata = data[data[:, 0]==event]
        ntracks = edata[-1, 1]

        for datum in edata:
            adjacency_matrix_0[datum[1]][datum[2]] = pred[0]
            adjacency_matrix_1[datum[1]][datum[2]] = pred[1]
            adjacency_matrix_2[datum[1]][datum[2]] = pred[2]

            if datum[1]==ntracks:
                feature_matrix.append(datum[25:47]) 

        feature_matrix.append(edata[-1, 3:25]) # add last track
        
        for iempty in range(55-ntracks):
            feature_matrix.append(empty_track) # add dummy tracks

        sym_adjacency_matrix_0 = TOOLS.get_symmetric(adjacency_matrix_0)
        sym_adjacency_matrix_1 = TOOLS.get_symmetric(adjacency_matrix_1)
        sym_adjacency_matrix_2 = TOOLS.get_symmetric(adjacency_matrix_2)

        A.append([sym_adjacency_matrix_0, sym_adjacency_matrix_1, sym_adjacency_matrix_2])
        X.append([feature_matrix])

    np.save("test_adjacency_matrix.npy", A, fix_imports=True)
    np.save("test_feature_matrix.npy", X, fix_imports=True)
