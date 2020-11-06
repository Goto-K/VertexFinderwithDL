import EVALTOOLS
import TOOLS
import MODELBANK
import numpy as np
from keras.utils import np_utils
from tensorflow import keras
from tqdm import tqdm


if __name__ == "__main__":
    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_05.npy"
    X_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_05_X.npy"
    A_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_05_A.npy"
    Label_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_05_Label.npy"

    data = np.load(variable_name)

    # event check
    for i, ev in enumerate(data[:, 0]):
        if i>0 and data[i-1, 0]>ev:
            sys.exit("event numbers are not in ascending order")

    EVENT_NUMBER = int(data[-1, 0])
    print("Max Event : " + str(EVENT_NUMBER))
    A = []
    X = []
    Label = []

    for event in tqdm(range(EVENT_NUMBER)):
        edata = data[data[:, 0]==event]
        ntracks = (1 + np.sqrt(1 + 8*len(edata)))/2
        #ntracks = edata[-1, 1]

        adjacency_matrix = [[0 for j in range(int(ntracks))] for i in range(int(ntracks))] # Max number of tracks (padding)
        label_matrix = [[0 for j in range(int(ntracks))] for i in range(int(ntracks))] # Max number of tracks (padding)
        feature_matrix = []

        for datum in edata:
            theta1 = np.arctan(1/datum[7])
            if theta1 < 0:
                theta1 = theta1 + np.pi
            theta2 = np.arctan(1/datum[29])
            if theta2 < 0:
                theta2 = theta2 + np.pi

            eta1 = - np.log(np.tan(theta1/2)) 
            eta2 = - np.log(np.tan(theta2/2)) 
            distance = (datum[5] - datum[27])**2 + (eta1 - eta2)**2
            adjacency_matrix[int(datum[1])][int(datum[2])] = np.sqrt(distance)

            label_matrix[int(datum[1])][int(datum[2])] = 0
            if datum[57] != 0:
                label_matrix[int(datum[1])][int(datum[2])] = 1

            if datum[1]==ntracks-1:
                feature_matrix.append(datum[25:47]) 

        feature_matrix.append(edata[-1, 3:25]) # add last track

        for t in range(int(ntracks)):
            label_matrix[t][t] = 1 # loop matrix
        
        sym_adjacency_matrix = TOOLS.get_symmetric(adjacency_matrix)
        sym_label_matrix = TOOLS.get_symmetric(label_matrix)

        X.append([feature_matrix])
        A.append([sym_adjacency_matrix])
        Label.append([sym_label_matrix])

    np.save(X_name, X, fix_imports=True)
    np.save(A_name, A, fix_imports=True)
    np.save(Label_name, Label, fix_imports=True)
