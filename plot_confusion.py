from Networks.Tools import datatools, modeltools, evaltools
from Networks.PairModel import models, training
from matplotlib import ticker, cm, colors
from sklearn.metrics import confusion_matrix
from datetime import datetime
import itertools, os, pickle, codecs
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Pair Model Evaluation...")
    print("Confusion Matrix ...")

    modelA = np.load("data/numpy/confusion/Confusion_Matrix_Model_A.npy")
    modelB = np.load("data/numpy/confusion/Confusion_Matrix_Model_B.npy")
    modelC = np.load("data/numpy/confusion/Confusion_Matrix_Model_C.npy")
    modelD = np.load("data/numpy/confusion/Confusion_Matrix_Model_D.npy")

    classes = ["NC", "PV", "SVCC", "SVBB", "TVCC", "SVBC", "Others"]
    evaltools.plot_confusion_matrix(modelA[1], classes, "Model_A", Fontsize=10, title="Confusion_Matrix_Efficiency")
    evaltools.plot_confusion_matrix(modelA[2], classes, "Model_A", Fontsize=10, title="Confusion_Matrix_Purity")

    evaltools.plot_confusion_matrix(modelB[1] - modelA[1], classes, "Model_B", Fontsize=10, title="Relative_Value_Confusion_Matrix_Efficiency")
    evaltools.plot_confusion_matrix(modelB[1] - modelA[2], classes, "Model_B", Fontsize=10, title="Relative_Value_Confusion_Matrix_Purity")

    evaltools.plot_confusion_matrix(modelC[1] - modelA[1], classes, "Model_C", Fontsize=10, title="Relative_Value_Confusion_Matrix_Efficiency")
    evaltools.plot_confusion_matrix(modelC[1] - modelA[2], classes, "Model_C", Fontsize=10, title="Relative_Value_Confusion_Matrix_Purity")

    evaltools.plot_confusion_matrix(modelD[1] - modelA[1], classes, "Model_D", Fontsize=10, title="Relative_Value_Confusion_Matrix_Efficiency")
    evaltools.plot_confusion_matrix(modelD[1] - modelA[2], classes, "Model_D", Fontsize=10, title="Relative_Value_Confusion_Matrix_Purity")



    

    """
    data_path = "data/numpy/Pair_evaluating_reshaped.npy"
    data = np.load(data_path)
    index = np.random.permutation(len(data))
    data = data[index][:1000000]


    model_name = "Pair_Model_Standard_201129_Standard_loss_weights_vertex0.1_position0.9_1000epochs_plus_loss_weights_vertex0.9_position0.1_1500epochs_plus_loss_weights_vertex0.95_position0.05_500epochs_lr_0.0001"
    variables, true_vertex, true_position = datatools.GetPairData(data, Cov=True)
    model = modeltools.LoadPairModel(model_name)
    model.compile(loss={'Vertex_Output': 'categorical_crossentropy', 'Position_Output': 'mean_squared_logarithmic_error'},
                  optimizer='SGD',
                  metrics=['accuracy', 'mae'])
    predict_vertex, predict_position = model.predict([variables], verbose=1)
    predict_vertex = np.array(predict_vertex, dtype=float)
    predict_vertex = np.argmax(predict_vertex, axis=1)
    true_vertex = np.argmax(true_vertex, axis=1)
    cm = []
    cmtmp = confusion_matrix(true_vertex, predict_vertex)
    cmeff = cmtmp.astype('float') / cmtmp.sum(axis=1)[:, np.newaxis]
    cmpur = cmtmp.astype('float') / cmtmp.sum(axis=0)[np.newaxis, :]
    cm.append(cmtmp)
    cm.append(cmeff)
    cm.append(cmpur)
    np.save("data/numpy/confusion/Confusion_Matrix_Model_A.npy", np.array(cm))


    model_name = "Pair_Model_Standard_201129_ModelB_loss_weights_vertex0.1_position0.9_1000epochs_plus_loss_weights_vertex0.9_position0.1_1500epochs_plus_loss_weights_vertex0.95_position0.05_500epochs_lr_0.0001"
    variables, true_vertex, true_position = datatools.GetPairData(data, Cov=False)
    model = modeltools.LoadPairModel(model_name)
    model.compile(loss={'Vertex_Output': 'categorical_crossentropy', 'Position_Output': 'mean_squared_logarithmic_error'},
                  optimizer='SGD',
                  metrics=['accuracy', 'mae'])
    predict_vertex, predict_position = model.predict([variables], verbose=1)
    predict_vertex = np.array(predict_vertex, dtype=float)
    predict_vertex = np.argmax(predict_vertex, axis=1)
    true_vertex = np.argmax(true_vertex, axis=1)
    cm = []
    cmtmp = confusion_matrix(true_vertex, predict_vertex)
    cmeff = cmtmp.astype('float') / cmtmp.sum(axis=1)[:, np.newaxis]
    cmpur = cmtmp.astype('float') / cmtmp.sum(axis=0)[np.newaxis, :]
    cm.append(cmtmp)
    cm.append(cmeff)
    cm.append(cmpur)
    np.save("data/numpy/confusion/Confusion_Matrix_Model_B.npy", np.array(cm))


    model_name = "Pair_Model_Standard_201129_ModelC"
    variables, true_vertex, true_position = datatools.GetPairData(data, Cov=True)
    model = modeltools.LoadPairModel(model_name)
    model.compile(loss={'Vertex_Output': 'categorical_crossentropy'},
                  optimizer='SGD',
                  metrics=['accuracy'])
    predict_vertex = model.predict([variables, true_position])
    predict_vertex = np.array(predict_vertex, dtype=float)
    predict_vertex = np.argmax(predict_vertex, axis=1)
    true_vertex = np.argmax(true_vertex, axis=1)
    cm = []
    cmtmp = confusion_matrix(true_vertex, predict_vertex)
    cmeff = cmtmp.astype('float') / cmtmp.sum(axis=1)[:, np.newaxis]
    cmpur = cmtmp.astype('float') / cmtmp.sum(axis=0)[np.newaxis, :]
    cm.append(cmtmp)
    cm.append(cmeff)
    cm.append(cmpur)
    np.save("data/numpy/confusion/Confusion_Matrix_Model_C.npy", np.array(cm))


    model_name = "Pair_Model_Standard_201129_ModelD_loss_weights_vertex0.1_position0.9_1000epochs_plus_loss_weights_vertex0.9_position0.1_1500epochs_plus_loss_weights_vertex0.95_position0.05_500epochs_lr_0.0001"
    variables, true_vertex, true_position = datatools.GetPairData(data, Cov=True)
    model = modeltools.LoadPairModel(model_name)
    model.compile(loss={'Vertex_Output': 'categorical_crossentropy', 'Position_Output': 'mean_squared_logarithmic_error'},
                  optimizer='SGD',
                  metrics=['accuracy', 'mae'])
    predict_vertex, predict_position = model.predict([variables])
    predict_vertex = np.array(predict_vertex, dtype=float)
    predict_vertex = np.argmax(predict_vertex, axis=1)
    true_vertex = np.argmax(true_vertex, axis=1)
    cm = []
    cmtmp = confusion_matrix(true_vertex, predict_vertex)
    cmeff = cmtmp.astype('float') / cmtmp.sum(axis=1)[:, np.newaxis]
    cmpur = cmtmp.astype('float') / cmtmp.sum(axis=0)[np.newaxis, :]
    cm.append(cmtmp)
    cm.append(cmeff)
    cm.append(cmpur)
    np.save("data/numpy/confusion/Confusion_Matrix_Model_D.npy", np.array(cm))
    """
