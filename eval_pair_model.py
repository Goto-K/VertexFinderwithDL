from Networks.Tools import datatools, modeltools, evaltools
from Networks.PairModel import models, training
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Pair Model Evaluation...")
    
    model_name = "Pair_Model_Standard_201129_Standard_loss_weights_vertex0.1_position0.9_1000epochs_plus_loss_weights_vertex0.9_position0.1_1500epochs_plus_loss_weights_vertex0.95_position0.05_500epochs_lr_0.0001"
    history_path_first = "History_Pair_Model_Standard_201129_Standard_loss_weights_vertex0.1_position0.9_1000epochs_20201130"
    history_path_second = "History_Pair_Model_Standard_201129_Standard_loss_weights_vertex0.1_position0.9_1000epochs_plus_loss_weights_vertex0.9_position0.1_1500epochs_20201203"
    history_path_third = "History_Pair_Model_Standard_201129_Standard_loss_weights_vertex0.1_position0.9_1000epochs_plus_loss_weights_vertex0.9_position0.1_1500epochs_plus_loss_weights_vertex0.95_position0.05_500epochs_lr_0.0001_20201203"

    data_path = "data/numpy/Pair_evaluating_reshaped.npy"

    history_first = evaltools.LoadPairHistory(history_path_first)
    history_second = evaltools.LoadPairHistory(history_path_second)
    history_third = evaltools.LoadPairHistory(history_path_third)

    #loss, Vertex_Output_loss, Position_Output_loss, Vertex_Output_accuracy, Vertex_Output_mae, Position_Output_accuracy, Position_Output_mae
    
    Vertex_Output_loss = history_first['Vertex_Output_loss'] + history_second['Vertex_Output_loss'] + history_third['Vertex_Output_loss']
    val_Vertex_Output_loss = history_first['val_Vertex_Output_loss'] + history_second['val_Vertex_Output_loss'] + history_third['val_Vertex_Output_loss']
    Position_Output_loss = history_first['Position_Output_loss'] + history_second['Position_Output_loss'] + history_third['Position_Output_loss']
    val_Position_Output_loss = history_first['val_Position_Output_loss'] + history_second['val_Position_Output_loss'] + history_third['val_Position_Output_loss']

    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(111)
    ln11=ax1.plot(Vertex_Output_loss, color="red", label="Classification Loss - Train")
    ln12=ax1.plot(val_Vertex_Output_loss, color="red", label="Classification Loss - Validation", linestyle="dotted")
    
    ax2 = ax1.twinx()
    ln21=ax2.plot(Position_Output_loss, color="cyan", label="Regression Loss - Train")
    ln22=ax2.plot(val_Position_Output_loss, color="blue", label="Regression Loss - Validation", linestyle="dotted")
    
    h11, l11 = ax1.get_legend_handles_labels()
    h21, l21 = ax2.get_legend_handles_labels()
    
    ax1.legend(h11+h21, l11+l21)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Classification Loss")
    ax1.grid(True)
    ax2.set_ylabel("Regression Loss")
    plt.savefig("data/figure/loss/Loss_" + model_name + ".pdf")

    data = np.load(data_path)
    variables, true_vertex, true_position = datatools.GetPairData(data, Cov=True)
    model = modeltools.LoadPairModel(model_name)

    model.compile(loss={'Vertex_Output': 'categorical_crossentropy', 'Position_Output': 'mean_squared_logarithmic_error'},
                  optimizer='SGD',
                  metrics=['accuracy', 'mae'])

    predict_vertex, predict_position = model.predict([variables])
    predict_vertex = np.array(predict_vertex, dtype=float)

    classes = ["NC", "PV", "SVCC", "SVBB", "TVCC", "SVBC", "Others"]
    evaltools.ConfusionMatrix(predict_vertex, true_vertex, model_name, classes)
    evaltools.PlotRegression(predict_position[:, 0], true_position, model_name, MaxLog=5, Bins=1000)

    """
    rocs = np.load("data/numpy/roc/Efficiency_Curve_Pair_Model_Standard_201125_loss_weights_vertex0.1_position0.9_1000epochs_plus_loss_weights_vertex0.9_position0.1_1500epochs_plus_loss_weights_vertex0.8_position0.2_500epochs_lr_0.0001_thre_allsig_allbg_sig_bg.npy")
    evaltools.DrawPairROCCurve(rocs, classes, model_name)
    """

    evaltools.EfficiencyCurve(predict_vertex, true_vertex, model_name)

