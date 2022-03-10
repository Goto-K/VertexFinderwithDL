from Networks.Tools import datatools, modeltools, evaltools
from Networks.PairModel import models, training
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Pair Model Training")
    
    model_name = "Pair_Model_Standard_Overfitting_201125_loss_weights_vertex0.0_position1.0"
    history_path = "History_Pair_Model_Standard_Overfitting_201125_loss_weights_vertex0.0_position1.0_20201129"
    data_path = "data/numpy/Pair_evaluating_reshaped.npy"


    history = evaltools.LoadHistory(history_path)
    Vertex_Output_loss = history['Vertex_Output_loss']
    val_Vertex_Output_loss = history['val_Vertex_Output_loss']
    Position_Output_loss = history['Position_Output_loss']
    val_Position_Output_loss = history['val_Position_Output_loss']

    
    """
    plt.plot(Vertex_Output_loss, color="magenta", label="Classification Loss - Train")
    plt.plot(val_Vertex_Output_loss, color="red", label="Classification Loss - Validation")
    """
    plt.plot(Position_Output_loss, color="cyan", label="Regression Loss - Train")
    plt.plot(val_Position_Output_loss, color="blue", label="Regression Loss - Validation")
    plt.legend()
    plt.xlabel("Epochs")
    #plt.ylabel("Classification Loss")
    plt.ylabel("Regression Loss")
    plt.grid(True)
    plt.savefig("data/figure/Loss_" + model_name + ".png")

    """

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


    evaltools.EfficiencyCurve(predict_vertex, true_vertex, model_name)

    evaltools.PlotRegression(predict_position, true_position, model_name, MaxLog=5, Bins=1000)

    """
