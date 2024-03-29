from Networks.Tools import datatools, modeltools
from Networks.PairModel import models, training
import numpy as np


if __name__ == "__main__":
    print("Pair Model Training")
    
    model_name = "Pair_Model_Standard_201129_ModelC"

    data_path = "data/numpy/Pair_training_reshaped.npy"

    cov = True
    node_dim = 256
    batch_size = 1024
    nb_epochs = 4000
    validation_split = 0.2
    lr = 0.001
    #NC : 0.0090, PV : 0.0175, SVCC : 0.3375, SVBB : 0.1800, TVCC : 0.3509, SVBC : 0.1260, Others : 1.0
    custom_weights = [0.0090, 0.0175, 0.3375, 0.1800, 0.3509, 0.1260, 1.0]
    
    data = np.load(data_path)
    variables, vertex, position = datatools.GetPairData(data, Cov=cov)

    model = models.PairNetworkA(variables, vertex, NODE_DIM=node_dim)
    
    model, history = training.PairModelCTraining(model, model_name, variables, vertex, position, 
                                                 BATCH_SIZE=batch_size, NB_EPOCHS=nb_epochs, VALIDATION_SPLIT=validation_split, LR=lr,
                                                 Custom_Weights=custom_weights)

    modeltools.SavePairHistory(history, model_name)
    modeltools.SavePairModel(model, model_name)

