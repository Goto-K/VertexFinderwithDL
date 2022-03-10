from Networks.Tools import datatools
from Networks.PairModel import models, training


if __name__ == "__main__":
    print("Pair Data Making")
    
    model_name = "Pair_Model_Standard_cc_03-04_bb_04-05"
    model_path = "model/" + model_name

    cc03_data_path = "data/numpy/cc/cc_07_pair.npy"
    cc04_data_path = "data/numpy/cc/cc_08_pair.npy"
    bb04_data_path = "data/numpy/bb/bb_08_pair.npy"
    bb05_data_path = "data/numpy/bb/bb_09_pair.npy"

    cov = True
    node_dim = 256
    batch_size = 1024
    nb_epochs = 2500
    validation_split = 0.2
    lr = 0.001
    custom_weights = [0.02, 0.08, 1, 0.5, 0.37]
    loss_weights = [0.5, 0.5]

    #variables, vertex, position = datatools.LoadPairData(cc03_data_path, cc04_data_path, bb04_data_path, bb05_data_path, saved_data_name="Pair_training", Cov=cov)
    variables, vertex, position = datatools.LoadPairData(cc03_data_path, cc04_data_path, bb04_data_path, bb05_data_path, saved_data_name="Pair_evaluating", Cov=cov)

