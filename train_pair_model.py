import Networks


if __name__ == "__main__":
    print("Pair Model Training")
    
    model_name = "Pair_Model_Standard_cc_03-04_bb_04-05"
    model_path = "model/" + model_name

    bb04_data_path = "data/numpy/bb/bb_04_shaped.npy"
    bb04_original_data_path = "data/numpy/bb/bb_04.npy"
    bb05_data_path = "data/numpy/bb/bb_05_shaped.npy"
    bb05_original_data_path = "data/numpy/bb/bb_05.npy"

    chi = True
    node_dim = 256
    batch_size = 1024
    nb_epochs = 2500
    validation_split = 0.2
    lr = 0.001
    custom_weights = [0.02, 0.08, 1, 0.5, 0.37]
    loss_weights = [0.5, 0.5]

    data, original_data = Networks.Tools.datatools.LoadPairData(cc03_data_path, cc03_original_data_path, cc04_data_path, cc04_original_data_path, 
                                                                 bb04_data_path, bb04_original_data_path, bb05_data_path, bb05_original_data_path)

    variables, vertex, position = Networks.Tools.datatools.GetPairData(data, original_data, Chi=chi)

    model = Networks.PairModel.models.PairModelStandard(variables, vertex, NODE_DIM=node_dim)
    history = Networks.PairModel.training.PairModelTraining(model, model_name, variables, vertex, position, 
                                                            BATCH_SIZE=batch_size, NB_EPOCHS=nb_epochs, VALIDATION_SPLIT=validation_split, LR=lr,
                                                            Custom_Weights=custom_weights, Loss_Weights=loss_weights)


    Networks.Tools.modeltools.SavePairHistory(history, model_name)
    Networks.Tools.modeltools.SavePairModel(model, model_name)
 
