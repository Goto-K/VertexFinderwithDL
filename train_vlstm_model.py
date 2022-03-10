from Networks.Tools import datatools
from Networks.VLSTMModel import models, training
import numpy as np


if __name__ == "__main__":
    print("Pair Model Training")
    
    model_name = "VLSTM_Model_Standard_cc_05-06_bb_06-07"

    data_path = "data/numpy/Pair_training_reshaped.npy"

    encoder_units = 256
    decoder_units= 256
    batch_size = 32
    nb_epochs = 100
    nb_samples = 50000
    validation_split = 0.2
    lr = 0.001

    #reinforce loss of tracks without pair
    pair_reinforce = False

    data = np.load(data_path)
    variables, vertex, position = datatools.GetVLSTMData(data)

    model = models.AttentionVLSTMModel(pair, tracks, ENCODER_UNITS=encoder_units, DECODER_UNITS=decoder_units):

    model, history = training.VLSTMModelTraining(model, model_name, pair, tracks, labels, 
                                                 BATCH_SIZE=batch_size, NB_EPOCHS=nb_epochs, NB_SAMPLES=nb_samples, VALIDATION_SPLIT=validation_split, LR=lr,
                                                 pair_reinforce=pair_reinforce)
    
    modeltools.SaveVLSTMHistory(history, model_name)
    modeltools.SaveAttentionVLSTMModel(model, model_name)

