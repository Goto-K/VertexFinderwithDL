#=======================================================================================================#
#== TOOLS FOR MODEL TRAINING AND SAVED =================================================================#
#=======================================================================================================#
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import os, json, pickle, codecs, h5py
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec

from datetime import datetime


def SavePairHistory(history, model_name):

    now = datetime.now()
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../log/PairHistory/History_" + model_name + "_" + now.strftime("%Y%m%d"))

    with codecs.open(path, mode='wb') as f:
        pickle.dump(history.history, f)

    print("History Saved")

def SavePairModel(model, model_name):

    model_json = model.to_json()
    open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../model/" + model_name + ".json"), 'w').write(model_json)
    print("Model Saved as json")
    model.save_weights(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../model/" + model_name + ".h5"));
    print("Weights Saved as h5")
    tf.saved_model.save(model, os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../model/" + model_name))

