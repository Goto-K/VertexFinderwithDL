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
    print("Model Saved as pb")


def SaveVLSTMHistory(history, model_name):

    now = datetime.now()
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../log/VLSTMHistory/History_" + model_name + "_" + now.strftime("%Y%m%d"))

    with codecs.open(path, mode='wb') as f:
        pickle.dump(history.history, f)

    print("History Saved")


def SaveVLSTMModel(model, model_name):

    model_json = model.to_json()
    open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../model/" + model_name + ".json"), 'w').write(model_json)
    print("Model Saved as json")
    model.save_weights(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../model/" + model_name + ".h5"))
    print("Weights Saved as h5")
    tf.saved_model.save(model, os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../model/" + model_name))
    print("Model Saved as pb")


def SaveAttentionVLSTMModel(model, model_name):

    model_json = model.to_json()
    open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../model/" + model_name + ".json"), 'w').write(model_json)
    print("Model Saved as json")

    file = h5py.File(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../model/" + model_name + ".h5"), 'w')
    weight = model.get_weights()
    for i in range(len(weight)):
        file.create_dataset('weight'+str(i),data=weight[i])
    file.close()
    print("Weights Saved as h5")
    tf.saved_model.save(model, os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../model/" + model_name))
    print("Model Saved as pb")


def appendHist(h1, h2):
    if h1 == {}:
        return h2
    else:
        dest = {}
        for key, value in h1.items():
            dest[key] = value + h2[key]
        return dest


def LoadVLSTMModel(model_name):
    from ../VLSTMModel import layer

    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../model/" + model_name)
    json_path = model_path + ".json"
    h5_path = model_path + ".h5"
    structure = pathlib.Path(json_path).read_text()

    model = model_from_json(structure, custom_objects={'VLSTMCellSimple':layer.VLSTMCellSimple})
    model.load_weights(h5_path)
    model.summary()

    return model


def LoadAttentionVLSTMModel(model_name):
    from ../VLSTMModel import layer

    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../model/" + model_name)
    json_path = model_path + ".json"
    h5_path = model_path + ".h5"
    structure = pathlib.Path(json_path).read_text()

    model = model_from_json(structure, custom_objects={'VLSTMCellEncoder':layer.VLSTMCellEncoder, 'AttentionVLSTMCell':layer.AttentionVLSTMCell})
    file=h5py.File(h5_path,'r')
    weight = []
    for i in range(len(file.keys())):
        weight.append(file['weight'+str(i)][:])
    model.set_weights(weight)
    model.summary()

    return model


