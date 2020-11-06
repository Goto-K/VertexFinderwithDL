import tensorflow as tf
import pathlib, h5py
from tensorflow.keras.models import model_from_json
import CUSTOM
import LOSS


def load_model(model_name):
    #model = load_model("/home/ilc/gotok/WorkDir/python/model/" + model_name + "_" + date + ".h5", compile=False)
    model_path = "/home/goto/ILC/Deep_Learning/model/" + model_name
    json_path = model_path + ".json"
    h5_path = model_path + ".h5"
    structure = pathlib.Path(json_path).read_text()
    model = model_from_json(structure, custom_objects={'VLSTMCell':CUSTOM.VLSTMCell, 'VLSTMCell_M':CUSTOM.VLSTMCell_M, 'AttentionVLSTMCell':CUSTOM.AttentionVLSTMCell})
    file=h5py.File(h5_path,'r')
    weight = []
    for i in range(len(file.keys())):
        weight.append(file['weight'+str(i)][:])
    model.set_weights(weight)
    model.summary()
    return model

def load_model_p(model_name):
    #model = load_model("/home/ilc/gotok/WorkDir/python/model/" + model_name + "_" + date + ".h5", compile=False)

    model_path = "/home/goto/ILC/Deep_Learning/model/" + model_name
    json_path = model_path + ".json"
    h5_path = model_path + ".h5"
    structure = pathlib.Path(json_path).read_text()

    model = model_from_json(structure, custom_objects={'custom_categorical_crossentropy': LOSS.custom_categorical_crossentropy([0.02, 0.08, 1, 0.5, 0,37])})
    model.load_weights(h5_path)
    model.summary()

    return model


if __name__ == '__main__':
    model_name = "Attention_Bidirectional_VLSTM_Model_vfdnn06_50000samples_100epochs_ps_100epochs_s"
    loaded_model = load_model(model_name)
    print("##### Model Load ####")
    loaded_model.save(model_name, save_format="tf")

