#=======================================================================================================#
#===TOOLS FOR DATA ANALYSIS=============================================================================#
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



def data(filename="/home/ilc/gotok/WorkDir/python/data/ccbar_all_test_chi_ver2.npy"):

    NB_CLASSES = 3
    LOW = 42
    HIGH = 4
    
    XY = np.load(filename)
    X = XY[:, 0:LOW]
    Y = XY[:, LOW:]

    print(X.shape[0], 'X shape')
    print(Y.shape[0], 'Y shape')

    x_train, x_test, y_train_, y_test_ = train_test_split(X, Y, train_size=0.8)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    print(x_train.shape, 'train samples')
    print(x_test.shape, 'test samples')

    yconnect_train = np_utils.to_categorical(y_train_[:, HIGH], NB_CLASSES)
    yconnect_test = np_utils.to_categorical(y_test_[:, HIGH], NB_CLASSES)

    y_train = np.concatenate([y_train_[:, :HIGH], yconnect_train], 1)
    y_test = np.concatenate([y_test_[:, :HIGH], yconnect_test], 1)

    print(x_train.shape)
    print(y_train.shape)
    print(y_train[0])

    return x_train, y_train, x_test, y_test


def data_adjust(MAX_TRACK, data_track_labels, data_tracks, data_track_pairs, data_track_plabels):

    MAX_TRACK = MAX_TRACK + 1

    teach_track_labels = []
    teach_tracks = []
    teach_track_pairs = []
    for labels, tracks, pairs, plabels in zip(data_track_labels, data_tracks, data_track_pairs, data_track_plabels):
        for label, pair, plabel in zip(labels, pairs, plabels):
            pair = np.array(pair).reshape([44])
            teach_track = []
            teach_track_label = []
            if len(label) < MAX_TRACK:
                continue
            plabel_t = [list(x) for x in zip(*plabel)]
            for i, (l, track, pl) in enumerate(zip(label, tracks, plabel_t)):
                if i == MAX_TRACK:  break
                track = np.concatenate([track, pl])
                teach_track.append(track)
                teach_track_label.append(l)

            teach_track_labels.append(teach_track_label)
            teach_tracks.append(teach_track)
            teach_track_pairs.append(pair)

    teach_track_labels = np.array(teach_track_labels)
    teach_tracks = np.array(teach_tracks)
    teach_track_pairs = np.array(teach_track_pairs)
    
    print(teach_track_labels.shape)
    print(teach_tracks.shape)
    print(teach_track_pairs.shape)

    return teach_track_labels, teach_tracks, teach_track_pairs


def data_zero_padding(data_track_labels, data_tracks, data_track_pairs, data_track_plabels):

    length = 0
    for tracks in data_tracks:
        if len(tracks) > length:
            length = len(tracks)

    print(length)

    teach_track_labels = []
    teach_tracks = []
    teach_track_pairs = []
    for labels, tracks, pairs, plabels in zip(data_track_labels, data_tracks, data_track_pairs, data_track_plabels):
        for label, pair, plabel in zip(labels, pairs, plabels):
            pair = np.array(pair).reshape([44]).astype(np.float32)
            teach_track = []
            teach_track_label = []

            plabel_t = [list(x) for x in zip(*plabel)]

            for i, (l, track, pl) in enumerate(zip(label, tracks, plabel_t)):
                track = np.concatenate([[1], track, pl])
                #track = np.concatenate([track, pl])
                teach_track.append(track)
                teach_track_label.append(l)

            zero_track = np.zeros(track.shape)

            for j in range(length - i - 1):
                teach_track.append(zero_track)
                teach_track_label.append(0)

            teach_track_labels.append(teach_track_label)
            teach_tracks.append(teach_track)
            teach_track_pairs.append(pair)

    teach_track_labels = np.array(teach_track_labels)
    teach_tracks = np.array(teach_tracks)
    teach_track_pairs = np.array(teach_track_pairs)
    
    print(teach_track_labels.shape)
    print(teach_tracks.shape)
    print(teach_track_pairs.shape)

    print(type(teach_track_labels))
    print(type(teach_tracks))
    print(type(teach_track_pairs))

    print(type(teach_track_labels[0]))
    print(type(teach_tracks[0]))
    print(type(teach_track_pairs[0]))

    return teach_track_labels, teach_tracks, teach_track_pairs


def appendHist(h1, h2):
    if h1 == {}:
        return h2
    else:
        dest = {}
        for key, value in h1.items():
            dest[key] = value + h2[key]
        return dest


def load_history(path):

    #n = {}
    if os.path.exists(path):
        #with codecs.open(path, mode='r', encoding='utf-8') as f:
            #n = json.loads(f.read())
        with codecs.open(path, mode='rb') as f:
            n = pickle.load(f)

    return n


def save_history(history, model_name):

    now = datetime.now()
    #path = "HISTORY_" + model_name + "_" + now.strftime("%Y%m%d") + ".json"
    path = "HISTORY_" + model_name + "_" + now.strftime("%Y%m%d")

    #with codecs.open(path, mode='w', encoding='utf-8') as f:
        #json.dump(str(history), f, separators=(',', ':'), sort_keys=True, indent=4)
    with codecs.open(path, mode='wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

    print("history saved")


def save_pair_history(history, model_name):

    now = datetime.now()
    #path = "HISTORY_" + model_name + "_" + now.strftime("%Y%m%d") + ".json"
    path = "HISTORY_" + model_name + "_" + now.strftime("%Y%m%d")

    #with codecs.open(path, mode='w', encoding='utf-8') as f:
        #json.dump(str(history), f, separators=(',', ':'), sort_keys=True, indent=4)
    with codecs.open(path, mode='wb') as f:
        pickle.dump(history.history, f)

    print("history saved")


def save_model(model, model_name):

    model_json = model.to_json()
    open("/home/goto/ILC/Deep_Learning/model/" + model_name + ".json", 'w').write(model_json)
    print("model saved")
    model.save_weights("/home/goto/ILC/Deep_Learning/model/" + model_name + ".h5");
    print("weights saved")

def save_attention_model(model, model_name):

    model_json = model.to_json()
    open("/home/goto/ILC/Deep_Learning/model/" + model_name + ".json", 'w').write(model_json)
    print("model saved")

    file = h5py.File("/home/goto/ILC/Deep_Learning/model/" + model_name + ".h5",'w')
    weight = model.get_weights()
    for i in range(len(weight)):
        file.create_dataset('weight'+str(i),data=weight[i])
    file.close()
    print("weights saved")

def save_bestmodel(model, model_name):

    model_json = model.to_json()
    open("/home/goto/ILC/Deep_Learning/model/BEST_" + model_name + ".json", 'w').write(model_json)
    print("model saved")
    model.save_weights("./model/BEST_" + model_name + ".h5");
    print("weights saved")


class RepeatVector4D(Layer):

    def __init__(self, n, **kwargs):
        self.n = n
        self.input_spec = [InputSpec(ndim=3)]
        super(RepeatVector4D, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.n, input_shape[1], input_shape[2])

    def call(self, x, mask=None):
        x = K.expand_dims(x, 1)
        pattern = K.stack([1, self.n, 1, 1])
        return K.tile(x, pattern)

def squeeze_rear2axes_operator( x4d ) :
    shape = tf.shape( x4d ) # get dynamic tensor shape
    x3d = tf.reshape( x4d, [shape[0], shape[1], shape[2] * shape[3]] )
    return x3d

def squeeze_rear2axes_shape( x4d_shape ) :
    in_batch, in_tracknum1, in_tracknum2, in_attdim = x4d_shape
    if ( None in [ in_tracknum1, in_tracknum2 ] ) :
        output_shape = ( in_batch, None, None )
    else :
        output_shape = ( in_batch, in_tracknum1, in_tracknum2 * in_attdim )
    return output_shape

