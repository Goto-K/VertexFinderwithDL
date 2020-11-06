import MODELTOOLS
import LSTMMODEL
import CUSTOM
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
from sklearn.metrics import accuracy_score


"""
'nevent', 'ntr1track', 'ntr2track', # 0 1 2

'tr1d0', 'tr1z0', 'tr1phi', 'tr1omega', 'tr1tanlam', 'tr1charge', 'tr1energy', # 3 4 5 6 7 8 9
'tr1covmatrixd0d0', 'tr1covmatrixd0z0', 'tr1covmatrixd0ph', 'tr1covmatrixd0om', 'tr1covmatrixd0tl', # 10 11 12 13 14
'tr1covmatrixz0z0', 'tr1covmatrixz0ph', 'tr1covmatrixz0om', 'tr1covmatrixz0tl', 'tr1covmatrixphph', # 15 16 17 18 19
'tr1covmatrixphom', 'tr1covmatrixphtl', 'tr1covmatrixomom', 'tr1covmatrixomtl', 'tr1covmatrixtltl', # 20 21 22 23 24

'tr2d0', 'tr2z0', 'tr2phi', 'tr2omega', 'tr2tanlam', 'tr2charge', 'tr2energy', # 25 26 27 28 29 30 31
'tr2covmatrixd0d0', 'tr2covmatrixd0z0', 'tr2covmatrixd0ph', 'tr2covmatrixd0om', 'tr2covmatrixd0tl', # 32 33 34 35 36
'tr2covmatrixz0z0', 'tr2covmatrixz0ph', 'tr2covmatrixz0om', 'tr2covmatrixz0tl', 'tr2covmatrixphph', # 37 38 39 40 41
'tr2covmatrixphom', 'tr2covmatrixphtl', 'tr2covmatrixomom', 'tr2covmatrixomtl', 'tr2covmatrixtltl', # 42 43 44 45 46

'vchi2', 'vposx', 'vposy', 'vposz', 'mass', 'mag', 'vec', 'tr1selection', 'tr2selection', 'v0selection', # 47 48 49 50 51 52 53 54 55 56
'connect', 'lcfiplustag' # 57 58
"""


if __name__ == "__main__":
    
    encoder_input = 256
    decoder_units = 256
    nb_samples = 50000
    nb_epochs = 100


    name_teach_track_pairs = "/home/goto/ILC/Deep_Learning/data/tmp_vfdnn_06_pairs.npy"
    name_teach_tracks = "/home/goto/ILC/Deep_Learning/data/tmp_vfdnn_06_tracks.npy"
    name_targets = "/home/goto/ILC/Deep_Learning/data/tmp_vfdnn_06_targets.npy"
    teach_track_pairs = np.load(name_teach_track_pairs, allow_pickle=True)
    teach_tracks = np.load(name_teach_tracks, allow_pickle=True)
    targets = np.load(name_targets, allow_pickle=True)
    print(teach_tracks.shape)
    print(teach_track_pairs.shape)

    load_model = tf.saved_model.load("/home/goto/ILC/Deep_Learning/model/Attention_Bidirectional_VLSTM_Model_vfdnn06_50000samples_100epochs/")
    infer = load_model.signatures["serving_default"]
    print(infer.structured_input_signature)
    #labels = infer(tf.constant(teach_track_pairs[200000:200010], dtype=tf.float32), tf.constant(teach_tracks[200000:200010, :, :23], dtype=tf.float32))
    infer._num_positional_args = 3
    labels = infer(tf.constant(teach_tracks[200000:200010, :12, :23], dtype=tf.float32), tf.constant(teach_tracks[200000:200010, :, :23], dtype=tf.float32),
                   tf.constant(teach_track_pairs[200000:200010], dtype=tf.float32))["Decoder_Attention_VLSTM"]
    print(type(labels))
    numlabels = labels.numpy()
    print(type(numlabels))

    print(numlabels) 
    numlabels = numlabels.astype(np.float32)
    print(targets[200000:200010, :12, 0]) 
    #acc = accuracy_score(targets[200000:200010], numlabels)
    #print(acc)
