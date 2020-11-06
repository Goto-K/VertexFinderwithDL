import MODELTOOLS
import PAIRMODEL
import numpy as np
from keras.utils import np_utils


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
    model_name = "GraphAttention_Test_01"

    #variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_02.npy"
    X_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_02_X_30000events.npy"
    A_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_02_A_30000events.npy"
    Label_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_02_Label_30000events.npy"

    MAX_TRACK = 50

    #data = np.load(variable_name)
    _X = np.load(X_name, allow_pickle=True)
    _A = np.load(A_name, allow_pickle=True)
    _Label = np.load(Label_name, allow_pickle=True)

    print("file load !")
    
    X = []
    A = []
    Targets = []
    Label = []
    for x, a, l in zip(_X, _A, _Label):
        x, a, l = np.array(x[0]), np.array(a[0]), np.array(l[0])
        mask_matrix = [[0 for j in range(int(MAX_TRACK))] for i in range(int(MAX_TRACK))]
        for i in range(int(a.shape[0])):
            for j in range(int(a.shape[1])):
                mask_matrix[i][j] = 1
        new_x = np.pad(x, [(0, MAX_TRACK - x.shape[0]), (0, 0)], 'constant')
        new_a = np.pad(a, [(0, MAX_TRACK - a.shape[0]), (0, MAX_TRACK - a.shape[1])], 'constant')
        new_l = np.pad(l, [(0, MAX_TRACK - l.shape[0]), (0, MAX_TRACK - l.shape[1])], 'constant')
        target = np.concatenate([np.expand_dims(new_l, axis=-1), np.expand_dims(mask_matrix, axis=-1)], axis=-1)
        X.append(new_x)
        A.append(new_a)
        Label.append(new_l)
        Targets.append(target)

    X, A, Label, Targets = np.array(X, dtype=float), np.array(A, dtype=float), np.array(Label, dtype=float), np.array(Targets, dtype=int)
    # Targets [30000, 50, 50, 2]
    print(X.shape)
    print(A.shape)
    print(Label.shape)
    print(Targets.shape)

    model, history = PAIRMODEL.pair_graph_model(X, A, Targets, 1500)
    MODELTOOLS.save_model(model, model_name)
    MODELTOOLS.save_history(history, model_name)
