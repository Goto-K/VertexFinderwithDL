import MODELTOOLS
import TOOLS
import PAIRMODEL
import numpy as np
from keras.utils import np_utils
from tensorflow.keras.models import Model



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
    #model_name = "Pair_Model_vfdnn04_1Msamples_affinity_250epochs_2"
    model_name = "Pair_Model_vfdnn04_1Msamples_2500epochs_NoCov_ChiPosMassMagVecTrSelV0Sel"

    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_04_shaped.npy"
    ori_variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_04.npy"
    #variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_04_chi100_shaped.npy"
    chi = False
    data = np.load(variable_name)
    ori_data = np.load(ori_variable_name)

    print("file load !")

    #variables = data[:1000000, 3:47] # low
    #variables = np.concatenate([data[:1000000, 3:9], data[:1000000, 25:31]], 1)
    #variables = data[:1000000, 3:57]
    variables = np.concatenate([data[:1000000, 3:9], data[:1000000, 25:31], data[:1000000, 47:57]], 1)
    print("Variable shape: " + str(variables.shape))
    test_variables = data[1000000:1200000, 3:47] # low
    labels = np_utils.to_categorical(data[:1000000, 57], 5)
    test_labels = np_utils.to_categorical(data[1000000:1200000, 57], 5)
    
    ori_data[:, 48:51] = TOOLS.cartesian2polar(ori_data[:, 48:51]) 
    pos = ori_data[:1000000, 48]

    #adacos_model, history = PAIRMODEL.pair_adacos_model(variables, labels, 2500, chi=chi)
    #adacos_model, history = PAIRMODEL.pair_adacos_model(variables, labels, 100, chi=chi)
    #model, history = PAIRMODEL.pair_affinity_model(variables, labels, 250)

    #model = PAIRMODEL.pair_adacos_eval(adacos_model)
    #model.evaluate(test_variables, test_labels)

    model, history = PAIRMODEL.pair_model(variables, labels, 2500, chi=chi)

    MODELTOOLS.save_pair_history(history, model_name)
    MODELTOOLS.save_model(model, model_name)
