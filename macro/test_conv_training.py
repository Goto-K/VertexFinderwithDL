import MODELTOOLS
import MODELBANK
import CONVMODEL
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

    model_name = "Test_Pair_Conv_Model_0.05Msamples_100epochs"

    image_name1 = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03_image_0.05M.npy"
    #image_name2 = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03_image_0.05M-0.10M.npy"
    #image_name3 = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03_image_0.10M-0.15M.npy"
    #image_name4 = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03_image_0.15M-0.20M.npy"
    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03_shaped.npy"

    fulldata = np.load(variable_name)
    variables = fulldata[:50000, 3:47]

    targets = np_utils.to_categorical(fulldata[:50000, 57], 5)

    images = np.load(image_name1)
    #images2 = np.load(image_name2)
    #images3 = np.load(image_name3)
    #images4 = np.load(image_name4)
    #images = np.concatenate([images1, images2], 0)

    print("file load !")

    model, history = CONVMODEL.conv_model(images, variables, targets, 100, chi=False)
    
    MODELTOOLS.save_pair_history(history, model_name)
    MODELTOOLS.save_model(model, model_name)
