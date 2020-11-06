import TOOLS
import numpy as np
from tqdm import tqdm


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

    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_08.npy"
    save_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_08_chi100_shaped.npy"

    _data = np.load(variable_name)

    data = []
    for datum in tqdm(_data):
        if datum[47] < 100:
            data.append(datum)

    data = np.array(data)

    data[:, 3], data[:, 25] = TOOLS.shaper_tanh(data[:, 3], 1.0, 1.0, 0.0, 0.0), TOOLS.shaper_tanh(data[:, 25], 1.0, 1.0, 0.0, 0.0) # d0
    data[:, 4], data[:, 26] = TOOLS.shaper_tanh(data[:, 4], 1.0, 1.0, 0.0, 0.0), TOOLS.shaper_tanh(data[:, 26], 1.0, 1.0, 0.0, 0.0) # z0
    data[:, 5], data[:, 27] = TOOLS.shaper_linear(data[:, 5], 1/np.pi, 0.0, 0.0), TOOLS.shaper_linear(data[:, 27], 1/np.pi, 0.0, 0.0) # phi
    data[:, 6], data[:, 28] = TOOLS.shaper_tanh(data[:, 6], 1.0, 200, 0.0, 0.0), TOOLS.shaper_tanh(data[:, 28], 1.0, 200, 0.0, 0.0) # omega
    data[:, 7], data[:, 29] = TOOLS.shaper_tanh(data[:, 7], 1.0, 0.3, 0.0, 0.0), TOOLS.shaper_tanh(data[:, 29], 1.0, 0.3, 0.0, 0.0) # tan(lambda)
    data[:, 9], data[:, 31] = TOOLS.shaper_tanh(data[:, 9], 1.0, 0.5, 5.0, 0.0), TOOLS.shaper_tanh(data[:, 31], 1.0, 0.5, 5.0, 0.0) # energy
    # covmatrix
    num1, num2 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    for n1, n2 in zip(num1, num2):
        data[:, n1], data[:, n2] = TOOLS.shaper_tanh(data[:, n1], 1.0, 8000, 0.0005, 0.0), TOOLS.shaper_tanh(data[:, n2], 1.0, 8000, 0.0005, 0.0)
    # chi-squared
    data[:, 47] = TOOLS.shaper_tanh(data[:, 47], 1.0, 0.3, 7.5, 0.0)
    # mass
    data[:, 51] = TOOLS.shaper_tanh(data[:, 51], 2.0, 0.15, 0.0, -1.0)
    # mag
    data[:, 52] = TOOLS.shaper_tanh(data[:, 52], 2.0, 0.2, 0.0, -1.0)
    # vec
    data[:, 53] = TOOLS.shaper_tanh(data[:, 53], 1.0, 0.1, 0.0, 0.0)
        
    # vertex position
    data[:, 48:51] = TOOLS.cartesian2polar(data[:, 48:51]) 
    data[:, 48] = TOOLS.shaper_tanh(data[:, 48], 2.0, 0.2, 0.0, -1.0)
    data[:, 49], data[:, 50] = TOOLS.shaper_linear(data[:, 49], 2/np.pi, np.pi/2, 0.0), TOOLS.shaper_linear(data[:, 50], 1/np.pi, 0.0, 0.0)

    np.save(save_name, data)

