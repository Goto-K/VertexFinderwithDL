import numpy as np
from keras.utils import np_utils
import random, os
from tqdm import tqdm

def LoadPairData(cc03_data_path, cc04_data_path, bb04_data_path, bb05_data_path, Cov=True):

    cc03data = np.load(cc03_data_path)
    cc04data = np.load(cc04_data_path)
    bb04data = np.load(bb04_data_path)
    bb05data = np.load(bb05_data_path)
    
    ccdata = np.concatenate([cc03data, cc04data], 0)
    bbdata = np.concatenate([bb04data, bb05data], 0)

    print("CC Data Sampling")
    newccdata = []
    for datum in tqdm(ccdata):
        if (datum[57]==0 and random.random()>0.5) or (datum[57]==1 and random.random()>0.5) or datum[57]==3 or datum[57]==4:
            continue
        elif datum[57]==5: # others
            datum[57] = 6
        newccdata.append(datum)

    print("BB Data Sampling")
    newbbdata = []
    for datum in tqdm(bbdata):
        if (datum[57]==0 and random.random()>0.5) or (datum[57]==1 and random.random()>0.5):
            continue
        elif datum[57]==2:
            datum[57] = 4
        elif datum[57]==4:
            datum[57] = 5
        elif datum[57]==5:
            datum[57] = 6
        newbbdata.append(datum)

    data = np.concatenate([newccdata, newbbdata], 0)
    data[:, 48:51] = cartesian2polar(data[:, 48:51]) 
    position = data[:, 48]
    np.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../data/numpy/Pair_training_original.npy"), data, allow_pickle=True)
    print("Original Data Saved")

    data[:, 3], data[:, 25] = shaper_tanh(data[:, 3], 1.0, 1.0, 0.0, 0.0), shaper_tanh(data[:, 25], 1.0, 1.0, 0.0, 0.0) # d0
    data[:, 4], data[:, 26] = shaper_tanh(data[:, 4], 1.0, 1.0, 0.0, 0.0), shaper_tanh(data[:, 26], 1.0, 1.0, 0.0, 0.0) # z0
    data[:, 5], data[:, 27] = shaper_linear(data[:, 5], 1/np.pi, 0.0, 0.0), shaper_linear(data[:, 27], 1/np.pi, 0.0, 0.0) # phi
    data[:, 6], data[:, 28] = shaper_tanh(data[:, 6], 1.0, 200, 0.0, 0.0), shaper_tanh(data[:, 28], 1.0, 200, 0.0, 0.0) # omega
    data[:, 7], data[:, 29] = shaper_tanh(data[:, 7], 1.0, 0.3, 0.0, 0.0), shaper_tanh(data[:, 29], 1.0, 0.3, 0.0, 0.0) # tan(lambda)
    data[:, 9], data[:, 31] = shaper_tanh(data[:, 9], 1.0, 0.5, 5.0, 0.0), shaper_tanh(data[:, 31], 1.0, 0.5, 5.0, 0.0) # energy
    # covmatrix
    num1, num2 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    for n1, n2 in zip(num1, num2):
        data[:, n1], data[:, n2] = shaper_tanh(data[:, n1], 1.0, 8000, 0.0005, 0.0), shaper_tanh(data[:, n2], 1.0, 8000, 0.0005, 0.0)

    np.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../data/numpy/Pair_training_reshaped.npy"), data, allow_pickle=True)
    print("Reshaped Data Saved")

    if Cov:
        variables = data[:, 3:47]
    else:
        variables = np.concatenate([data[:, 3:9], data[:, 25:31]], 0)
    vertex = np_utils.to_categorical(data[:, 57], 7)

    return variables, vertex, position


def GetPairData(data, Cov=True):
    if Cov:
        variables = data[:, 3:47]
    else:
        variables = np.concatenate([data[:, 3:9], data[:, 25:31]], 0)
    vertex = np_utils.to_categorical(data[:, 57], 7)
    position = data[:, 48]
    print("Data Loaded")

    return variables, vertex, position


def shaper_tanh(x, a=1.0, b=1.0, c=0.0, d=0.0):

    return a*np.tanh(b*(x-c))+d


def shaper_linear(x, a=1.0, b=0.0, c=0.0):

    return a*(x-b)+c


def under_sample(X, Y):

    from imblearn.under_sampling import RandomUnderSampler

    numy = np.min(np.count_nonzero(Y==0), np.count_nonzero(Y==1), np.count_nonzero(Y==2), np.count_nonzero(Y==3), np.count_nonzero(Y==4))
    rus = RandomUnderSampler(sampling_strategy={0:numy, 1:numy, 2:numy, 3:numy, 4:numy}, random_state=0)
    X_resampled, Y_resampled = rus.fit_sample(X, Y)

    return X_resampled, Y_resampled


def cartesian2polar(position):

    num = position.shape[0]
    newPosition = np.empty([num,3], dtype=np.float64)
    newPosition[:,0] = np.linalg.norm(position, axis=1)
    newPosition[:,1] = np.arccos(position[:,2] / newPosition[:,0])
    newPosition[:,2] = np.arctan2(position[:,1], position[:,0])
    nan_index = np.isnan(newPosition[:,1])
    newPosition[nan_index,1] = 0

    return newPosition


