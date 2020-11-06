import TOOLS
import sys
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
    for i in sys.argv[1:]:
        pdpath = "/home/goto/ILC/Deep_Learning/data/vfdnn/"
        npyname = pdpath + "vfdnn_" + str(i) + "_chi100.npy"
        npyname_s = pdpath + "vfdnn_" + str(i) + "_chi100_track_image_05.npy"
        #npyname = pdpath + "test/test_complete.npy"
        #npyname_s = pdpath + "test/test_complete_track_image.npy"

        print("loading...")
        data = np.load(npyname)
        tr1s, tr2s = data[1200000:1500000, 3:9], data[1200000:1500000, 25:31]
        t = np.arange(-10.0, 10.0, 0.1)

        print("calculating...")
        data_tracks = []
        for tr1, tr2 in tqdm(zip(tr1s, tr2s)):
            track1, track2 = TOOLS.t_tracker(tr1, tr2, t, curvature=False)
            data_tracks.append([track1, track2])
        data_tracks = np.array(data_tracks, dtype='float')

        print("saving...")
        np.save(npyname_s, data_tracks, fix_imports=True)
