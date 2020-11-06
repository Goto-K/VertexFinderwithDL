import ROOTTOOLS
import TOOLS
import sys
import numpy as np


if __name__ == "__main__":
    for i in sys.argv[1:]:
        pdpath = "/home/goto/ILC/Deep_Learning/data/"
        npyname_s = pdpath + "vfdnn_" + str(i) + "_shaped.npy"
        npyname = pdpath + "vfdnn_" + str(i) + ".npy"
        fnames = ["/home/goto/ILC/Deep_Learning/data/root/vfdnn_" + str(i) + "_complete.root"]
        #fnames = ["/home/goto/ILC/Deep_Learning/data/test/test_complete.root"]

        bnames = ['nevent', 'ntr1track', 'ntr2track',
                  # Track 1 : 22 input variables
                  'tr1d0', 'tr1z0', 'tr1phi', 'tr1omega', 'tr1tanlam', 'tr1charge', 'tr1energy',
                  'tr1covmatrixd0d0', 'tr1covmatrixd0z0', 'tr1covmatrixd0ph', 'tr1covmatrixd0om', 'tr1covmatrixd0tl',
                  'tr1covmatrixz0z0', 'tr1covmatrixz0ph', 'tr1covmatrixz0om', 'tr1covmatrixz0tl', 'tr1covmatrixphph',
                  'tr1covmatrixphom', 'tr1covmatrixphtl', 'tr1covmatrixomom', 'tr1covmatrixomtl', 'tr1covmatrixtltl',
                  # Track 2 : 22 input variables
                  'tr2d0', 'tr2z0', 'tr2phi', 'tr2omega', 'tr2tanlam', 'tr2charge', 'tr2energy',
                  'tr2covmatrixd0d0', 'tr2covmatrixd0z0', 'tr2covmatrixd0ph', 'tr2covmatrixd0om', 'tr2covmatrixd0tl',
                  'tr2covmatrixz0z0', 'tr2covmatrixz0ph', 'tr2covmatrixz0om', 'tr2covmatrixz0tl', 'tr2covmatrixphph',
                  'tr2covmatrixphom', 'tr2covmatrixphtl', 'tr2covmatrixomom', 'tr2covmatrixomtl', 'tr2covmatrixtltl', 
                  # fitter feature value
                  'vchi2', 'vposx', 'vposy', 'vposz', 'mass', 'mag', 'vec', 'tr1selection', 'tr2selection', 'v0selection',
                  'connect', 'lcfiplustag', 
                  'tr1id', 'tr1pdg', 'tr1ssid', 'tr1sspdg', 'tr1ssc', 'tr1ssb', 'tr1oth', 'tr1pri', 
                  'tr2id', 'tr2pdg', 'tr2ssid', 'tr2sspdg', 'tr2ssc', 'tr2ssb', 'tr2oth', 'tr2pri'
                  ]

        data, nevent = ROOTTOOLS.fileload_setnames(fnames, "track0", bnames)
        data = np.array(data).reshape(nevent, len(bnames))
        np.save(npyname, data, fix_imports=True)
        #np.save("test_complete.npy", data, fix_imports=True)

        # a tanh(b(x-c)) + d
        # a (x-b) + c
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


        print("saving...")
        #np.save(npyname_s, data, fix_imports=True)
        np.save("gcn_vfdnn_03_shaped.npy", data, fix_imports=True)
