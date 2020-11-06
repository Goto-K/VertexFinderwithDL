import numpy as np
import matplotlib.pyplot as plt
import TOOLS

BIN = 1000
data = np.load("/home/goto/ILC/Deep_Learning/data/test/test_complete.npy")

num = [3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 47, 48, 49, 50, 51, 52, 53]
bnames = ['d0', 'z0', 'phi', 'omega', 'tanlam', 'energy',
          'covmatrixd0d0', 'covmatrixd0z0', 'covmatrixd0ph', 'covmatrixd0om', 'covmatrixd0tl',
          'covmatrixz0z0', 'covmatrixz0ph', 'covmatrixz0om', 'covmatrixz0tl', 'covmatrixphph',
          'covmatrixphom', 'covmatrixphtl', 'covmatrixomom', 'covmatrixomtl', 'covmatrixtltl',
          'vchi2', 'vertex_position_r', 'vertex_position_theta', 'vertex_position_phi', 'mass', 'mag', 'vec']

data[:, 3], data[:, 25] = TOOLS.shaper_tanh(data[:, 3], 1.0, 1.0, 0.0, 0.0), TOOLS.shaper_tanh(data[:, 25], 1.0, 1.0, 0.0, 0.0) 
data[:, 4], data[:, 26] = TOOLS.shaper_tanh(data[:, 4], 1.0, 1.0, 0.0, 0.0), TOOLS.shaper_tanh(data[:, 26], 1.0, 1.0, 0.0, 0.0) # z0
data[:, 5], data[:, 27] = TOOLS.shaper_linear(data[:, 5], 1/np.pi, 0.0, 0.0), TOOLS.shaper_linear(data[:, 27], 1/np.pi, 0.0, 0.0) # phi
data[:, 6], data[:, 28] = TOOLS.shaper_tanh(data[:, 6], 1.0, 200, 0.0, 0.0), TOOLS.shaper_tanh(data[:, 28], 1.0, 200, 0.0, 0.0) # omega
data[:, 7], data[:, 29] = TOOLS.shaper_tanh(data[:, 7], 1.0, 0.3, 0.0, 0.0), TOOLS.shaper_tanh(data[:, 29], 1.0, 0.3, 0.0, 0.0) # tan(lambda)
data[:, 9], data[:, 31] = TOOLS.shaper_tanh(data[:, 9], 1.0, 0.5, 5.0, 0.0), TOOLS.shaper_tanh(data[:, 31], 1.0, 0.5, 5.0, 0.0) # energy

num1, num2 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
for n1, n2 in zip(num1, num2):
    data[:, n1], data[:, n2] = TOOLS.shaper_tanh(data[:, n1], 1.0, 8000, 0.0005, 0.0), TOOLS.shaper_tanh(data[:, n2], 1.0, 8000, 0.0005, 0.0)

data[:, 47] = TOOLS.shaper_tanh(data[:, 47], 1.0, 0.3, 7.5, 0.0)
data[:, 51] = TOOLS.shaper_tanh(data[:, 51], 2.0, 0.15, 0.0, -1.0)
data[:, 52] = TOOLS.shaper_tanh(data[:, 52], 2.0, 0.2, 0.0, -1.0)        
data[:, 53] = TOOLS.shaper_tanh(data[:, 53], 1.0, 0.1, 0.0, 0.0)

data[:, 48:51] = TOOLS.cartesian2polar(data[:, 48:51]) 
data[:, 48] = TOOLS.shaper_tanh(data[:, 48], 2.0, 0.2, 0.0, -1.0)        
data[:, 49], data[:, 50] = TOOLS.shaper_linear(data[:, 49], 2/np.pi, np.pi/2, 0.0), TOOLS.shaper_linear(data[:, 50], 1/np.pi, 0.0, 0.0)

notconnected = [datum for datum in data if datum[-2]==0]
primary = [datum for datum in data if datum[-2]==1]
secondarycc = [datum for datum in data if datum[-2]==2]
secondarybb = [datum for datum in data if datum[-2]==3] 
secondarybc = [datum for datum in data if datum[-2]==4]

notconnected, primary, secondarycc, secondarybb, secondarybc = np.array(notconnected, dtype=float), np.array(primary, dtype=float), np.array(secondarycc, dtype=float), np.array(secondarybb, dtype=float), np.array(secondarybc, dtype=float)

for n, bname in zip(num, bnames):
    plt.hist(notconnected[:, n], bins=BIN, color="black", label="not connected", histtype="step", density=True)
    plt.hist(primary[:, n], bins=BIN, color="green", label="primary vertex", histtype="step", density=True) 
    plt.hist(secondarycc[:, n], bins=BIN, color="red", label="secondary vertex cc", histtype="step", density=True)
    plt.hist(secondarybb[:, n], bins=BIN, color="blue", label="secondary vertex bb", histtype="step", density=True)
    plt.hist(secondarybc[:, n], bins=BIN, color="purple", label="secondary vertex bc", histtype="step", density=True)
    title = "Track : " + bname
    file = "US_Hist_Track_" + bname + ".png"
    plt.title(title)
    plt.legend()
    plt.savefig(file)
    plt.clf()
