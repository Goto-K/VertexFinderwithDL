import TOOLS
import numpy as np
import matplotlib.pyplot as plt

data_lowF_highF_noconvF = TOOLS.file_open("../summary/Text/HISTORY_Test_Comparison_Model_lowF_highF_noconvF_20200421.txt")
data_lowT_highF_noconvF = TOOLS.file_open("../summary/Text/HISTORY_Test_Comparison_Model_lowT_highF_noconvF_20200421.txt")
data_lowF_highT_noconvF = TOOLS.file_open("../summary/Text/HISTORY_Test_Comparison_Model_lowF_highT_noconvF_20200421.txt")
data_lowF_highT_noconvT = TOOLS.file_open("../summary/Text/HISTORY_Test_Comparison_Model_lowF_highT_noconvT_20200419.txt")
data_lowT_highF_noconvT = TOOLS.file_open("../summary/Text/HISTORY_Test_Comparison_Model_lowT_highF_noconvT_20200419.txt")

data_lowF_highF_noconvF = np.array(data_lowF_highF_noconvF, dtype=float)
data_lowT_highF_noconvF = np.array(data_lowT_highF_noconvF, dtype=float)
data_lowT_highF_noconvT = np.array(data_lowT_highF_noconvT, dtype=float)
data_lowF_highT_noconvF = np.array(data_lowF_highT_noconvF, dtype=float)
data_lowF_highT_noconvT = np.array(data_lowF_highT_noconvT, dtype=float)

plt.plot(data_lowF_highT_noconvF[:, 0], data_lowF_highT_noconvF[:, 1], color="blue", label="training accuracy : conv + low + high")
plt.plot(data_lowF_highT_noconvF[:, 0], data_lowF_highT_noconvF[:, 2], color="darkblue", label="validation accuracy : conv + low + high")
plt.plot(data_lowT_highF_noconvF[:, 0], data_lowT_highF_noconvF[:, 1], color="red", label="training accuracy : conv + low")
plt.plot(data_lowT_highF_noconvF[:, 0], data_lowT_highF_noconvF[:, 2], color="darkred", label="validation accuracy : conv + low")
plt.plot(data_lowF_highT_noconvT[:, 0], data_lowF_highT_noconvT[:, 1], color="orange", label="training accuracy : low + high")
plt.plot(data_lowF_highT_noconvT[:, 0], data_lowF_highT_noconvT[:, 2], color="darkorange", label="validation accuracy : low + high")
plt.plot(data_lowF_highF_noconvF[:, 0], data_lowF_highF_noconvF[:, 1], color="darkviolet", label="training accuracy : conv")
plt.plot(data_lowF_highF_noconvF[:, 0], data_lowF_highF_noconvF[:, 2], color="purple", label="validation accuracy : conv")
plt.plot(data_lowT_highF_noconvT[:, 0], data_lowT_highF_noconvT[:, 1], color="green", label="training accuracy : low")
plt.plot(data_lowT_highF_noconvT[:, 0], data_lowT_highF_noconvT[:, 2], color="darkgreen", label="validation accuracy : low")
plt.legend()
plt.show()
