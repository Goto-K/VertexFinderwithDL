from Networks.Tools import datatools, modeltools, evaltools
from Networks.PairModel import models, training
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Pair Model Evaluation...")
    print("Confusion Matrix...")

    modelA = np.load("data/numpy/roc/Efficiency_Curve_Pair_Model_Standard_201129_Standard_thre_allsig_allbg_sig_bg.npy")
    modelB = np.load("data/numpy/roc/Efficiency_Curve_Pair_Model_Standard_201129_ModelB_thre_allsig_allbg_sig_bg.npy")
    modelC = np.load("data/numpy/roc/Efficiency_Curve_Pair_Model_Standard_201129_ModelC_thre_allsig_allbg_sig_bg.npy")
    modelD = np.load("data/numpy/roc/Efficiency_Curve_Pair_Model_Standard_201129_ModelD_thre_allsig_allbg_sig_bg.npy")
    
    classes = ["NC", "PV", "SVCC", "SVBB", "TVCC", "SVBC", "Others"]
    for i, cla in enumerate(classes):
        tmpA = modelA[i]
        tmpB = modelB[i]
        tmpC = modelC[i]
        tmpD = modelD[i]

        fig = plt.figure()
        plt.plot(tmpA[:, 0], tmpA[:, 3]/tmpA[:, 1], label=cla + " - Model A", color="black", linewidth=4)
        plt.plot(tmpA[:, 0], tmpA[:, 4]/tmpA[:, 2], label="Background - Model A", linestyle="dashed", color="black", linewidth=4)
        plt.plot(tmpB[:, 0], tmpB[:, 3]/tmpB[:, 1], label=cla + " - Model B", color="blue", linewidth=4)
        plt.plot(tmpB[:, 0], tmpB[:, 4]/tmpB[:, 2], label="Background - Model B", linestyle="dashed", color="blue", linewidth=4)
        plt.plot(tmpC[:, 0], tmpC[:, 3]/tmpC[:, 1], label=cla + " - Model C", color="red", linewidth=4)
        plt.plot(tmpC[:, 0], tmpC[:, 4]/tmpC[:, 2], label="Background - Model C", linestyle="dashed", color="red", linewidth=4)
        plt.plot(tmpD[:, 0], tmpD[:, 3]/tmpD[:, 1], label=cla + " - Model D", color="green", linewidth=4)
        plt.plot(tmpD[:, 0], tmpD[:, 4]/tmpD[:, 2], label="Background - Model D", linestyle="dashed", color="green", linewidth=4)

        plt.title(cla + " Efficiency Score", fontsize=14)
        plt.xlabel(cla + " Score", fontsize=14)
        plt.ylabel("Efficiency", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(fontsize=13, loc='upper right')
        plt.grid(color='b', linestyle='dotted', linewidth=1)
        fig.set_figheight(9)
        fig.set_figwidth(4)
        fig.subplots_adjust(left=0.2)
        plt.tight_layout()
        plt.savefig("data/figure/roc/Summary_Efficiency_Curve_Pair_Model" + cla + ".pdf")
        plt.clf()

        fig = plt.figure()
        plt.plot(tmpA[:, 4]/tmpA[:, 2], tmpA[:, 3]/tmpA[:, 1], label="ROC Curve - Model A", color="black", linewidth=4)
        plt.plot(tmpB[:, 4]/tmpB[:, 2], tmpB[:, 3]/tmpB[:, 1], label="ROC Curve - Model B", color="blue", linewidth=4)
        plt.plot(tmpC[:, 4]/tmpC[:, 2], tmpC[:, 3]/tmpC[:, 1], label="ROC Curve - Model C", color="red", linewidth=4)
        plt.plot(tmpD[:, 4]/tmpD[:, 2], tmpD[:, 3]/tmpD[:, 1], label="ROC Curve - Model D", color="green", linewidth=4)
        plt.xscale('log')
        plt.xlim(0.0001, 1)

        plt.title(cla + " ROC Curve", fontsize=13)
        plt.xlabel("Background Efficiency", fontsize=13)
        plt.ylabel(cla + " Efficiency", fontsize=13)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        
        plt.legend(fontsize=13, loc='upper right')
        plt.grid(color='b', linestyle='dotted', linewidth=1)
        fig.set_figheight(9)
        fig.set_figwidth(4)
        fig.subplots_adjust(left=0.2)
        plt.tight_layout()
        plt.savefig("data/figure/roc/Summary_ROC_Curve_Pair_Model" + cla + ".pdf")
        plt.clf()


