#=======================================================================================================#
#===EVALTOOLS FOR DATA ANALYSIS=============================================================================#
#=======================================================================================================#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
import tensorflow as tf
import keras.backend as K
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from datetime import datetime
import itertools



def save_chi(predict, true, pred_vf, true_vf, x_data, model_name):

    now = datetime.now()
    path = "data/Chi_square_" + model_name + "_" + now.strftime("%Y%m%d") + ".txt"

    with open(path, mode='w') as f:
        f.write("number, input variables, chi-square, position, vertex finder\n")
        for i, (x, pv, tv, pc, px, py, pz, tc, tx, ty, tz) in enumerate(zip(x_data, pred_vf, true_vf, predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3], 
                                                                 true[:, 0], true[:, 1], true[:, 2], true[:, 3])):
            f.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\
                     {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n"\
                     .format(i, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10],
                             x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20],
                             x[21], x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30],
                             x[31], x[32], x[33], x[34], x[35], x[36], x[37], x[38], x[39], x[40],
                             x[41], pc, px, py, pz, tc, tx, ty, tz, pv[0], pv[1], pv[2], tv[0], tv[1], tv[2]))


def plot_confusion_matrix(cm, classes, model_name, Fontsize, title="Confusion_Matrix", cmap=plt.cm.Blues):

    plt.figure(figsize=(5,5), dpi=500)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(tick_marks[-1]+0.5, tick_marks[0]-0.5)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'), fontsize=Fontsize,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    now = datetime.now()
    plt.savefig(title + '_' + model_name + '_' + now.strftime("%Y%m%d") + '.png')
    plt.clf()


def ana_confusion_matrix(predict_vertex_finder, true_vertex_finder, model_name, classes, Norm=True):

    predict_vertex_finder = np.argmax(predict_vertex_finder, axis=1)
    true_vertex_finder = np.argmax(true_vertex_finder, axis=1)
    cmtmp = confusion_matrix(true_vertex_finder, predict_vertex_finder)
    cm = cmtmp.astype('float') / cmtmp.sum(axis=1)[:, np.newaxis]
    print(cmtmp)
    print(cm)
    if Norm:
        plot_confusion_matrix(cm, classes, model_name + "_Normalization", Fontsize=10)
    else:
        plot_confusion_matrix(cmtmp, classes, model_name, Fontsize=5)


def cut_curve(predict_vertex_finder, true_vertex_finder, model_name):

    true_vertex_finder = np.argmax(true_vertex_finder, axis=1)
    Ypp = np.concatenate([true_vertex_finder.reshape(-1, 1), predict_vertex_finder], 1)
    true_secondary = len([ypp for ypp in Ypp if ypp[0]==1])
    true_not_secondary = len(Ypp) - true_secondary
    cuts = np.arange(0.00, 1.01, 0.01)
    CUT = []
    ROC = []
    PR = []
    for cut in tqdm(cuts):
        predicted_secondary = 0
        true_positive = 0
        for ypp in Ypp:
            if ypp[2]>cut:
                predicted_secondary = predicted_secondary + 1
                if ypp[0]==1:
                    true_positive = true_positive + 1

        false_positive = predicted_secondary - true_positive
        false_negative = true_secondary - true_positive
        true_negative = true_not_secondary - false_positive

        if true_positive + false_positive == 0:
            Precision = true_positive
        else:
            Precision = true_positive / (true_positive + false_positive)
        if true_positive + false_negative == 0:
            Recall = true_positive
        else:
            Recall = true_positive / (true_positive + false_negative)
        if false_positive + true_negative == 0:
            FPR = false_positive
        else:
            FPR = false_positive / (false_positive + true_negative)

        CUT.append([cut, true_positive, false_negative, false_positive, true_negative])
        ROC.append([cut, Recall, FPR])
        PR.append([cut, Precision, Recall])

    CUT = np.array(CUT, dtype=float)
    cpath = "/home/goto/ILC/Deep_Learning/cut_performance_" + model_name + ".npy"
    np.save(cpath, CUT, fix_imports=True)
    ROC = np.array(ROC, dtype=float)
    rpath = "/home/goto/ILC/Deep_Learning/roc_curve_" + model_name + ".npy"
    np.save(rpath, ROC, fix_imports=True)
    PR = np.array(PR, dtype=float)
    ppath = "/home/goto/ILC/Deep_Learning/pr_curve_" + model_name + ".npy"
    np.save(ppath, PR, fix_imports=True)


def cut_curve_binary(predict, true, padding, model_name):

    predict = np.ravel(predict)
    true = np.ravel(true)
    padding = np.ravel(padding)
    cuts = np.arange(0.00, 1.01, 0.01)
    all_samples = len([p for p in padding if p==1])
    CUT = []
    ROC = []
    PR = []
    for cut in tqdm(cuts):
        positive = 0
        true_positive = 0
        negative = 0
        false_negative = 0
        for p,t,m in zip(predict, true, padding):
            if m == 0:
                continue
            elif p > cut:
                positive = positive + 1
                if t == 1:
                    true_positive = true_positive + 1
            else:
                negative = negative + 1
                if t == 1:
                    false_negative = false_negative + 1


        false_positive = positive - true_positive
        true_negative = negative - false_negative

        if true_positive + false_positive == 0:
            Precision = true_positive
        else:
            Precision = true_positive / (true_positive + false_positive)
        if true_positive + false_negative == 0:
            Recall = true_positive
        else:
            Recall = true_positive / (true_positive + false_negative)
        if false_positive + true_negative == 0:
            FPR = false_positive
        else:
            FPR = false_positive / (false_positive + true_negative)
        if false_positive + true_negative == 0:
            Specificity = true_negative
        else:
            Specificity = true_negative / (false_positive + true_negative)

        CUT.append([cut, true_positive, false_negative, false_positive, true_negative])
        ROC.append([cut, Recall, FPR, 1-Specificity])
        PR.append([cut, Precision, Recall])

    CUT = np.array(CUT, dtype=float)
    cpath = "/home/goto/ILC/Deep_Learning/cut_performance_" + model_name + ".npy"
    np.save(cpath, CUT, fix_imports=True)
    ROC = np.array(ROC, dtype=float)
    rpath = "/home/goto/ILC/Deep_Learning/roc_curve_" + model_name + ".npy"
    np.save(rpath, ROC, fix_imports=True)
    PR = np.array(PR, dtype=float)
    ppath = "/home/goto/ILC/Deep_Learning/pr_curve_" + model_name + ".npy"
    np.save(ppath, PR, fix_imports=True)


def ana_regression(predict_chi_square, true_chi_square, model_name, labels):

    for i in range(4):
        plt.figure()
        xlabel = labels[i] + "_pre"
        ylabel = labels[i] + "_true"
        title = model_name + "_" + labels[i] 
        plt.scatter(predict_chi_square[:, i], true_chi_square[:, i], s=1)

        plt.title(labels[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        now = datetime.now()
        plt.savefig(title + '_' + now.strftime("%Y%m%d") + '.png')
        plt.clf()


def ana_regression_heat(predict_chi_square, true_chi_square, model_name, labels, l=4, log=False):

    for i in range(l):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        xlabel = labels[i] + "_pred"
        ylabel = labels[i] + "_true"
        title = model_name + "_" + labels[i] + "_heatmap" 

        if log:
            H = ax.hist2d(predict_chi_square[:, i], true_chi_square[:, i], bins=np.logspace(0,11,50), norm=colors.LogNorm())
        else:
            H = ax.hist2d(predict_chi_square[:, i], true_chi_square[:, i], bins=50, norm=colors.LogNorm())

        ax.set_aspect('equal')
        fig.colorbar(H[3],ax=ax)

        plt.title(labels[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        plt.axes().set_aspect('equal')
        now = datetime.now()
        plt.savefig(title + '_' + now.strftime("%Y%m%d") + '.png')
        plt.clf()


def chi2_performance(pred_chi, true_chi, thr=0.3):
    abs_error = np.abs(np.log(pred_chi) - np.log(true_chi))
    truth, false = 0, 0
    for ae in abs_error:
        if ae < thr:
            truth+=1
        else:
            false+=1

    return truth, false
