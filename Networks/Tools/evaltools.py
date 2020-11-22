import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
from sklearn.metrics import confusion_matrix
from datetime import datetime
import itertools



def plot_confusion_matrix(cm, classes, model_name, Fontsize, title="Confusion_Matrix", cmap=plt.cm.Blues):

    plt.figure(figsize=(5,5), dpi=500)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    #plt.ylim(tick_marks[-1]+0.5, tick_marks[0]-0.5)
    plt.ylim(tick_marks[0]-0.5, tick_marks[-1]+0.5)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'), fontsize=Fontsize,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    now = datetime.now()
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../data/figure/' + title + '_' + model_name + '.png'))
    plt.clf()


def ConfusionMatrix(predict_vertex_finder, true_vertex_finder, model_name, classes):

    predict_vertex_finder = np.argmax(predict_vertex_finder, axis=1)
    true_vertex_finder = np.argmax(true_vertex_finder, axis=1)
    cmtmp = confusion_matrix(true_vertex_finder, predict_vertex_finder)
    cmeff = cmtmp.astype('float') / cmtmp.sum(axis=1)[:, np.newaxis]
    cmpur = cmtmp.astype('float') / cmtmp.sum(axis=0)[:, np.newaxis]
    print(cmtmp)
    plot_confusion_matrix(cmtmp, classes, model_name, Fontsize=5)
    plot_confusion_matrix(cmeff, classes, model_name + "_Efficiency", Fontsize=10)
    plot_confusion_matrix(cmpur, classes, model_name + "_Purity", Fontsize=10)


def EfficiencyCurve(pred, true, model_name):

    true = np.argmax(true, axis=1)
    true_pred = np.concatenate([true.reshape(-1, 1), pred], 1)

    classes = np.arange(len(pred[0]))
    cuts = np.arange(0.00, 1.01, 0.01)
    efficiency_curve = []
    for cla in classes:
        class_efficiency_curve = []
        all_signal = len([datum for datum in true_pred if datum[0]==cla])
        all_background = len([datum for datum in true_pred if datum[0]!=cla])
        for cut in cuts:
            signal = len([datum for datum in true_pred if datum[cla+1] > cut and datum[0]==cla])
            background = len([datum for datum in true_pred if datum[cla+1] > cut and datum[0]!=cla])
            class_efficiency_curve.append([cut, all_signal, all_background, signal, background])
        efficiency_curve.append(class_efficiency_curve)

    efficiency_curve = np.array(efficiency_curve, dtype=float)
    np.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../data/numpy/Efficiency_Curve_" + model_name + "_thre_allsig_allbg_sig_bg.npy"), efficiency_curve, allow_pickle=True)


def PlotRegression(pred, true, model_name, MaxLog=5, Bins=1000):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    xlabel = "Predicted Vertex Position - Radial Direction"
    ylabel = "True Vertex Position - Radial Direction"
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../data/figure/Vertex_Position_" + model_name + ".png")

    H = ax.hist2d(pred, true, bins=np.logspace(0, MaxLog, Bins), norm=colors.LogNorm())

    ax.set_aspect('equal')
    fig.colorbar(H[3], ax=ax)

    plt.title("Vertex Position - Radial Direction")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.axes().set_aspect('equal')
    plt.savefig(save_path)

