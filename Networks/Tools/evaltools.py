import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
from sklearn.metrics import confusion_matrix
from datetime import datetime
import itertools, os, pickle, codecs
from tqdm import tqdm



def plot_confusion_matrix(cm, classes, model_name, Fontsize, title="Confusion_Matrix", cmap=plt.cm.Blues):

    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../data/figure/confusion/' + title + '_' + model_name + '.pdf')
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
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    now = datetime.now()
    plt.savefig(save_path)
    plt.clf()


def ConfusionMatrix(predict_vertex_finder, true_vertex_finder, model_name, classes):

    predict_vertex_finder = np.argmax(predict_vertex_finder, axis=1)
    true_vertex_finder = np.argmax(true_vertex_finder, axis=1)
    cmtmp = confusion_matrix(true_vertex_finder, predict_vertex_finder)
    cmeff = cmtmp.astype('float') / cmtmp.sum(axis=1)[:, np.newaxis]
    cmpur = cmtmp.astype('float') / cmtmp.sum(axis=0)[np.newaxis, :]
    print(cmtmp)
    plot_confusion_matrix(cmtmp, classes, model_name, Fontsize=5)
    plot_confusion_matrix(cmeff, classes, model_name, Fontsize=10, title="Confusion_Matrix_Efficiency")
    plot_confusion_matrix(cmpur, classes, model_name, Fontsize=10, title="Confusion_Matrix_Purity")


def EfficiencyCurve(pred, true, model_name):

    savw_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../data/numpy/roc/Efficiency_Curve_" + model_name + "_thre_allsig_allbg_sig_bg.npy")
    true = np.argmax(true, axis=1)
    true_pred = np.concatenate([true.reshape(-1, 1), pred], 1)

    classes = np.arange(len(pred[0]))
    cuts = np.arange(0.00, 1.01, 0.01)
    efficiency_curve = []
    for cla in tqdm(classes):
        class_efficiency_curve = []
        all_signal = len([datum for datum in true_pred if datum[0]==cla])
        all_background = len([datum for datum in true_pred if datum[0]!=cla])
        for cut in tqdm(cuts):
            signal = len([datum for datum in true_pred if datum[cla+1] > cut and datum[0]==cla])
            background = len([datum for datum in true_pred if datum[cla+1] > cut and datum[0]!=cla])
            class_efficiency_curve.append([cut, all_signal, all_background, signal, background])
        efficiency_curve.append(class_efficiency_curve)

    efficiency_curve = np.array(efficiency_curve, dtype=float)
    np.save(save_path, efficiency_curve, allow_pickle=True)


def PlotRegression(pred, true, model_name, MaxLog=3, Bins=1000):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    title = "Vertex Position - Radial Direction"
    xlabel = "Predicted Vertex Position - Radial Direction"
    ylabel = "True Vertex Position - Radial Direction"
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../data/figure/position/Vertex_Position_" + model_name + ".pdf")

    H = ax.hist2d(pred, true, bins=[np.logspace(-2, MaxLog, Bins), np.logspace(-2, MaxLog, Bins)], norm=colors.LogNorm())

    ax.set_aspect('equal')
    fig.colorbar(H[3], ax=ax)

    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.axes().set_aspect('equal')
    plt.savefig(save_path)


def LoadPairHistory(loss_name):
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../log/PairHistory/" + loss_name)
    if os.path.exists(path):
        with codecs.open(path, mode='rb') as f:
            history = pickle.load(f)

    return history

def LoadVLSTMHistory(loss_name):
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../log/VLSTMHistory/" + loss_name)
    if os.path.exists(path):
        with codecs.open(path, mode='rb') as f:
            history = pickle.load(f)

    return history

def DrawAttentionWeights(attention, model_name, sample, MaxTrack=60):
    import networkx as nx

    title = "Attention Weight Graph"
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../data/figure/attention/Attention_Weights_" + model_name + "_" + sample + ".pdf")

    G = nx.DiGraph()
    encoder_track_list = ["etr"+str(i) for i in range(0, MaxTrack)]
    decoder_track_list = ["dtr"+str(i) for i in range(0, MaxTrack)]
    G.add_nodes_from(encoder_track_list)
    G.add_nodes_from(decoder_track_list)

    pos = {}
    for i in range(60):
        for j in range(60):
            dtr = "dtr" + str(i)
            etr = "etr" + str(j)
            G.add_edge(etr, dtr, weight=attention[j, i])
        pos["etr"+str(i)] = (i, 3)
        pos["dtr"+str(i)] = (i, 1)

    edge_weight = [ d["weight"] for (u,v,d) in G.edges(data=True)]
    
    plt.title(title)
    plt.xlim(-1, MaxTrack+1) 
    plt.ylim(0, 4)
    
    nodes = nx.draw_networkx_nodes(G, pos=pos, node_size=20, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos=pos, node_size=20, edge_color=edge_weith, alpha=0.5, arrows=False, edge_cmap=plt.cm.Blues, width=2)

    for i in enumerate(G.number_of_edge()):
        edges[i].set_alpha(edge_weight[i])

    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig(save_path)


def DrawPairROCCurve(rocs, classes, model_name=""):
    for roc, cla in zip(rocs, classes):
        save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../data/figure/roc/Efficiency_Score_" + cla + "_" + model_name + ".pdf")
        fig = plt.figure()
        plt.rcParams["font.size"] = 15
        #thre_allsig_allbg_sig_bg
        plt.plot(roc[:, 0], roc[:, 3]/roc[:, 1], label=cla + " Efficiency")
        plt.plot(roc[:, 0], roc[:, 4]/roc[:, 2], label="Background Efficiency")
        plt.title(cla + " Efficiency Score")
        plt.xlabel(cla + " Score")
        plt.ylabel("Efficiency")
        plt.legend()
        plt.grid(color='b', linestyle='dotted', linewidth=1)
        fig.set_figheight(9)
        fig.set_figwidth(4)
        fig.subplots_adjust(left=0.2)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.clf()

        save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../data/figure/roc/ROC_Curve_" + cla + "_" + model_name + ".pdf")
        fig = plt.figure()
        #thre_allsig_allbg_sig_bg
        plt.plot(roc[:, 4]/roc[:, 2], roc[:, 3]/roc[:, 1], label="ROC Curve")
        plt.title(cla + " ROC Curve")
        plt.xlabel("Background Efficiency")
        plt.ylabel(cla + " Efficiency")
        plt.legend()
        plt.grid(color='b', linestyle='dotted', linewidth=1)
        fig.set_figheight(9)
        fig.set_figwidth(4)
        fig.subplots_adjust(left=0.2)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.clf()

