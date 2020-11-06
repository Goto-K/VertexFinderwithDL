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


data = np.load("data/test/all_test_complete.npy")

predict_vertex_finder = []
true_vertex_finder = []

for datum in data:
    if datum[58]==0:
        predict_vertex_finder.append([1, 0, 0])
    elif datum[58]==1:
        predict_vertex_finder.append([0, 0, 1])
    else:
        predict_vertex_finder.append([0, 1, 0])

for datum in data:
    if datum[57]==0:
        true_vertex_finder.append([1, 0, 0])
    elif datum[57]==1:
        true_vertex_finder.append([0, 1, 0])
    else:
        true_vertex_finder.append([0, 0, 1])

classes = ["not connected", "primary", "secondary"]

ana_confusion_matrix(predict_vertex_finder, true_vertex_finder, "Lcfiplus_Tag", classes, Norm=True)
ana_confusion_matrix(predict_vertex_finder, true_vertex_finder, "Lcfiplus_Tag", classes, Norm=False)
