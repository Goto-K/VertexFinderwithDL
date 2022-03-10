import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard, Callback
from time import gmtime, strftime
import os

from . import loss


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")


def PairModelTraining(model, model_name, x_train, vertex_train, position_train, BATCH_SIZE=1024, NB_EPOCHS=2500, VALIDATION_SPLIT=0.2, LR=0.001,
                      Custom_Weights=[0.0090, 0.0175, 0.3375, 0.1800, 0.3509, 0.1260, 1.0], Loss_Weights=[0.5, 0.5]):
    # Tensor Board
    set_dir_name = 'PairTensorBoard'
    set_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../log/" + set_dir_name)
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_time = tictoc
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../log/" + set_dir_name + "/" + model_name + directory_time)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with CustomObjectScope({'custom_categorical_crossentropy': loss.custom_categorical_crossentropy(Custom_Weights)}):
        model.compile(loss={'Vertex_Output': 'custom_categorical_crossentropy', 'Position_Output': 'mean_squared_logarithmic_error'},
                      loss_weights={'Vertex_Output':Loss_Weights[0], 'Position_Output':Loss_Weights[1]},
                      optimizer=SGD(learning_rate=LR),
                      metrics=['accuracy', 'mae'])

    callbacks = [TensorBoard(log_dir=log_dir)]

    history = model.fit(x_train, [vertex_train, position_train],
                        batch_size=BATCH_SIZE,
                        epochs=NB_EPOCHS,
                        callbacks=callbacks,
                        verbose=1,
                        validation_split=VALIDATION_SPLIT)

    return model, history


def PairModelCTraining(model, model_name, x_train, vertex_train, position_train, BATCH_SIZE=1024, NB_EPOCHS=2500, VALIDATION_SPLIT=0.2, LR=0.001,
                       Custom_Weights=[0.0090, 0.0175, 0.3375, 0.1800, 0.3509, 0.1260, 1.0]):
    # Tensor Board
    set_dir_name = 'PairTensorBoard'
    set_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../log/" + set_dir_name)
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_time = tictoc
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../log/" + set_dir_name + "/" + model_name + directory_time)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with CustomObjectScope({'custom_categorical_crossentropy': loss.custom_categorical_crossentropy(Custom_Weights)}):
        model.compile(loss={'Vertex_Output': 'custom_categorical_crossentropy'},
                      optimizer=SGD(learning_rate=LR),
                      metrics=['accuracy'])

    callbacks = [TensorBoard(log_dir=log_dir)]

    history = model.fit([x_train, position_train], vertex_train,
                        batch_size=BATCH_SIZE,
                        epochs=NB_EPOCHS,
                        callbacks=callbacks,
                        verbose=1,
                        validation_split=VALIDATION_SPLIT)

    return model, history

