import MODELTOOLS
import MODELBANK
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.callbacks import TensorBoard, Callback
from time import gmtime, strftime
import os

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")



if __name__ == "__main__":
    NB_EPOCHS = 20
    BATCH_SIZE = 4096
    VERBOSE = 1
    NB_CLASSES = 5
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)
    set_dir_name='TEST_MODEL'
    set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    model_name = "Test_Comparison_Model_lowF_highT_noconvF"
    save_model_name = "Test_Comparison_Model_lowF_highT_noconvF_re_vfdnn02"

    image_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_02_track_image.npy"
    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_02_shaped.npy"

    images = np.load(image_name)
    data = np.load(variable_name)

    variables = data[:, 3:57] # high // only conv
    label_teacher = np_utils.to_categorical(data[:, 57], 5)

    print("file load !")

    model = MODELBANK.load_model(model_name)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=OPTIMIZER, 
                  metrics=['accuracy'])

    # Tensor Board
    tictoc = strftime("%Y%m%d%H%M", gmtime())
    directory_name = tictoc
    log_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name + "/" + set_dir_name + '_' + directory_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    callbacks = [TensorBoard(log_dir=log_dir)]

    history = model.fit([images, variables], label_teacher,
                        batch_size=BATCH_SIZE, 
                        epochs=NB_EPOCHS, 
                        callbacks=callbacks,
                        verbose=VERBOSE, 
                        validation_split=VALIDATION_SPLIT)

    MODELTOOLS.save_model(model, save_model_name)
    MODELTOOLS.save_history(history, save_model_name, chi=False)

