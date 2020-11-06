import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, RNN, Activation, TimeDistributed, Masking, BatchNormalization
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow import keras


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

from time import gmtime, strftime
import os
import pathlib
import CUSTOM
import ATTENTION
import MODELTOOLS
tracks = np.zeros([10000, 54, 33]) 
ENCODER_INPUT=256
DECODER_UNITS=256
NB_SAMPLES=50000
pair_reinforce=False
BATCH_SIZE = 32
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIMIZER = keras.optimizers.Adam(lr=0.001)
MAX_TRACK_NUM = None
INPUT_DIM = tracks.shape[2]
DECODER_OUTPUT = 1

set_dir_name='Test_Attention_VLSTM_MODEL'
set_dir = "/home/goto/ILC/Deep_Learning/log/" + set_dir_name
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
class RepeatVector4D(Layer):

    def __init__(self, n, **kwargs):
        self.n = n
        self.input_spec = [InputSpec(ndim=3)]
        super(RepeatVector4D, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.n, input_shape[1], input_shape[2])

    def call(self, x, mask=None):
        x = K.expand_dims(x, 1)
        pattern = K.stack([1, self.n, 1, 1])
        return K.tile(x, pattern)

def squeeze_rear2axes_operator( x4d ) :
    shape = tf.shape( x4d ) # get dynamic tensor shape
    x3d = tf.reshape( x4d, [shape[0], shape[1], shape[2] * shape[3]] )
    return x3d

def squeeze_rear2axes_shape( x4d_shape ) :
    in_batch, in_tracknum1, in_tracknum2, in_attdim = x4d_shape
    if ( None in [ in_tracknum1, in_tracknum2 ] ) :
        output_shape = ( in_batch, None, None )
    else :
        output_shape = ( in_batch, in_tracknum1, in_tracknum2 * in_attdim )
    return output_shape


pair_input = Input(shape=(44,))
track_input = Input(shape=(None, INPUT_DIM))

_, MAX_TRACK_NUM, _INPUT_DIM = track_input.get_shape().as_list()

encoder_init_state = Dense(ENCODER_INPUT)(pair_input)
ecnoder_init_state = BatchNormalization()(encoder_init_state)
encoder_init_state = Activation('relu')(encoder_init_state)
encoder_init_state = Dense(ENCODER_INPUT)(encoder_init_state)
encoder_init_state = BatchNormalization()(ecnoder_init_state)
encoder_init_state = Activation('relu')(encoder_init_state)

decoder_init_state = Dense(DECODER_UNITS)(pair_input)
denoder_init_state = BatchNormalization()(decoder_init_state)
decoder_init_state = Activation('relu')(decoder_init_state)
decoder_init_state = Dense(DECODER_UNITS)(decoder_init_state)
decoder_init_state = BatchNormalization()(denoder_init_state)
decoder_init_state = Activation('relu')(decoder_init_state)

vlstm_cell = CUSTOM.VLSTMCell_M(ENCODER_INPUT)
encoder = RNN(vlstm_cell, return_sequences=True, name='VLSTM_encoder',)(track_input, initial_state=[encoder_init_state, encoder_init_state])
encoder_repeat = RepeatVector4D(MAX_TRACK_NUM)(encoder)
encoder_reshape = tf.keras.layers.Lambda( squeeze_rear2axes_operator, output_shape = squeeze_rear2axes_shape )(encoder_repeat)
#encoder_reshape = tf.keras.backend.reshape(encoder_repeat, (None, MAX_TRACK_NUM, MAX_TRACK_NUM*ENCODER_INPUT))
tra_att = tf.keras.layers.concatenate([track_input, encoder_reshape], axis=-1)

attentionvlstm_cell = CUSTOM.AttentionVLSTMCell_v2(DECODER_UNITS, ENCODER_INPUT, DECODER_OUTPUT, MAX_TRACK_NUM, INPUT_DIM)
decoder = RNN(attentionvlstm_cell, return_sequences=True, name='Attention_VLSTM_decoder')(tra_att, initial_state=[decoder_init_state, decoder_init_state])
model = Model(inputs=[pair_input, track_input], outputs=decoder)
