from keras.utils.generic_utils import get_custom_objects

import tensorflow as tf 
import numpy as np 
from tensorflow.python.keras import backend as K
from tensorflow import keras
from tensorflow.keras.layers import AbstractRNNCell, Concatenate, Dot
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, LSTMCell, _caching_device, _generate_zero_filled_state_for_cell
from tensorflow.keras.layers import Input, Dense, RNN, LSTM, Activation, TimeDistributed, Masking, BatchNormalization, Reshape, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import activations, constraints, initializers, regularizers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import constant_op, ops
from tensorflow.python.ops import array_ops, math_ops, clip_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.keras import backend_config
from tensorflow.python.training.tracking import data_structures

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

from Reccurent import RNN


def _config_for_enable_caching_device(rnn_cell):
  """Return the dict config for RNN cell wrt to enable_caching_device field.
  Since enable_caching_device is a internal implementation detail for speed up
  the RNN variable read when running on the multi remote worker setting, we
  don't want this config to be serialized constantly in the JSON. We will only
  serialize this field when a none default value is used to create the cell.
  Args:
    rnn_cell: the RNN cell for serialize.
  Returns:
    A dict which contains the JSON config for enable_caching_device value or
    empty dict if the enable_caching_device value is same as the default value.
  """
  default_enable_caching_device = ops.executing_eagerly_outside_functions()
  if rnn_cell._enable_caching_device != default_enable_caching_device:
    return {'enable_caching_device': rnn_cell._enable_caching_device}
  return {}



#####################################################################################################
#####################################################################################################

#class VLSTMCell(AbstractRNNCell):
class VLSTMCell(DropoutRNNCellMixin, Layer):
  # Cell class for the LSTM layer.

  def __init__(self,
               units,
               units_out,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               dense_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               implementation=1,
               **kwargs):

    if ops.executing_eagerly_outside_functions():
      self._enable_caching_device = kwargs.pop('enable_caching_device', True)
    else:
      self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    super(VLSTMCell, self).__init__(**kwargs)
    self.units = units
    self.units_out = units_out
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.dense_activation = activations.get(dense_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    if implementation != 1:
      logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
      self.implementation = 1
    else:
      self.implementation = implementation
    self.state_size = data_structures.NoDependency([self.units, self.units])
    self.output_size = 1
    #self.input_spec = InputSpec(ndim=3)

  """
  @property
  def state_size(self):
    #return [self.units, self.units_out]
    return [self.units, self.units]
  """

  @tf_utils.shape_type_conversion
  def build(self, input_shape): # difinition of the weights
    default_caching_device = _caching_device(self)
    input_dim = input_shape[-1]
    self.kernel = self.add_weight( # W
        shape=(input_dim, self.units * 4), # "* 4" means "o, f, i, z"
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)
    self.recurrent_kernel = self.add_weight( # R
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)
    self.dense_kernel = self.add_weight( # Last Dense kernel
        shape=(self.units * 1, self.units_out),
        name='dense_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)
    """
    self.dense_cn_kernel = self.add_weight( # Last Dense kernel
        shape=(self.units * 2, self.units),
        name='dense_cn_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)
        #caching_device=default_caching_device)
    """

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.Ones()((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 2,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight( # b
          shape=(self.units * 4,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          caching_device=default_caching_device)
    else:
      self.bias = None
    self.built = True

  def _compute_update_vertex(self, x, V_tm1):
    """Computes carry and output using split kernels."""
    # x = W * track
    x_i, x_f, x_c, x_o = x
    V_tm1_i, V_tm1_f, V_tm1_c, V_tm1_o, V_tm1_o2, V_tm1_u, V_tm1_v = V_tm1
    # i = x_i + V_tm1_i * R_i
    #   = W_i * track + V_tm1_1 * R_i
    
    i = self.recurrent_activation(
        x_i + K.dot(V_tm1_i, self.recurrent_kernel[:, :self.units]))
    f = self.recurrent_activation(
        x_f + K.dot(V_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
    c = self.activation(
        x_c + K.dot(V_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    
    # U = update vertex
    U = f * V_tm1_u + i * c

    o = self.recurrent_activation(
        x_o + K.dot(V_tm1_o, self.recurrent_kernel[:, self.units * 3:]))

    h_temp = o * self.activation(V_tm1_o2)
    #h_temp2 = self.dense_activation(K.dot(h_temp, self.dense_cn_kernel[:self.units, :]))
    #h_temp3 = self.dense_activation(K.dot(h_temp2, self.dense_cn_kernel[self.units:self.units * 2, :]))
    # h size [self.units]
    h = self.dense_activation(K.dot(h_temp, self.dense_kernel))
    # h size [1] activated sigmoid
    
    V = h * U + (1-h) * V_tm1_v
    return h, V

  def call(self, inputs, states, training=None):
    V_tm1 = states[0] # previous Vertex state

    if self.implementation == 1:
      # input = track
      inputs_i = inputs
      inputs_f = inputs
      inputs_c = inputs
      inputs_o = inputs
      # k = W
      k_i, k_f, k_c, k_o = array_ops.split(
              self.kernel, num_or_size_splits=4, axis=1)
      x_i = K.dot(inputs_i, k_i)
      x_f = K.dot(inputs_f, k_f)
      x_c = K.dot(inputs_c, k_c)
      x_o = K.dot(inputs_o, k_o)
      if self.use_bias:
        b_i, b_f, b_c, b_o = array_ops.split(
            self.bias, num_or_size_splits=4, axis=0)
        x_i = K.bias_add(x_i, b_i)
        x_f = K.bias_add(x_f, b_f)
        x_c = K.bias_add(x_c, b_c)
        x_o = K.bias_add(x_o, b_o)

      V_tm1_i = V_tm1
      V_tm1_f = V_tm1
      V_tm1_c = V_tm1
      V_tm1_o = V_tm1
      V_tm1_o2 = V_tm1
      V_tm1_u = V_tm1
      V_tm1_v = V_tm1
      x = (x_i, x_f, x_c, x_o)

      V_tm1 = (V_tm1_i, V_tm1_f, V_tm1_c, V_tm1_o, V_tm1_o2, V_tm1_u, V_tm1_v)
      h, V = self._compute_update_vertex(x, V_tm1)

    return h, [V, V]

  def get_config(self):
    config = {
        'units':
            self.units,
        'units_out':
            self.units_out,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'implementation':
            self.implementation
        #'input_spec' :
        #    self.input_spec
    }
    base_config = super(VLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    V = tf.zeros([batch_size, self.units], dtype)
    return [V, V]

"""
  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(
        self, inputs, batch_size, dtype))

def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
  if inputs is not None:
    batch_size = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
  return _generate_zero_filled_state(batch_size, cell.state_size, dtype)

def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
  #Generate a zero filled tensor with shape [batch_size, state_size].
  if batch_size_tensor is None or dtype is None:
    raise ValueError(
        'batch_size and dtype cannot be None while constructing initial state: '
        'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

  def create_zeros(unnested_state_size):
    flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
    init_state_size = [batch_size_tensor] + flat_dims
    return array_ops.zeros(init_state_size, dtype=dtype)

  if nest.is_sequence(state_size):
    return nest.map_structure(create_zeros, state_size)
  else:
    return create_zeros(state_size)
"""


def _config_for_enable_caching_device(rnn_cell):
  """Return the dict config for RNN cell wrt to enable_caching_device field.
  Since enable_caching_device is a internal implementation detail for speed up
  the RNN variable read when running on the multi remote worker setting, we
  don't want this config to be serialized constantly in the JSON. We will only
  serialize this field when a none default value is used to create the cell.
  Args:
    rnn_cell: the RNN cell for serialize.
  Returns:
    A dict which contains the JSON config for enable_caching_device value or
    empty dict if the enable_caching_device value is same as the default value.
  """
  default_enable_caching_device = ops.executing_eagerly_outside_functions()
  if rnn_cell._enable_caching_device != default_enable_caching_device:
    return {'enable_caching_device': rnn_cell._enable_caching_device}
  return {}

if __name__ == "__main__":

    BATCH_SIZE = 32
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    INPUT_DIM = 23
    NODE_SIZE = 32
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)

    name_teach_track_pairs = "/home/goto/ILC/Deep_Learning/data/tmp_vfdnn_06_pairs.npy"
    name_teach_tracks = "/home/goto/ILC/Deep_Learning/data/tmp_vfdnn_06_tracks.npy"
    name_targets = "/home/goto/ILC/Deep_Learning/data/tmp_vfdnn_06_targets.npy"
    teach_track_pairs = np.load(name_teach_track_pairs, allow_pickle=True)
    teach_tracks = np.load(name_teach_tracks, allow_pickle=True)
    targets = np.load(name_targets, allow_pickle=True)
    labels = targets[:, :, 0]

    print(teach_track_pairs.shape)
    print(teach_tracks.shape)
    print(targets.shape)

    pair_input = Input(shape=(44,), name='Pair_Input')
    track_input = Input(shape=(None, INPUT_DIM), name='Track_Input')
    x = TimeDistributed(Dense(NODE_SIZE, name='Embedding_Dense'))(track_input)

    init_state = Dense(NODE_SIZE, name='Dense_1')(pair_input)
    init_state = BatchNormalization(name='BatchNorm_1')(init_state)
    init_state = Activation('relu', name='Activation_1')(init_state)

    cell = VLSTMCell(NODE_SIZE, 1)
    #cell = LSTMCell(32)

    x = RNN(cell, return_sequences=True, name='VLSTM',)(x, initial_state=[init_state, init_state])
    #x = LSTM(32, return_sequences=True, name='VLSTM',)(x, initial_state=[init_state, init_state])
    #x = RNN(cell, return_sequences=True, name='VLSTM',)(x)
    #x = TimeDistributed(Dense(1, name='LAST_Dense'))(x)

    model = Model(inputs=[pair_input, track_input], outputs=x)

    model.summary()

    model.compile(loss="binary_crossentropy",
                  optimizer=OPTIMIZER,
                  metrics=["accuracy"])


    history = model.fit([teach_track_pairs[:10000], teach_tracks[:10000, :, :23]], labels[:10000],
                        batch_size=BATCH_SIZE,
                        epochs=1,
                        verbose=VERBOSE,
                        validation_split=VALIDATION_SPLIT)

    tf.saved_model.save(model, "savetest_as_pb")




