from keras.utils.generic_utils import get_custom_objects

import tensorflow as tf 
import numpy as np 
from tensorflow.keras.layers import RNN, AbstractRNNCell
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras import activations, constraints, initializers, regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import constant_op, ops
from tensorflow.python.ops import array_ops, math_ops, clip_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.keras import backend_config



#####################################################################################################
#####################################################################################################

class VLSTMCell(AbstractRNNCell):
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

  @property
  def state_size(self):
    #return [self.units, self.units_out]
    return [self.units, self.units]

  #@tf_utils.shape_type_conversion
  def build(self, input_shape): # difinition of the weights
    #default_caching_device = _caching_device(self)
    input_dim = input_shape[-1]
    self.kernel = self.add_weight( # W
        shape=(input_dim, self.units * 4), # "* 4" means "o, f, i, z"
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
        #caching_device=default_caching_device)
    self.recurrent_kernel = self.add_weight( # R
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)
        #caching_device=default_caching_device)
    self.dense_kernel = self.add_weight( # Last Dense kernel
        shape=(self.units * 1, self.units_out),
        name='dense_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)
        #caching_device=default_caching_device)

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
          constraint=self.bias_constraint)
          #caching_device=default_caching_device)
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
    }
    base_config = super(VLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=True, batch_size=True, dtype=None):
    return list(_generate_zero_filled_state_for_cell(
        self, inputs, batch_size, dtype))


#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

def _backtrack_identity(tensor):
  while tensor.op.type == 'Identity':
    tensor = tensor.op.inputs[0]
  return tensor

def _constant_to_tensor(x, dtype):
  """Convert the input `x` to a tensor of type `dtype`.
  This is slightly faster than the _to_tensor function, at the cost of
  handling fewer cases.
  Arguments:
      x: An object to be converted (numpy arrays, floats, ints and lists of
        them).
      dtype: The destination type.
  Returns:
      A tensor.
  """
  return constant_op.constant(x, dtype=dtype)

epsilon = backend_config.epsilon


def binary_crossentropy(pair_reinforce=False, from_logits=False): 
  """Binary crossentropy between an output tensor and a target tensor.
  Arguments:
      target: A tensor with the same shape as `output`.
      output: A tensor.
      from_logits: Whether `output` is expected to be a logits tensor.
          By default, we consider that `output`
          encodes a probability distribution.
  Returns:
      A tensor.
  """

  def binary_crossentropy_loss(yX_train, output):
    #nb_classes = output.shape[1]
    target = yX_train[:, :, 0]
    target = tf.expand_dims(target, -1)
    zero_padding = yX_train[:, :, 1]
    zero_padding = tf.expand_dims(zero_padding, -1)
    pair1 = yX_train[:, :, 2]
    pair1 = tf.expand_dims(pair1, -1)
    pair2 = yX_train[:, :, 3]
    pair2 = tf.expand_dims(pair2, -1)
    track = yX_train[:, :, 4]
    track = tf.expand_dims(track, -1)

    if from_logits:
      return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

    if not isinstance(output, (ops.EagerTensor, variables_module.Variable)):
      output = _backtrack_identity(output)
      if output.op.type == 'Sigmoid':
        # When sigmoid activation function is used for output operation, we
        # use logits from the sigmoid function directly to compute loss in order
        # to prevent collapsing zero when training.
        assert len(output.op.inputs) == 1
        output = output.op.inputs[0]
        return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

    epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)

    target_zp = tf.boolean_mask(target, zero_padding==1)
    output_zp = tf.boolean_mask(output, zero_padding==1)
    pair1_zp = tf.boolean_mask(pair1, zero_padding==1)
    pair2_zp = tf.boolean_mask(pair2, zero_padding==1)
    track_zp = tf.boolean_mask(track, zero_padding==1)

    target_zp_con = tf.boolean_mask(target_zp, target_zp==1)
    output_zp_con = tf.boolean_mask(output_zp, target_zp==1)
    pair1_zp_con = tf.boolean_mask(pair1_zp, target_zp==1)
    pair2_zp_con = tf.boolean_mask(pair2_zp, target_zp==1)
    track_zp_con = tf.boolean_mask(track_zp, target_zp==1)

    target_zp_con_wop1 = tf.boolean_mask(target_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
    output_zp_con_wop1 = tf.boolean_mask(output_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
    pair2_zp_con_wop1 = tf.boolean_mask(pair2_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
    track_zp_con_wop1 = tf.boolean_mask(track_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))

    target_zp_con_wop = tf.boolean_mask(target_zp_con_wop1, tf.math.not_equal(pair2_zp_con_wop1, track_zp_con_wop1))
    output_zp_con_wop = tf.boolean_mask(output_zp_con_wop1, tf.math.not_equal(pair2_zp_con_wop1, track_zp_con_wop1))


    # Compute cross entropy from probabilities.

    bce = zero_padding * target * math_ops.log(output + epsilon())
    bce += zero_padding * (1 - target) * math_ops.log(1 - output + epsilon())

    if pair_reinforce:
        bce += target_zp_con_wop * math_ops.log(output_zp_con_wop + epsilon())

    return -bce

  return binary_crossentropy_loss


#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

def accuracy_all(y_true, y_pred, threshold=0.5):
  """Calculates how often predictions matches binary labels.
  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    threshold: (Optional) Float representing the threshold for deciding whether
      prediction values are 1 or 0.
  Returns:
    Binary accuracy values. shape = `[batch_size, d0, .. dN-1]`
  """

  nb_classes = y_pred.shape[1]
  target = y_true[:, :, 0]
  target = tf.expand_dims(target, -1)
  zero_padding = y_true[:, :, 1]
  zero_padding = tf.expand_dims(zero_padding, -1)
  """
  if not isinstance(output, (ops.EagerTensor, variables_module.Variable)):
    output = _backtrack_identity(output)
    if output.op.type == 'Sigmoid':
      # When sigmoid activation function is used for output operation, we
      # use logits from the sigmoid function directly to compute loss in order
      # to prevent collapsing zero when training.
      assert len(output.op.inputs) == 1
      output = output.op.inputs[0]
      return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

  epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
  output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
  """
  threshold = math_ops.cast(threshold, y_pred.dtype)
  output = math_ops.cast(y_pred > threshold, y_pred.dtype)

  target_zp = tf.boolean_mask(target, zero_padding==1)
  output_zp = tf.boolean_mask(output, zero_padding==1)
  #score = tf.boolean_mask(zero_padding, math_ops.equal(target, output))

  return K.mean(math_ops.equal(target_zp, output_zp), axis=-1)


def accuracy(y_true, y_pred, threshold=0.5):

  nb_classes = y_pred.shape[1]
  target = y_true[:, :, 0]
  target = tf.expand_dims(target, -1)
  zero_padding = y_true[:, :, 1]
  zero_padding = tf.expand_dims(zero_padding, -1)
  pair1 = y_true[:, :, 2]
  pair1 = tf.expand_dims(pair1, -1)
  pair2 = y_true[:, :, 3]
  pair2 = tf.expand_dims(pair2, -1)
  track = y_true[:, :, 4]
  track = tf.expand_dims(track, -1)

  threshold = math_ops.cast(threshold, y_pred.dtype)
  output = math_ops.cast(y_pred > threshold, y_pred.dtype)

  target_zp = tf.boolean_mask(target, zero_padding==1)
  output_zp = tf.boolean_mask(output, zero_padding==1)
  pair1_zp = tf.boolean_mask(pair1, zero_padding==1)
  pair2_zp = tf.boolean_mask(pair2, zero_padding==1)
  track_zp = tf.boolean_mask(track, zero_padding==1)

  target_zp_wop1 = tf.boolean_mask(target_zp, tf.math.not_equal(pair1_zp, track_zp))
  output_zp_wop1 = tf.boolean_mask(output_zp, tf.math.not_equal(pair1_zp, track_zp))
  pair2_zp_wop1 = tf.boolean_mask(pair2_zp, tf.math.not_equal(pair1_zp, track_zp))
  track_zp_wop1 = tf.boolean_mask(track_zp, tf.math.not_equal(pair1_zp, track_zp))

  target_zp_wop = tf.boolean_mask(target_zp_wop1, tf.math.not_equal(pair2_zp_wop1, track_zp_wop1))
  output_zp_wop = tf.boolean_mask(output_zp_wop1, tf.math.not_equal(pair2_zp_wop1, track_zp_wop1))

  return K.mean(math_ops.equal(target_zp_wop, output_zp_wop), axis=-1)


def true_positive(y_true, y_pred, threshold=0.5):

  nb_classes = y_pred.shape[1]
  target = y_true[:, :, 0]
  target = tf.expand_dims(target, -1)
  zero_padding = y_true[:, :, 1]
  zero_padding = tf.expand_dims(zero_padding, -1)
  pair1 = y_true[:, :, 2]
  pair1 = tf.expand_dims(pair1, -1)
  pair2 = y_true[:, :, 3]
  pair2 = tf.expand_dims(pair2, -1)
  track = y_true[:, :, 4]
  track = tf.expand_dims(track, -1)

  threshold = math_ops.cast(threshold, y_pred.dtype)
  output = math_ops.cast(y_pred > threshold, y_pred.dtype)

  target_zp = tf.boolean_mask(target, zero_padding==1)
  output_zp = tf.boolean_mask(output, zero_padding==1)
  pair1_zp = tf.boolean_mask(pair1, zero_padding==1)
  pair2_zp = tf.boolean_mask(pair2, zero_padding==1)
  track_zp = tf.boolean_mask(track, zero_padding==1)

  target_zp_con = tf.boolean_mask(target_zp, target_zp==1)
  output_zp_con = tf.boolean_mask(output_zp, target_zp==1)
  pair1_zp_con = tf.boolean_mask(pair1_zp, target_zp==1)
  pair2_zp_con = tf.boolean_mask(pair2_zp, target_zp==1)
  track_zp_con = tf.boolean_mask(track_zp, target_zp==1)

  target_zp_con_wop1 = tf.boolean_mask(target_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
  output_zp_con_wop1 = tf.boolean_mask(output_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
  pair2_zp_con_wop1 = tf.boolean_mask(pair2_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
  track_zp_con_wop1 = tf.boolean_mask(track_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))

  target_zp_con_wop = tf.boolean_mask(target_zp_con_wop1, tf.math.not_equal(pair2_zp_con_wop1, track_zp_con_wop1))
  output_zp_con_wop = tf.boolean_mask(output_zp_con_wop1, tf.math.not_equal(pair2_zp_con_wop1, track_zp_con_wop1))

  return K.mean(math_ops.equal(target_zp_con_wop, output_zp_con_wop), axis=-1)


def false_negative(y_true, y_pred, threshold=0.5):

  nb_classes = y_pred.shape[1]
  target = y_true[:, :, 0]
  target = tf.expand_dims(target, -1)
  zero_padding = y_true[:, :, 1]
  zero_padding = tf.expand_dims(zero_padding, -1)
  pair1 = y_true[:, :, 2]
  pair1 = tf.expand_dims(pair1, -1)
  pair2 = y_true[:, :, 3]
  pair2 = tf.expand_dims(pair2, -1)
  track = y_true[:, :, 4]
  track = tf.expand_dims(track, -1)

  threshold = math_ops.cast(threshold, y_pred.dtype)
  output = math_ops.cast(y_pred > threshold, y_pred.dtype)

  target_zp = tf.boolean_mask(target, zero_padding==1)
  output_zp = tf.boolean_mask(output, zero_padding==1)
  pair1_zp = tf.boolean_mask(pair1, zero_padding==1)
  pair2_zp = tf.boolean_mask(pair2, zero_padding==1)
  track_zp = tf.boolean_mask(track, zero_padding==1)

  target_zp_con = tf.boolean_mask(target_zp, target_zp==1)
  output_zp_con = tf.boolean_mask(output_zp, target_zp==1)
  pair1_zp_con = tf.boolean_mask(pair1_zp, target_zp==1)
  pair2_zp_con = tf.boolean_mask(pair2_zp, target_zp==1)
  track_zp_con = tf.boolean_mask(track_zp, target_zp==1)

  target_zp_con_wop1 = tf.boolean_mask(target_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
  output_zp_con_wop1 = tf.boolean_mask(output_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
  pair2_zp_con_wop1 = tf.boolean_mask(pair2_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
  track_zp_con_wop1 = tf.boolean_mask(track_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))

  target_zp_con_wop = tf.boolean_mask(target_zp_con_wop1, tf.math.not_equal(pair2_zp_con_wop1, track_zp_con_wop1))
  output_zp_con_wop = tf.boolean_mask(output_zp_con_wop1, tf.math.not_equal(pair2_zp_con_wop1, track_zp_con_wop1))

  return K.mean(math_ops.not_equal(target_zp_con_wop, output_zp_con_wop), axis=-1)


def false_positive(y_true, y_pred, threshold=0.5):

  nb_classes = y_pred.shape[1]
  target = y_true[:, :, 0]
  target = tf.expand_dims(target, -1)
  zero_padding = y_true[:, :, 1]
  zero_padding = tf.expand_dims(zero_padding, -1)
  pair1 = y_true[:, :, 2]
  pair1 = tf.expand_dims(pair1, -1)
  pair2 = y_true[:, :, 3]
  pair2 = tf.expand_dims(pair2, -1)
  track = y_true[:, :, 4]
  track = tf.expand_dims(track, -1)

  threshold = math_ops.cast(threshold, y_pred.dtype)
  output = math_ops.cast(y_pred > threshold, y_pred.dtype)

  target_zp = tf.boolean_mask(target, zero_padding==1)
  output_zp = tf.boolean_mask(output, zero_padding==1)
  pair1_zp = tf.boolean_mask(pair1, zero_padding==1)
  pair2_zp = tf.boolean_mask(pair2, zero_padding==1)
  track_zp = tf.boolean_mask(track, zero_padding==1)

  target_zp_con = tf.boolean_mask(target_zp, target_zp==0)
  output_zp_con = tf.boolean_mask(output_zp, target_zp==0)
  pair1_zp_con = tf.boolean_mask(pair1_zp, target_zp==0)
  pair2_zp_con = tf.boolean_mask(pair2_zp, target_zp==0)
  track_zp_con = tf.boolean_mask(track_zp, target_zp==0)

  target_zp_con_wop1 = tf.boolean_mask(target_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
  output_zp_con_wop1 = tf.boolean_mask(output_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
  pair2_zp_con_wop1 = tf.boolean_mask(pair2_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
  track_zp_con_wop1 = tf.boolean_mask(track_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))

  target_zp_con_wop = tf.boolean_mask(target_zp_con_wop1, tf.math.not_equal(pair2_zp_con_wop1, track_zp_con_wop1))
  output_zp_con_wop = tf.boolean_mask(output_zp_con_wop1, tf.math.not_equal(pair2_zp_con_wop1, track_zp_con_wop1))

  return K.mean(math_ops.not_equal(target_zp_con_wop, output_zp_con_wop), axis=-1)


def true_negative(y_true, y_pred, threshold=0.5):

  nb_classes = y_pred.shape[1]
  target = y_true[:, :, 0]
  target = tf.expand_dims(target, -1)
  zero_padding = y_true[:, :, 1]
  zero_padding = tf.expand_dims(zero_padding, -1)
  pair1 = y_true[:, :, 2]
  pair1 = tf.expand_dims(pair1, -1)
  pair2 = y_true[:, :, 3]
  pair2 = tf.expand_dims(pair2, -1)
  track = y_true[:, :, 4]
  track = tf.expand_dims(track, -1)

  threshold = math_ops.cast(threshold, y_pred.dtype)
  output = math_ops.cast(y_pred > threshold, y_pred.dtype)

  target_zp = tf.boolean_mask(target, zero_padding==1)
  output_zp = tf.boolean_mask(output, zero_padding==1)
  pair1_zp = tf.boolean_mask(pair1, zero_padding==1)
  pair2_zp = tf.boolean_mask(pair2, zero_padding==1)
  track_zp = tf.boolean_mask(track, zero_padding==1)

  target_zp_con = tf.boolean_mask(target_zp, target_zp==0)
  output_zp_con = tf.boolean_mask(output_zp, target_zp==0)
  pair1_zp_con = tf.boolean_mask(pair1_zp, target_zp==0)
  pair2_zp_con = tf.boolean_mask(pair2_zp, target_zp==0)
  track_zp_con = tf.boolean_mask(track_zp, target_zp==0)

  target_zp_con_wop1 = tf.boolean_mask(target_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
  output_zp_con_wop1 = tf.boolean_mask(output_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
  pair2_zp_con_wop1 = tf.boolean_mask(pair2_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))
  track_zp_con_wop1 = tf.boolean_mask(track_zp_con, tf.math.not_equal(pair1_zp_con, track_zp_con))

  target_zp_con_wop = tf.boolean_mask(target_zp_con_wop1, tf.math.not_equal(pair2_zp_con_wop1, track_zp_con_wop1))
  output_zp_con_wop = tf.boolean_mask(output_zp_con_wop1, tf.math.not_equal(pair2_zp_con_wop1, track_zp_con_wop1))

  return K.mean(math_ops.equal(target_zp_con_wop, output_zp_con_wop), axis=-1)

