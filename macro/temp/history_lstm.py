import tensorflow as tf 
import numpy as np 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import RNN, AbstractRNNCell
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras import activations, constraints, initializers, regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops

class SLSTMCell(AbstractRNNCell):
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               **kwargs):

    super(SLSTMCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

  @property
  def state_size(self):
    return [self.units, self.units]

  def build(self, input_shape):
    input_dim = input_shape[-1]
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    if self.use_bias:
      self.bias = self.add_weight(
          shape=(self.units * 4,),
          name='bias',
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, states, training=None):
    print(self.kernel)
    V_tm1 = states[0]  # previous memory state

    k_i, k_f, k_c, k_o = array_ops.split(
          self.kernel, num_or_size_splits=4, axis=1)
    x_i = K.dot(inputs, k_i)
    x_f = K.dot(inputs, k_f)
    x_c = K.dot(inputs, k_c)
    x_o = K.dot(inputs, k_o)
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

    h = o * self.activation(V_tm1_o2)
    V = h * U + (1-h) * V_tm1_v
    
    return(h, [V, h])


tf.random.set_seed(111)
np.random.seed(111)

model = Sequential([
    RNN(SLSTMCell(1, activation=None), input_shape=(None, 1), return_sequences=True)
])
model.compile(optimizer=SGD(lr=0.0001), loss="mean_squared_error")
%history -f history_lstm.py
