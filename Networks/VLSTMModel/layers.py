import tensorflow as tf 
from tensorflow.keras.layers import AbstractRNNCell
from tensorflow.python.keras import activations, constraints, initializers, regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops


# Vertex LSTM Cell
# VLSTMCellSimple : Standard Cell
# VLSTMCellEncoder : Encoder part for Attention VLSTM
# AttentionVLSTMCell : Decoder part for Attention VLSTM
class VLSTMCellSimple(AbstractRNNCell):

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

    super(VLSTMCellSimple, self).__init__(**kwargs)
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
    return [self.units, self.units]

  #@tf_utils.shape_type_conversion
  def build(self, input_shape): # difinition of the weights
    input_dim = input_shape[-1]
    self.kernel = self.add_weight( # W
        shape=(input_dim, self.units * 4), # "* 4" means "o, f, i, z"
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight( # R
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)
    self.dense_kernel = self.add_weight( # Last Dense kernel
        shape=(self.units * 1, self.units_out),
        name='dense_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

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
    base_config = super(VLSTMCellSimple, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#####################################################################################################
#####################################################################################################

class VLSTMCellEncoder(AbstractRNNCell):
  # Cell class for the LSTM layer.

  def __init__(self,
               units,
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

    super(VLSTMCellEncoder, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
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
    return [self.units, self.units]

  #@tf_utils.shape_type_conversion
  def build(self, input_shape): # difinition of the weights
    input_dim = input_shape[-1]
    self.batch_size = input_shape[0]
    self.kernel = self.add_weight( # W
        shape=(input_dim, self.units * 4), # "* 4" means "o, f, i, z"
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight( # R
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

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
    else:
      self.bias = None
    self.built = True

  def _compute_update_vertex(self, x, V_tm1):
    """Computes carry and output using split kernels."""
    x_i, x_f, x_c, x_o = x
    V_tm1_i, V_tm1_f, V_tm1_c, V_tm1_o, V_tm1_o2, V_tm1_u, V_tm1_v = V_tm1

    i = self.recurrent_activation(
        x_i + K.dot(V_tm1_i, self.recurrent_kernel[:, :self.units]))
    f = self.recurrent_activation(
        x_f + K.dot(V_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
    c = self.activation(
        x_c + K.dot(V_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))

    U = f * V_tm1_u + i * c

    o = self.recurrent_activation(
        x_o + K.dot(V_tm1_o, self.recurrent_kernel[:, self.units * 3:]))

    h = o * self.activation(V_tm1_o2)

    V = h * U + (1-h) * V_tm1_v
    #the size of h is [self.units]
    return h, V

  def call(self, inputs, states, training=None):
    V_tm1 = states[0] # previous Vertex state


    if self.implementation == 1:
      inputs_i = inputs
      inputs_f = inputs
      inputs_c = inputs
      inputs_o = inputs
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
    base_config = super(VLSTMCellEncoder, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#####################################################################################################
#####################################################################################################

class AttentionVLSTMCell(AbstractRNNCell):

  def __init__(self,
               units,
               att_input_dim,
               output_dim,
               timestep,
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

    super(AttentionVLSTMCell, self).__init__(**kwargs)
    self.units = units
    self.att_input_dim = att_input_dim
    self.output_dim = output_dim
    self.timestep = timestep
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
    return [self.timestep*self.att_input_dim, self.units]

  def build(self, input_shape): # difinition of the weights
    self.feature_dim = input_shape[-1]
    self.batch_size = input_shape[0]

    """
      attention kernel weight V(self.units,), W(self.units, self.units),
                              U(self.input_dim, self.units), b(self.units,)
    """
    self.attention_kernel_V = self.add_weight(
        shape=(self.units,),
        name='attention_kernel_V',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)

    self.attention_kernel_W = self.add_weight(
        shape=(self.feature_dim, self.units),
        name='attention_kernel_W',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)

    self.attention_kernel_U = self.add_weight(
        shape=(self.att_input_dim, self.units),
        name='attention_kernel_U',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)

    self.attention_kernel_b = self.add_weight(
        shape=(self.units,),
        name='attention_kernel_b',
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint)

    self.kernel = self.add_weight( # W
        shape=(self.feature_dim, self.units * 4), # "* 4" means "o, f, i, z"
        name='att_kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)

    self.recurrent_kernel = self.add_weight( # R
        shape=(self.units, self.units * 4),
        name='att_recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    self.dense_kernel = self.add_weight( # Last Dense kernel
        shape=(self.units * 1, self.output_dim),
        name='att_dense_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    self.context_kernel = self.add_weight( # R
        shape=(self.att_input_dim, self.units * 4),
        name='att_recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

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
          name='att_bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    self.built = True

  @tf.function
  def _time_distributed_dense(self, x, w, b=None, dropout=None,
                              input_dim=None, output_dim=None,
                              timesteps=None, training=None):
    if input_dim is None:
        input_dim = K.shape(x)[2]
    if timesteps is None:
        timesteps = K.shape(x)[1]
    if output_dim is None:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x

  def _compute_update_vertex(self, x, V_tm1, c):
    """Computes carry and output using split kernels."""
    x_i, x_f, x_c, x_o = x
    V_tm1_i, V_tm1_f, V_tm1_c, V_tm1_o, V_tm1_o2, V_tm1_u, V_tm1_v = V_tm1
    c_i, c_f, c_c, c_o = c

    i = self.recurrent_activation(
        x_i
        + K.dot(V_tm1_i, self.recurrent_kernel[:, :self.units])
        + K.dot(c_i, self.context_kernel[:, :self.units])) # Attention information

    f = self.recurrent_activation(
        x_f
        + K.dot(V_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2])
        + K.dot(c_f, self.context_kernel[:, self.units:self.units * 2]))

    c = self.activation(
        x_c
        + K.dot(V_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3])
        + K.dot(c_c, self.context_kernel[:, self.units * 2:self.units * 3]))

    U = f * V_tm1_u + i * c

    o = self.recurrent_activation(
        x_o
        + K.dot(V_tm1_o, self.recurrent_kernel[:, self.units * 3:])
        + K.dot(c_o, self.context_kernel[:, self.units * 3:]))

    h_temp = o * self.activation(V_tm1_o2)
    # h size [self.units]
    h = self.dense_activation(K.dot(h_temp, self.dense_kernel))
    # h size [1] activated sigmoid

    V = h * U + (1-h) * V_tm1_v
    return h, V

  def call(self, inputs, states, training=None):
    # store the whole sequence so we can "attend" to it at each timestep

    att = states[0] # Attention input (track num, input dim) / key
    self.x_seq = K.reshape(att, (-1, self.timestep, self.att_input_dim)) # Attention input (track num, input dim)
    V_tm1 = states[1] # previous Vertex state (units)

    # Additive Attention Bahdanau et al., 2015
    self._uxpb = self._time_distributed_dense(self.x_seq,
                                              self.attention_kernel_U,
                                              b=self.attention_kernel_b,
                                              input_dim=self.att_input_dim,
                                              timesteps=self.timestep,
                                              output_dim=self.units)

    # repeat the input track to the length of the sequence (track num, feature dim))
    _tt = K.repeat(inputs, self.timestep) # inputs / query
    _Wxtt = K.dot(_tt, self.attention_kernel_W)
    et = K.dot(activations.tanh(_Wxtt + self._uxpb), K.expand_dims(self.attention_kernel_V)) # Energy

    """
    #Dot-Product Attention Luong et al., 2015 / Scaled Dot-Product Attention Vaswani 2017
    self.x_seq /= np.sqrt(self.att_input_dim)
    et = K.batch_dot(K.expand_dims(inputs), self.x_seq, axes=[1, 2])
    et = K.reshape(et, (-1, self.timestep, 1))
    """

    at = K.exp(et)
    at_sum = K.sum(at, axis=1)
    at_sum_repeated = K.repeat(at_sum, self.timestep)
    at /= at_sum_repeated  # attention weights ({batchsize}, track num, 1)
    context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1) # context


    if self.implementation == 1:
      inputs_i = inputs
      inputs_f = inputs
      inputs_c = inputs
      inputs_o = inputs
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

      c_i = context
      c_f = context
      c_c = context
      c_o = context

      x = (x_i, x_f, x_c, x_o)
      V_tm1 = (V_tm1_i, V_tm1_f, V_tm1_c, V_tm1_o, V_tm1_o2, V_tm1_u, V_tm1_v)
      c = (c_i, c_f, c_c, c_o)

      h, V = self._compute_update_vertex(x, V_tm1, c)

    return [h, at], [att, V]

  def get_config(self):
    config = {
        'units':
            self.units,
        'att_input_dim':
            self.att_input_dim,
        'output_dim':
            self.output_dim,
        'timestep':
            self.timestep,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'dense_activation':
            activations.serialize(self.dense_activation),
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
    base_config = super(AttentionVLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


