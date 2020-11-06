import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras import initializers, regularizers, constraints, activations
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import constant_op, ops
from tensorflow.python.ops import array_ops, math_ops, clip_ops
from tensorflow.python.ops import variables as variables_module

from keras.utils.generic_utils import get_custom_objects




class GraphAttentionLayer(Dense):
    """
    import from danielegrattarola/keras-gat
    https://github.com/danielegrattarola/keras-gat/blob/master/keras_gat/graph_attention_layer.py
    """

    def __init__(self,
                 feature_units,
                 attn_heads=1,
                 attn_heads_reduction="concat",  # {"concat", "average"}
                 activation="relu",
                 attn_kernel_initializer="glorot_uniform",
                 attn_kernel_regularizer=None,
                 attn_kernel_constraint=None,
                 attention=True,
                 return_attention=False,
                 node_level_bias=False,
                 use_bias=False,
                 **kwargs):

        if attn_heads_reduction not in {"concat", "average"}:
            raise ValueError("Possbile reduction methods: concat, average")

        super().__init__(units=feature_units,
                         activation=activation,
                         **kwargs)

        # Number of attention heads (K in the paper)
        self.attn_heads = attn_heads
        # Eq. 5 and 6 in the paper
        self.attn_heads_reduction = attn_heads_reduction

        self.attn_kernel_initializer \
            = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer \
            = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.attention = attention
        self.return_attention = return_attention
        self.node_level_bias = node_level_bias
        self.use_bias = use_bias
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        self.supports_masking = True
        # Populated by build()
        self.kernels = []
        self.biases = []
        self.neighbor_kernels = []
        self.attn_kernels = []
        self.attention_biases = []

        if attn_heads_reduction == "concat":
            # Output will have shape (..., K * F")
            self.output_dim = self.units * self.attn_heads
        else:
            # Output will have shape (..., F")
            self.output_dim = self.units

    def build(self, input_shape):
        X_dims, A_dims = [dims.as_list() for dims in input_shape]
        assert len(X_dims) == 3
        assert len(A_dims) == 3 and A_dims[1] == A_dims[2]

        _, N, F = X_dims

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.units),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name="kernel_{}".format(head))
            self.kernels.append(kernel)

            # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.units,),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name="bias_{}".format(head))
                self.biases.append(bias)

            if not self.attention:
                continue

            # Attention kernels
            neighbor_kernel = self.add_weight(
                                    shape=(F, self.units),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    name="kernel_neighbor_{}".format(head))

            attn_kernel = self.add_weight(
                                    shape=(self.units, 1),
                                    initializer=self.attn_kernel_initializer,
                                    regularizer=self.attn_kernel_regularizer,
                                    constraint=self.attn_kernel_constraint,
                                    name="attn_kernel_{}".format(head))

            self.neighbor_kernels.append(neighbor_kernel)
            self.attn_kernels.append(attn_kernel)

            """
            if self.use_bias:
                if self.node_level_bias:
                    biases = self.add_weight(shape=(N, N),
                                             initializer=self.bias_initializer,
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint,
                                             name="attention_bias")
                else:
                    biases = []
                    for kind in ["self", "neigbor"]:
                        name = "bias_attn_{}_{}".format(kind, head)
                        bias = self.add_weight(shape=(N,),
                                               initializer=self.bias_initializer,
                                               regularizer=self.bias_regularizer,
                                               constraint=self.bias_constraint,
                                               name=name)
                        biases.append(bias)
                self.attention_biases.append(biases)
            """

        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (B x N x F)
        A = inputs[1]  # Adjacency matrix (B x N x N)

        X_dims = X.get_shape().as_list()
        B, N, F = X_dims
        nn = tf.shape(A)[1]

        outputs = []
        attentions = []
        for head in range(self.attn_heads):
            # W in the paper (F x F" (units) )
            kernel = self.kernels[head]

            # Compute inputs to attention network
            features = K.dot(X, kernel) # (B x N x F")
            features = features + K.batch_dot(A, features) # (B x N x F")
            
            # Attention kernel a in the paper (2F" x 1)
            neighbor_kernel = self.neighbor_kernels[head] # (F x F")
            attn_kernel = self.attn_kernels[head] # (F" x 1)

            neighbor_features = K.dot(X, neighbor_kernel) # (B x N x F")

            attn_self = K.dot(features, attn_kernel) # (B x N x 1)
            attn_self = K.reshape(attn_self, (-1, nn, 1))
            attn_neighbor = K.dot(neighbor_features, attn_kernel) # 
            attn_neighbor = K.reshape(attn_neighbor, (-1, nn, 1))

            if self.use_bias and not self.node_level_bias:
                self_attn_bias, neigbor_attn_bias = self.attention_biases[head]
                attn_self = K.bias_add(attn_self, self_attn_bias) # (B * N * N) + (N * N)
                attn_neighbor = K.bias_add(attn_neighbor, neigbor_attn_bias)

            attention = attn_neighbor + tf.transpose(attn_self, (0, 2, 1))
            attention = tf.nn.tanh(attention)
            attention = K.reshape(attention, (-1, nn, nn))
            if self.use_bias and self.node_level_bias:
                bias = self.attention_biases[head]
                attention = K.bias_add(attention, bias)

            has_connection = tf.cast(tf.greater(A, 0.0), dtype=tf.float32)

            mask = -10e9 * (1.0 - has_connection)
            attention += mask
            attention = tf.nn.softmax(attention) * has_connection

            aggregation = K.batch_dot(attention, features)

            node_features = features + aggregation # (B × N × F")
            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention
            if self.return_attention:
                attentions.append(attention)

            outputs.append(node_features)

        # Aggregate the heads" output according to the reduction method
        if self.attn_heads_reduction == "concat":
            output = K.concatenate(outputs, axis=-1)  # (B x N x KF")
        else:
            output = K.mean(K.stack(outputs), axis=0)  # (B x N x F")
            # If "average", compute the activation here (Eq. 6)

        output = self.activation(output)

        if self.return_attention:
            #attentions = K.stack(attentions, axis=1)
            attentions = tf.convert_to_tensor(attentions[0])
            """
            print(type(output))
            print(output.shape)
            print(type(attentions))
            print(attentions.shape)
            """
            return (output, attentions)
        else:
            return output

    def compute_output_shape(self, input_shape):
        X_dims, A_dims = [dims.as_list() for dims in input_shape]
        assert len(X_dims) == 3
        assert len(A_dims) == 3
        output_shape = X_dims[0], X_dims[1], self.output_dim

        if self.return_attention:
            return (tf.TensorShape(output_shape),
                    tf.TensorShape(A_dims.insert(1, self.attn_heads)))
        else:
            return tf.TensorShape(output_shape)

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            output_mask = mask[0]
        else:
            output_mask = mask

        if self.return_attention:
            return [output_mask] + [None]
        else:
            return output_mask

    def get_config(self):
        config = {'feature_units': self.units,
                  'attn_heads': self.attn_heads,
                  'attn_heads_reduction': self.attn_heads_reduction,
                  'activation': self.activation,
                  'attn_kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'attn_kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'attn_kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'attention': self.attention,
                  'return_attention': self.return_attention,
                  'node_level_bias': self.node_level_bias,
                  'use_bias': self.use_bias
        }

        base_config = super(GraphAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


######################################################################################################
######################################################################################################
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


def binary_crossentropy(from_logits=False):

  def binary_crossentropy_loss(yX_train, output):
    target = yX_train[:, :, :, 0]
    target = tf.expand_dims(target, -1)
    zero_padding = yX_train[:, :, :, 1]
    zero_padding = tf.expand_dims(zero_padding, -1)

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
    target = tf.cast(target, tf.float32)
    zero_padding = tf.cast(zero_padding, tf.float32)

    # Compute cross entropy from probabilities.

    bce = zero_padding * target * math_ops.log(output + epsilon())
    bce += zero_padding * (1 - target) * math_ops.log(1 - output + epsilon())

    return -bce

  return binary_crossentropy_loss


######################################################################################################
######################################################################################################
def accuracy_all(y_true, y_pred, threshold=0.5):
  target = y_true[:, :, :, 0]
  target = tf.expand_dims(target, -1)
  zero_padding = y_true[:, :, :, 1]
  zero_padding = tf.expand_dims(zero_padding, -1)
  nb_classes = y_pred.shape[1]

  threshold = math_ops.cast(threshold, y_pred.dtype)
  output = math_ops.cast(y_pred > threshold, y_pred.dtype)

  target_zp = tf.boolean_mask(target, zero_padding==1)
  output_zp = tf.boolean_mask(output, zero_padding==1)
  #score = tf.boolean_mask(zero_padding, math_ops.equal(target, output))

  return K.mean(math_ops.equal(target_zp, output_zp), axis=-1)



######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
class GraphAdditiveAttentionLayer(Dense):
    """
    import from danielegrattarola/keras-gat
    https://github.com/danielegrattarola/keras-gat/blob/master/keras_gat/graph_attention_layer.py
    """

    def __init__(self,
                 feature_units,
                 attn_heads=1,
                 attn_heads_reduction="concat",  # {"concat", "average"}
                 activation="relu",
                 attn_kernel_initializer="glorot_uniform",
                 attn_kernel_regularizer=None,
                 attn_kernel_constraint=None,
                 attention=True,
                 return_attention=False,
                 node_level_bias=False,
                 use_bias=False,
                 **kwargs):

        if attn_heads_reduction not in {"concat", "average"}:
            raise ValueError("Possbile reduction methods: concat, average")

        super().__init__(units=feature_units,
                         activation=activation,
                         **kwargs)

        # Number of attention heads (K in the paper)
        self.attn_heads = attn_heads
        # Eq. 5 and 6 in the paper
        self.attn_heads_reduction = attn_heads_reduction

        self.attn_kernel_initializer \
            = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer \
            = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.attention = attention
        self.return_attention = return_attention
        self.node_level_bias = node_level_bias
        self.use_bias = use_bias
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        self.supports_masking = True
        # Populated by build()
        self.kernels = []
        self.neighbor_kernels = []
        self.attn_kernel_Ws = []
        self.attn_kernel_Us = []
        self.attn_kernel_Vs = []

        if attn_heads_reduction == "concat":
            # Output will have shape (..., K * F")
            self.output_dim = self.units * self.attn_heads
        else:
            # Output will have shape (..., F")
            self.output_dim = self.units

    def build(self, input_shape):
        X_dims, A_dims = [dims.as_list() for dims in input_shape]
        assert len(X_dims) == 3
        assert len(A_dims) == 3 and A_dims[1] == A_dims[2]

        _, N, F = X_dims

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.units),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name="kernel_{}".format(head))

            neighbor_kernel = self.add_weight(
                                    shape=(F, self.units),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    name="kernel_neighbor_{}".format(head))

            self.kernels.append(kernel)
            self.neighbor_kernels.append(neighbor_kernel)

            if not self.attention:
                continue

            # Attention kernels
            attn_kernel_W = self.add_weight(
                                    shape=(self.units, self.units),
                                    initializer=self.attn_kernel_initializer,
                                    regularizer=self.attn_kernel_regularizer,
                                    constraint=self.attn_kernel_constraint,
                                    name="attn_kernel_W_{}".format(head))

            attn_kernel_U = self.add_weight(
                                    shape=(self.units, self.units),
                                    initializer=self.attn_kernel_initializer,
                                    regularizer=self.attn_kernel_regularizer,
                                    constraint=self.attn_kernel_constraint,
                                    name="attn_kernel_U_{}".format(head))

            attn_kernel_V = self.add_weight(
                                    shape=(self.units,),
                                    initializer=self.attn_kernel_initializer,
                                    regularizer=self.attn_kernel_regularizer,
                                    constraint=self.attn_kernel_constraint,
                                    name="attn_kernel_V_{}".format(head))

            self.attn_kernel_Ws.append(attn_kernel_W)
            self.attn_kernel_Us.append(attn_kernel_U)
            self.attn_kernel_Vs.append(attn_kernel_V)


        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (B x N x F)
        A = inputs[1]  # Adjacency matrix (B x N x N)

        X_dims = X.get_shape().as_list()
        B, N, F = X_dims
        nn = tf.shape(A)[1]

        outputs = []
        attentions = []
        for head in range(self.attn_heads):
            # W in the paper (F x F" (units) )
            kernel = self.kernels[head]

            # Compute inputs to attention network
            features = K.dot(X, kernel) # (B x N x F")
            features = features + K.batch_dot(A, features) # (B x N x F")
            
            # Attention kernel a in the paper (2F" x 1)
            neighbor_kernel = self.neighbor_kernels[head] # (F x F")
            neighbor_features = K.dot(X, neighbor_kernel) # (B x N x F")

            # Additive Attention
            attn_kernel_W = self.attn_kernel_Ws[head] # (F" x F")
            attn_kernel_U = self.attn_kernel_Us[head] # (F" x F")
            attn_kernel_V = self.attn_kernel_Vs[head] # (F" x 1)
            #attention = []
            #attention = tf.zeros([0])
            attention = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            i = tf.constant(0)
            _attn_neighbor = K.dot(neighbor_features, attn_kernel_U)

            c = lambda i, attention, features, nn, attn_kernel_W, attn_kernel_V, _attn_neighbor : i < nn
            
            def b(i, attention, features, nn, attn_kernel_W, attn_kernel_V, _attn_neighbor):

                _one_feature = features[:, i, :]
                _one_feature_repeat = K.repeat(_one_feature, nn)
                _attn_feature = K.dot(_one_feature_repeat, attn_kernel_W)
                et = K.dot(tf.nn.tanh(_attn_feature + _attn_neighbor), K.expand_dims(attn_kernel_V))
                at = K.exp(et)
                at_sum = K.sum(at, axis=1)
                at_sum_repeated = K.repeat(at_sum, nn)
                at /= at_sum_repeated  # attention weights (B, N, 1)
                at = K.reshape(at, (-1, nn))
                print("at.shape")
                print(at)
                #attention.append(at)
                #if i == 0: attention = at
                #else: attention = tf.concat([attention, at], 2)
                attention = attention.write(i, at)

                return attention

            attention = tf.while_loop(cond=c, body=b, loop_vars=[i, attention, features, nn, attn_kernel_W, attn_kernel_V, _attn_neighbor])
            """
            for i in range(nn):
                _one_feature = features[:, i, :]
                _one_feature_repeat = K.repeat(_one_feature, nn)
                _attn_feature = K.dot(_one_feature_repeat, attn_kernel_W)
                et = K.dot(tf.nn.tanh(_attn_feature + _attn_neighbor), K.expand_dims(attn_kernel_V))
                at = K.exp(et)
                at_sum = K.sum(at, axis=1)
                at_sum_repeated = K.repeat(at_sum, nn)
                at /= at_sum_repeated  # attention weights (B, N, 1)
                at = K.reshape(at, (-1, nn))
                print("at.shape")
                print(at)
                #attention.append(at)
                #if i == 0: attention = at
                #else: attention = tf.concat([attention, at], 2)
                attention = attention.write(i, at)
            """

            #attention = tf.convert_to_tensor(attention)
            attention = attention.stack()
            attention = tf.transpose(attention, (1, 0, 2))
            print("attention.shape")
            print(attention)
            attention = K.reshape(attention, (-1, nn, nn))

            has_connection = tf.cast(tf.greater(A, 0.0), dtype=tf.float32)

            mask = -10e9 * (1.0 - has_connection)
            attention += mask
            attention = tf.nn.softmax(attention) * has_connection

            aggregation = K.batch_dot(attention, features)

            node_features = features + aggregation # (B × N × F")

            # Add output of attention
            if self.return_attention:
                attentions.append(attention)

            outputs.append(node_features)

        # Aggregate the heads" output according to the reduction method
        if self.attn_heads_reduction == "concat":
            output = K.concatenate(outputs, axis=-1)  # (B x N x KF")
        else:
            output = K.mean(K.stack(outputs), axis=0)  # (B x N x F")
            # If "average", compute the activation here (Eq. 6)

        output = self.activation(output)

        if self.return_attention:
            #attentions = K.stack(attentions, axis=1)
            attentions = tf.convert_to_tensor(attentions)
            return (output, attentions)
        else:
            return output

    def compute_output_shape(self, input_shape):
        X_dims, A_dims = [dims.as_list() for dims in input_shape]
        assert len(X_dims) == 3
        assert len(A_dims) == 3
        output_shape = X_dims[0], X_dims[1], self.output_dim

        if self.return_attention:
            return (tf.TensorShape(output_shape),
                    tf.TensorShape(A_dims.insert(1, self.attn_heads)))
        else:
            return tf.TensorShape(output_shape)

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            output_mask = mask[0]
        else:
            output_mask = mask

        if self.return_attention:
            return [output_mask] + [None]
        else:
            return output_mask

    def get_config(self):
        config = {'feature_units': self.units,
                  'attn_heads': self.attn_heads,
                  'attn_heads_reduction': self.attn_heads_reduction,
                  'activation': self.activation,
                  'attn_kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'attn_kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'attn_kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'attention': self.attention,
                  'return_attention': self.return_attention,
                  'node_level_bias': self.node_level_bias,
                  'use_bias': self.use_bias
        }

        base_config = super(GraphAdditiveAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



