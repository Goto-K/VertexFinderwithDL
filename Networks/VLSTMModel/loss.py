import tensorflow as tf 
from tensorflow.python.keras import activations, constraints, initializers, regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import constant_op, ops
from tensorflow.python.ops import math_ops, clip_ops
from tensorflow.python.ops import variables as variables_module


def binary_crossentropy(pair_reinforce=False, from_logits=False):

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
      while output.op.type == 'Identity':
        output = output.op.inputs[0]
      if output.op.type == 'Sigmoid':
        assert len(output.op.inputs) == 1
        output = output.op.inputs[0]
        return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

    epsilon_ = constant_op.constant(tf.keras.backend.epsilon(), dtype=output.dtype.base_dtype)
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

def accuracy_all(y_true, y_pred, threshold=0.5):

  nb_classes = y_pred.shape[1]
  target = y_true[:, :, 0]
  target = tf.expand_dims(target, -1)
  zero_padding = y_true[:, :, 1]
  zero_padding = tf.expand_dims(zero_padding, -1)

  threshold = math_ops.cast(threshold, y_pred.dtype)
  output = math_ops.cast(y_pred > threshold, y_pred.dtype)

  target_zp = tf.boolean_mask(target, zero_padding==1)
  output_zp = tf.boolean_mask(output, zero_padding==1)

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

