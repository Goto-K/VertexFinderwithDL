import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.engine.training import Model
from keras import layers



def custom_mean_squared_error(y_weight):
   
    def custom_mean_squared_error_loss(y_true, y_pred):

        # [r, theta, phi]
        _y_weight = K.constant(y_weight) if not K.is_tensor(y_weight) else y_weight
        y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)

        loss = K.sum(_y_weight * (y_pred - y_true)*(y_pred - y_true), axis=-1, keepdims=False)

        return loss

    return custom_mean_squared_error_loss


def custom_categorical_crossentropy(y_weight, from_logits=False, label_smoothing=0):

    def custom_categorical_crossentropy_loss(y_true, y_pred):
    
        _y_weight = K.constant(y_weight) if not K.is_tensor(y_weight) else y_weight
        y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)
        
        if label_smoothing is not 0:
        
            smoothing = K.cast_to_floatx(label_smoothing)
            
            def _smooth_labels():
            
                num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
                return y_true * (1.0 - smoothing) + (smoothing / num_classes)

            y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)

        if from_logits:
            output = softmax(y_pred)
        else:
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)

        loss = K.sum(y_true * _y_weight * -K.log(y_pred), axis=-1, keepdims=False)

        return loss
    
    return custom_categorical_crossentropy_loss


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed

