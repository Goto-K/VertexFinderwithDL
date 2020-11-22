import numpy as np
import tensorflow as tf
import keras.backend as K


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

