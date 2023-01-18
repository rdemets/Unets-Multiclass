"""Metrics for measuring machine learning algorithm performances
adapted from https://github.com/deaspo/Unet_MedicalImagingSegmentation
"""

from keras import backend as K
import tensorflow as tf
import numpy as np

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        #y_pred_ = tf.to_int32(y_pred > t)
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        if K.int_shape(y_pred)[-1] >1:
            num_class = K.int_shape(y_pred)[-1]
        else:    
            num_class = K.int_shape(y_pred)[-1]+1
        score, up_opt = tf.compat.v1.metrics.mean_iou(y_true, y_pred_, num_class)
        K.get_session().run(tf.compat.v1.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)