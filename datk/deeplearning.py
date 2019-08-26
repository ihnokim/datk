import tensorflow as tf
import numpy as np
import pandas as pd
import datk


def collect_data(df, collector):
    # collector(row) should return (x, y)
    xret = []
    yret = []
    for row in df.itertuples():
        xrow, yrow = collector(row)
        xret.append(xrow)
        yret.append(yrow)
    return np.array(xret), np.array(yret)


def rsq(v1, v2):
    mu1, var1 = tf.nn.moments(v1, axes=[0])
    mu2, var2 = tf.nn.moments(v2, axes=[0])
    d1 = v1 - mu1 * tf.ones(tf.shape(v1))
    d2 = v2 - mu2 * tf.ones(tf.shape(v2))
    return tf.square(tf.divide(tf.reduce_mean(tf.multiply(d1, d2), 0), tf.sqrt(tf.multiply(var1, var2))))