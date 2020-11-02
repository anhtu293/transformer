import tensorflow as tf


def padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def forward_mask(size):
    # mask tokens behind actual position
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask
