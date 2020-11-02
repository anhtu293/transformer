import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.linalg.matmul(q, k, transpose_b=True)
    d_k = tf.cast(tf.shape(q)[-1], tf.float32)
    logits = matmul_qk/tf.math.sqrt(d_k)
    if mask is not None:
        logits += (mask * 1e-9)
    weights = tf.nn.softmax(logits, axis=-1)
    outputs = tf.linalg.matmul(weights, v)

    return outputs, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = self.d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.linear = tf.keras.layers.Dense(d_model)

    def split_head(self, x, batch_size):
        # Transpose to (batch_size, num_heads, seq_length, self.depth)
        # before return for calculation in scaled_dot_product_attention
        new_dim = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(new_dim, perm=[0, 2, 1, 3])

    def __call__(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_head(q, batch_size)
        k = self.split_head(k, batch_size)
        v = self.split_head(v, batch_size)

        # Perform scaled dot product attention
        attn_outputs, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        # Retranspose to (batch_size, seq_length, num_heads, depth) to
        # reshape to original dimension
        attn_outputs = tf.transpose(attn_outputs, perm=[0, 2, 1, 3])
        outputs = tf.reshape(attn_outputs, (batch_size, -1, self.d_model))

        return outputs, attn_weights
