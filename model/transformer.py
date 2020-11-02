import tensorflow as tf
import numpy as np
from .attention import MultiHeadAttention


def get_angle(pos, i, d_model):
    angle = pos / np.power(10000, 2 * (i//2) / d_model)
    return angle


def positional_encoding(max_pos, d_model):
    angles = get_angle(np.arange(max_pos)[:, np.newaxis],
                       np.arange(d_model)[np.newaxis, :], d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    angles = np.expand_dims(angles, axis=0)
    return angles


class FFN(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dense_1 = tf.keras.layers.Dense(self.d_ff, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(self.d_model, activation='relu')

    def __call__(self, x):
        x = self.dense_1(x)
        outputs = self.dense_2(x)

        return outputs


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rate = rate
        self.multi_head_attn = MultiHeadAttention(d_model, num_heads)
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate)
        self.ffn = FFN(d_model, d_ff)

    def __call__(self, x, mask, training):
        attn_outputs, attn_weights = self.multi_head_attn(x, x, x, mask)
        attn_outputs = self.dropout_1(attn_outputs, training=training)
        x = self.norm_1(x + attn_outputs)
        ffn_outputs = self.ffn(x)
        ffn_outputs = self.dropout_2(ffn_outputs, training=training)
        outputs = self.norm_2(x + ffn_outputs)

        return outputs


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rate = 0.1
        self.masked_multi_head_attn = MultiHeadAttention(d_model,
                                                         num_heads)
        self.multi_head_attn = MultiHeadAttention(d_model, num_heads)
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate)
        self.dropout_3 = tf.keras.layers.Dropout(rate)
        self.ffn = FFN(d_model, d_ff)

    def __call__(self, x, mask, enc_outputs, enc_mask, training):
        attn_outputs_1, attn_weights_1 = self.masked_multi_head_attn(x, x, x, mask)
        attn_outputs_1 = self.dropout_1(attn_outputs_1, training=training)
        outputs_1 = self.norm_1(x + attn_outputs_1)
        attn_outputs_2, attn_weights_2 = self.multi_head_attn(outputs_1,
                                                              enc_outputs,
                                                              enc_outputs,
                                                              enc_mask)
        attn_outputs_2 = self.dropout_2(attn_outputs_2, training=training)
        outputs_2 = self.norm_2(attn_outputs_2 + outputs_1)
        ffn_outputs = self.ffn(outputs_2)
        ffn_outputs = self.dropout_3(ffn_outputs, training=training)
        outputs = self.norm_3(outputs_2 + ffn_outputs)

        return outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, num_layers,
                 max_pos_enc, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_pos_enc = max_pos_enc
        self.rate = rate
        self.input_vocab_size = input_vocab_size
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_pos_enc, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, d_ff)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, x, mask, training):
        seq_length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_length, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, num_layers,
                 max_pos_enc, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_pos_enc = max_pos_enc
        self.output_vocab_size = target_vocab_size
        self.rate = rate
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_pos_enc, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, d_ff)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, enc_outputs, enc_mask, x, output_mask, training):
        seq_length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_length, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            # noinspection PyCallingNonCallable
            x = self.dec_layers[i](x, output_mask, enc_outputs, enc_mask, training)

        return x


class Transformer(tf.keras.models.Sequential):
    def __init__(self, d_model, num_heads, d_ff, num_layers,
                 input_max_pe, target_max_pe, input_vocab_size, target_vocab_size):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.input_max_pe = input_max_pe
        self.output_max_pe = target_max_pe
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers,
                               input_max_pe, input_vocab_size)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers,
                               target_max_pe, target_vocab_size)
        self.linear = tf.keras.layers.Dense(target_vocab_size)

    def __call__(self, inp, target, enc_padding_mask, forward_mask, dec_padding_mask, training):
        x = self.encoder(inp, enc_padding_mask, training)
        x = self.decoder(x, dec_padding_mask, target, forward_mask, training)
        x = self.linear(x)

        return x
