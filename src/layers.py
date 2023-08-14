from tensorflow import keras

import tensorflow as tf

from tensorflow.keras import regularizers



class Lacuna(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            keras.layers.Bidirectional(
                keras.layers.LSTM(64,
                                  dropout=0.2,
                                  recurrent_dropout=0.2,
                                  return_sequences=False,
                                  kernel_regularizer=regularizers.l2(0.00001),
                                  bias_regularizer=regularizers.l2(0.00001),
                                  recurrent_regularizer=regularizers.l2(0.00001),
                                  activity_regularizer=regularizers.l2(0.00001))),

        ])

        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)

        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dense_dim': self.dense_dim
        })

        return config


class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.sec_dense_dim = self.dense_dim / 2

        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            keras.layers.Dense(dense_dim, activation='relu', kernel_regularizer=regularizers.l2(0.000001),
                               bias_regularizer=regularizers.l2(0.000001)),
            keras.layers.Dense(self.sec_dense_dim, activation='relu', kernel_regularizer=regularizers.l2(0.000001),
                               bias_regularizer=regularizers.l2(0.000001)),
            keras.layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(0.000001),
                               bias_regularizer=regularizers.l2(0.000001)),
        ])

        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)

        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dense_dim': self.dense_dim
        })

        return config