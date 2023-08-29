from tensorflow import keras
import config as conf
import tensorflow as tf
from tensorflow.keras import regularizers

# tf.keras.saving.get_custom_objects().clear()


# @tf.keras.saving.register_keras_serializable(package='src', name='lacuna_layer')
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


# @tf.keras.saving.register_keras_serializable(package='src', name='transformer_encoder_layer')
class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            keras.layers.Dense(dense_dim, activation='relu', kernel_regularizer=regularizers.l2(conf.L2),
                               bias_regularizer=regularizers.l2(conf.L2)),

            keras.layers.Dense(dense_dim, kernel_regularizer=regularizers.l2(conf.L2),
                               bias_regularizer=regularizers.l2(conf.L2)),
            keras.layers.Dense(dense_dim, activation='relu', kernel_regularizer=regularizers.l2(conf.L2),
                               bias_regularizer=regularizers.l2(conf.L2)),

            keras.layers.Dense(dense_dim, kernel_regularizer=regularizers.l2(conf.L2),
                               bias_regularizer=regularizers.l2(conf.L2)),
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
        base_config = super().get_config()
        config = {
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dense_dim': self.dense_dim,
            }
        return {**base_config, **config}

    # def from_config(cls_config, config):
    #     embed_dim = config.pop('embed_dim')
    #     num_heads = config.pop('num_heads')
    #     dense_dim = config.pop('dense_dim')
    #
    #     return cls_config(embed_dim, num_heads, dense_dim, **config)
