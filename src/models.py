from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import regularizers
from layers import Lacuna, TransformerEncoder
import config



# def build():
#     inputs = keras.Input(shape=(None,), dtype='int64')
#
#     x = keras.layers.Embedding(config.VOCABSIZE, 64)(inputs)
#     b1 = keras.layers.Bidirectional(
#         keras.layers.LSTM(256,
#                           return_sequences=True,
#                           kernel_regularizer=regularizers.l2(0.000001),
#                           bias_regularizer=regularizers.l2(0.000001)))(x)
#     b1 = keras.layers.Bidirectional(
#         keras.layers.LSTM(256,
#                           return_sequences=True,
#                           kernel_regularizer=regularizers.l2(0.000001),
#                           bias_regularizer=regularizers.l2(0.000001)))(b1)
#     b1 = keras.layers.Bidirectional(
#         keras.layers.LSTM(256,
#                           return_sequences=True,
#                           kernel_regularizer=regularizers.l2(0.000001),
#                           bias_regularizer=regularizers.l2(0.000001)))(b1)
#     b1 = keras.layers.Bidirectional(
#         keras.layers.LSTM(256,
#                           return_sequences=True,
#                           kernel_regularizer=regularizers.l2(0.000001),
#                           bias_regularizer=regularizers.l2(0.000001)))(b1)
#     b1 = keras.layers.Bidirectional(
#         keras.layers.LSTM(256,
#                           return_sequences=False,
#                           kernel_regularizer=regularizers.l2(0.000001),
#                           bias_regularizer=regularizers.l2(0.000001)))(b1)
#     b1 = keras.layers.Dense(256,
#                             activation='relu',
#                             kernel_regularizer=regularizers.l2(0.000001),
#                             bias_regularizer=regularizers.l2(0.000001),
#                             activity_regularizer=regularizers.l2(0.000001))(b1)
#     b1 = keras.layers.Dropout(0.2)(b1)
#     b1 = keras.layers.Dense(128,
#                             activation='relu',
#                             kernel_regularizer=regularizers.l2(0.000001),
#                             bias_regularizer=regularizers.l2(0.000001),
#                             activity_regularizer=regularizers.l2(0.000001))(b1)
#     b1 = keras.layers.Dense(64,
#                             activation='relu',
#                             kernel_regularizer=regularizers.l2(0.000001),
#                             bias_regularizer=regularizers.l2(0.000001),
#                             activity_regularizer=regularizers.l2(0.000001))(b1)
#     b1 = keras.layers.Dropout(0.2)(b1)
#     b1 = keras.layers.Dense(32,
#                             activation='relu',
#                             kernel_regularizer=regularizers.l2(0.000001),
#                             bias_regularizer=regularizers.l2(0.000001),
#                             activity_regularizer=regularizers.l2(0.000001))(b1)
#
#     outputs = keras.layers.Dense(config.OUTPUTDIM, activation='softmax')(b1)
#     model = keras.Model(inputs, outputs)
#
#     model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNINGRATE),
#                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#                   metrics=['accuracy'])
#
#     return model


def build():
    inputs = keras.Input(shape=(None,), dtype='int64')

    x = keras.layers.Embedding(config.VOCABSIZE, 64)(inputs)
    b1 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.000001), bias_regularizer=regularizers.l2(0.000001)))(x)
    b1 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.000001), bias_regularizer=regularizers.l2(0.000001)))(b1)
    b1 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.000001), bias_regularizer=regularizers.l2(0.000001)))(b1)
    b1 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.000001), bias_regularizer=regularizers.l2(0.000001)))(b1)
    b1 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.000001), bias_regularizer=regularizers.l2(0.000001)))(b1)
    b1 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.000001), bias_regularizer=regularizers.l2(0.000001)))(b1)
    b1 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.000001), bias_regularizer=regularizers.l2(0.000001)))(b1)
    b1 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=False, kernel_regularizer=regularizers.l2(0.000001), bias_regularizer=regularizers.l2(0.000001)))(b1)
    # b1 = keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.000001), bias_regularizer=regularizers.l2(0.000001), activity_regularizer=regularizers.l2(0.000001))(b1)
    # b1 = keras.layers.Dropout(0.2)(b1)
    # b1 = keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.000001), bias_regularizer=regularizers.l2(0.000001), activity_regularizer=regularizers.l2(0.000001))(b1)
    # b1 = keras.layers.Dropout(0.2)(b1)
    #
    # b1 = keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.000001), bias_regularizer=regularizers.l2(0.000001), activity_regularizer=regularizers.l2(0.000001))(b1)
    # b1 = keras.layers.Dropout(0.2)(b1)
    #
    # b1 = keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.000001), bias_regularizer=regularizers.l2(0.000001), activity_regularizer=regularizers.l2(0.000001))(b1)
    # b1 = keras.layers.Dropout(0.2)(b1)
    #
    b1 = keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.000001), bias_regularizer=regularizers.l2(0.000001), activity_regularizer=regularizers.l2(0.000001))(b1)

    outputs = keras.layers.Dense(config.OUTPUTDIM, activation='softmax')(b1)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNINGRATE),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


def build_lacuna(hp):
    inputs = keras.Input(shape=(None,), dtype='int64')

    x = keras.layers.Embedding(config.VOCABSIZE,
                               hp.Int('embed_dim', min_value=32, max_value=1024, step=32),
                               embeddings_regularizer=regularizers.l2(
                                   hp.Float('embeddings_regularizer', min_value=config.MINREGULARIZER,
                                            max_value=config.MAXREGULARIZER, sampling='log')))(inputs)

    b1 = keras.layers.Bidirectional(
        keras.layers.LSTM(hp.Int('units_l', min_value=32, max_value=1024, step=32),
                          activation=hp.Choice('activation', ['relu', 'tanh', 'selu', 'elu']),
                          return_sequences=False,
                          kernel_regularizer=regularizers.l2(
                              hp.Float('kernel_regularizer', min_value=config.MINREGULARIZER,
                                       max_value=config.MAXREGULARIZER, sampling='log')),
                          bias_regularizer=regularizers.l2(hp.Float('bias_regularizer', min_value=config.MINREGULARIZER,
                                                                    max_value=config.MAXREGULARIZER, sampling='log')),
                          recurrent_regularizer=regularizers.l2(
                              hp.Float('recurrent_regularizer', min_value=config.MINREGULARIZER,
                                       max_value=config.MAXREGULARIZER, sampling='log')),
                          activity_regularizer=regularizers.l2(
                              hp.Float('activity_regularizer', min_value=config.MINREGULARIZER,
                                       max_value=config.MAXREGULARIZER, sampling='log'))))(x)
    b1 = keras.layers.Dense(hp.Int('units_1', min_value=32, max_value=1024, step=32),
                            activation=hp.Choice('activation1', ['relu', 'tanh', 'elu']))(b1)
    b1 = keras.layers.Dense(hp.Int('units_2', min_value=32, max_value=1024, step=32),
                            activation=hp.Choice('activation2', ['relu', 'tanh', 'elu']))(b1)
    b1 = keras.layers.Dense(hp.Int('units_3', min_value=32, max_value=1024, step=32),
                            activation=hp.Choice('activation3', ['relu', 'tanh', 'elu']))(b1)
    b1 = keras.layers.Dense(hp.Int('units_4', min_value=32, max_value=1024, step=32),
                            activation=hp.Choice('activation4', ['relu', 'tanh', 'elu']))(b1)
    b1 = keras.layers.Dense(hp.Int('units_5', min_value=32, max_value=1024, step=32),
                            activation=hp.Choice('activation5', ['relu', 'tanh', 'elu']))(b1)
    b1 = keras.layers.Dense(hp.Int('units_6', min_value=32, max_value=1024, step=32),
                            activation=hp.Choice('activation6', ['relu', 'tanh', 'elu']))(b1)

    outputs = keras.layers.Dense(config.OUTPUTDIM, activation='softmax')(b1)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=hp.Float('learning_rate',
                               min_value=config.MINLEARNINGRATE,
                               max_value=config.MAXLEARNINGRATE,
                               sampling='log'), clipvalue=0.5, beta_1=0.9, beta_2=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model

#
# def build_simple_lacuna(hp=keras_tuner.HyperParameters()):
#     inputs = keras.Input(shape=(None,), dtype='int64')
#     x = keras.layers.Embedding(config.VOCABSIZE, config.EMBEDDIM, embeddings_regularizer=regularizers.l2(0.01))(inputs)
#     x = keras.layers.SpatialDropout1D(0.2)(x)
#
#     x = keras.layers.Bidirectional(
#         keras.layers.LSTM(hp.Int('units', min_value=32, max_value=512, step=32),
#                           dropout=0.2,
#                           recurrent_dropout=0.2,
#                           return_sequences=True,
#                           kernel_regularizer=regularizers.l2(0.001),
#                           bias_regularizer=regularizers.l2(0.001),
#                           recurrent_regularizer=regularizers.l2(0.001),
#                           activity_regularizer=regularizers.l2(0.001)))(x)
#     x = keras.layers.Bidirectional(
#         keras.layers.LSTM(352,
#                           dropout=0.2,
#                           recurrent_dropout=0.2,
#                           return_sequences=False,
#                           kernel_regularizer=regularizers.l2(0.001),
#                           bias_regularizer=regularizers.l2(0.001),
#                           recurrent_regularizer=regularizers.l2(0.001),
#                           activity_regularizer=regularizers.l2(0.001)))(x)
#
#     # x = keras.layers.Bidirectional(
#     #     keras.layers.LSTM(hp.Int('units', min_value=32, max_value=512, step=32),
#     #                       dropout=0.2,
#     #                       recurrent_dropout=0.2,
#     #                       return_sequences=True,
#     #                       kernel_regularizer=regularizers.l2(0.001),
#     #                       bias_regularizer=regularizers.l2(0.001),
#     #                       recurrent_regularizer=regularizers.l2(0.001),
#     #                       activity_regularizer=regularizers.l2(0.001)))(x)
#     # x = keras.layers.Bidirectional(
#     #     keras.layers.LSTM(hp.Int('units', min_value=32, max_value=512, step=32),
#     #                       dropout=0.2,
#     #                       recurrent_dropout=0.2,
#     #                       return_sequences=False,
#     #                       kernel_regularizer=regularizers.l2(0.001),
#     #                       bias_regularizer=regularizers.l2(0.001),
#     #                       recurrent_regularizer=regularizers.l2(0.001),
#     #                       activity_regularizer=regularizers.l2(0.001)))(x)
#
#     outputs = keras.layers.Dense(config.OUTPUTDIM,
#                                  activation='softmax',
#                                  kernel_regularizer=regularizers.l2(0.001),
#                                  bias_regularizer=regularizers.l2(0.001))(x)
#
#     model = keras.Model(inputs, outputs)
#
#     model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNINGRATE,
#                                                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#                                                   metrics=['accuracy']))
#
#     # model.compile(
#     #     optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate',
#     #                                                            min_value=config.MINLEARNINGRATE,
#     #                                                            max_value=config.MAXLEARNINGRATE,
#     #                                                            sampling='log')),
#     #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     #     metrics=['accuracy'])
#
#     return model
#
#
# def build_dense_lacuna(vocab_size, embed_dim, dense_dim, output_dim):
#     inputs = keras.Input(shape=(None,), dtype='int64')
#     x = keras.layers.Embedding(vocab_size, embed_dim, embeddings_regularizer=regularizers.l2(0.01))(inputs)
#
#     b1 = keras.layers.Conv1D(
#         filters=1024,
#         kernel_size=1,
#         strides=1,
#         dilation_rate=1,
#         activation='relu',
#         padding='valid',
#         kernel_regularizer=regularizers.l2(0.000001),
#         bias_regularizer=regularizers.l2(0.000001),
#     )(x)
#
#     b1 = keras.layers.Conv1D(
#         filters=1024,
#         kernel_size=1,
#         strides=1,
#         dilation_rate=1,
#         activation='relu',
#         padding='valid',
#         kernel_regularizer=regularizers.l2(0.000001),
#         bias_regularizer=regularizers.l2(0.000001),
#     )(b1)
#
#     b1 = keras.layers.GlobalMaxPooling1D()(b1)
#
#     b1 = keras.layers.Dense(1024,
#                             activation='relu',
#                             kernel_regularizer=regularizers.l2(0.01),
#                             bias_regularizer=regularizers.l2(0.01))(b1)
#
#     b2 = keras.layers.Conv1D(
#         filters=512,
#         kernel_size=1,
#         strides=1,
#         dilation_rate=1,
#         activation='relu',
#         padding='valid',
#         kernel_regularizer=regularizers.l2(0.000001),
#         bias_regularizer=regularizers.l2(0.000001),
#     )(x)
#
#     b2 = keras.layers.Conv1D(
#         filters=512,
#         kernel_size=1,
#         strides=1,
#         dilation_rate=1,
#         activation='relu',
#         padding='valid',
#         kernel_regularizer=regularizers.l2(0.000001),
#         bias_regularizer=regularizers.l2(0.000001),
#     )(b2)
#
#     b2 = keras.layers.GlobalMaxPooling1D()(b2)
#
#     b2 = keras.layers.Dense(512, activation='relu',
#                             kernel_regularizer=regularizers.l2(0.01),
#                             bias_regularizer=regularizers.l2(0.01))(b2)
#
#     b3 = keras.layers.Conv1D(
#         filters=256,
#         kernel_size=1,
#         strides=1,
#         dilation_rate=1,
#         activation='relu',
#         padding='valid',
#         kernel_regularizer=regularizers.l2(0.000001),
#         bias_regularizer=regularizers.l2(0.000001),
#     )(x)
#
#     b3 = keras.layers.Conv1D(
#         filters=256,
#         kernel_size=1,
#         strides=1,
#         dilation_rate=1,
#         activation='relu',
#         padding='valid',
#         kernel_regularizer=regularizers.l2(0.000001),
#         bias_regularizer=regularizers.l2(0.000001),
#     )(b3)
#
#     b3 = keras.layers.GlobalMaxPooling1D()(b3)
#
#     b3 = keras.layers.Dense(256,
#                             activation='relu',
#                             # kernel_initializer='lecun_normal',
#                             kernel_regularizer=regularizers.l2(0.01),
#                             bias_regularizer=regularizers.l2(0.01))(b3)
#
#     b4 = keras.layers.Conv1D(
#         filters=128,
#         kernel_size=1,
#         strides=1,
#         dilation_rate=1,
#         activation='relu',
#         padding='valid',
#         kernel_regularizer=regularizers.l2(0.000001),
#         bias_regularizer=regularizers.l2(0.000001),
#     )(x)
#     b4 = keras.layers.Conv1D(
#         filters=128,
#         kernel_size=1,
#         strides=1,
#         dilation_rate=1,
#         activation='relu',
#         padding='valid',
#         kernel_regularizer=regularizers.l2(0.000001),
#         bias_regularizer=regularizers.l2(0.000001),
#     )(b4)
#     b4 = keras.layers.GlobalMaxPooling1D()(b4)
#
#     b4 = keras.layers.Dense(128, activation='relu',
#                             kernel_regularizer=regularizers.l2(0.01),
#                             bias_regularizer=regularizers.l2(0.01))(b4)
#
#     x = keras.layers.Concatenate()([b4, b3, b2, b1])
#     x = keras.layers.Dense(128,
#                            kernel_regularizer=regularizers.l2(0.01),
#                            bias_regularizer=regularizers.l2(0.01))(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     outputs = keras.layers.Dense(output_dim,
#                                  activation='softmax',
#                                  kernel_regularizer=regularizers.l2(0.01),
#                                  bias_regularizer=regularizers.l2(0.01))(x)
#
#     model = keras.Model(inputs, outputs)
#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=1.5625e-04, clipvalue=0.5, beta_1=0.9, beta_2=0.9),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=['accuracy'])
#
#     return model
#
#
# def build_2b_lacuna(vocab_size, embed_dim, dense_dim, output_dim):
#     inputs = keras.Input(shape=(None,), dtype='int64')
#     x = keras.layers.Embedding(vocab_size, embed_dim, embeddings_regularizer=regularizers.l2(0.01))(inputs)
#
#     b1 = Lacuna(embed_dim, dense_dim, 12)(x)
#     # b1 = Lacuna(embed_dim, dense_dim, 6)(b1)
#     # b1 = keras.layers.Dense(64,
#     #                         activation='relu',
#     #                         kernel_regularizer=regularizers.l2(0.01),
#     #                         bias_regularizer=regularizers.l2(0.01))(b1)
#     b1 = keras.layers.Dense(32,
#                             activation='relu',
#                             kernel_regularizer=regularizers.l2(0.01),
#                             bias_regularizer=regularizers.l2(0.01))(b1)
#     b1 = keras.layers.GlobalMaxPooling1D()(b1)
#
#     b2 = Lacuna(embed_dim, dense_dim, 12)(x)
#     # b2 = Lacuna(embed_dim, dense_dim, 6)(b2)
#     # b2 = keras.layers.Dense(64, activation='relu',
#     #                         kernel_regularizer=regularizers.l2(0.01),
#     #                         bias_regularizer=regularizers.l2(0.01))(b2)
#     b2 = keras.layers.Dense(32, activation='relu',
#                             kernel_regularizer=regularizers.l2(0.01),
#                             bias_regularizer=regularizers.l2(0.01))(b2)
#     b2 = keras.layers.GlobalMaxPooling1D()(b2)
#
#     x = keras.layers.Concatenate()([b2, b1])
#     # x = keras.layers.Dense(64, activation='relu')(x)
#     x = keras.layers.Dense(32, activation='relu')(x)
#
#     outputs = keras.layers.Dense(output_dim,
#                                  activation='softmax',
#                                  kernel_regularizer=regularizers.l2(0.01),
#                                  bias_regularizer=regularizers.l2(0.01))(x)
#
#     model = keras.Model(inputs, outputs)
#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=0.001, decay=0.001, clipvalue=0.5, beta_1=0.9, beta_2=0.9),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
#
#     return model
#


def build_3bt(vocab_size, embed_dim, dense_dim, output_dim):
    inputs = keras.Input(shape=(None,), dtype='int64')
    x = keras.layers.Embedding(vocab_size, embed_dim, embeddings_regularizer=regularizers.l2(config.L2))(inputs)
    x = keras.layers.SpatialDropout1D(0.2)(x)
    x = TransformerEncoder(embed_dim, dense_dim, 3)(x)
    x = TransformerEncoder(embed_dim, dense_dim, 3)(x)
    x = TransformerEncoder(embed_dim, dense_dim, 3)(x)
    x = TransformerEncoder(embed_dim, dense_dim, 3)(x)
    x = TransformerEncoder(embed_dim, dense_dim, 3)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.GlobalMaxPooling1D()(x)


    outputs = keras.layers.Dense(output_dim,
                                 activation='softmax',
                                 kernel_regularizer=regularizers.l2(config.L2),
                                 bias_regularizer=regularizers.l2(config.L2))(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNINGRATE, clipvalue=0.5, beta_1=0.9, beta_2=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model
