# b1 = keras.layers.GaussianDropout(0.5)(b1)
    # b1 = keras.layers.GaussianNoise(0.2)(b1)
    # b1 = TransformerEncoder(embed_dim, dense_dim, 6)(b1)
    # b1 = keras.layers.GaussianDropout(0.5)(b1)
    # b1 = keras.layers.GaussianNoise(0.2)(b1)
    # b1 = keras.layers.Dense(dense_dim,
    #                         activation='relu',
    #                         kernel_regularizer=regularizers.l2(config.L2),
    #                         bias_regularizer=regularizers.l2(config.L2))(b1)
    # b1 = keras.layers.Dense(dense_dim,
    #                         activation='relu',
    #                         kernel_regularizer=regularizers.l2(config.L2),
    #                         bias_regularizer=regularizers.l2(config.L2))(b1)