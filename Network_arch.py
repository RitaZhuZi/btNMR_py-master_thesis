def build_model_fidnet(num_blocks = num_blocks, num_filters = num_filters, input_shape=(max_ns*2,3), dilations=dilations):
    def waveLayer(x,num_filters,dil):
        pad = dil * (kernel_size - 1)
        x_pad = tf.pad(x, [[0,0],[0,pad],[0,0]])
        x1 = keras.layers.Conv1D(filters = num_filters, kernel_size=kernel_size,
        padding="valid", dilation_rate=dil )(x_pad)
        x2 = keras.layers.Conv1D(filters = num_filters, kernel_size=kernel_size,
        padding="valid", dilation_rate=dil )(x_pad)
        x1 = keras.layers.Activation('tanh')(x1)
        x2 = keras.layers.Activation('sigmoid')(x2)
        z = x1*x2
        z = tf.pad(z, [[0,0],[0,pad_out],[0,0]])
        z = keras.layers.Conv1D(filters = num_filters*2,
        kernel_size=kernel_size, padding="valid")(z)
        z1, z2, z3 = keras.layers.Add()([z, tf.expand_dims(x[:,:,0], axis=-1)]), keras.layers.Add()([z, tf.expand_dims(x[:,:,1], axis=-1)]), keras.layers.Add()([z, tf.expand_dims(x[:,:,2], axis=-1)])
        return keras.layers.concatenate([z1,z2,z3], axis=-1), z
    dilations = dilations
    pad_out = kernel_size - 1
    input = keras.layers.Input(shape=input_shape)
    x = input
    skips = []
    for dil in dilations*num_blocks:
        x, skip = waveLayer(x, num_filters, dil)
        skips.append(skip)
    x = keras.layers.Activation('relu')(keras.layers.Add()(skips))
    x = tf.pad(x, [[0,0],[0,pad_out],[0,0]])
    x = keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size,
    padding="valid", activation='relu')(x)
    x = tf.pad(x, [[0,0],[0,pad_out],[0,0]])
    fin = keras.layers.Conv1D(filters=2, kernel_size=kernel_size,
    padding="valid", activation=modified_tanh)(x)
    model = keras.Model(inputs=input, outputs=fin)
    model.compile(optimizer=optimizer,
                  loss=loss_mse)
    return model

