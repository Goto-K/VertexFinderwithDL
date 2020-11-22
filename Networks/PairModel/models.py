from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Concatenate
from tensorflow.keras.models import Model


def PairModelStandard(x_train, vertex_train, NODE_DIM=256):

    INPUT_DIM = x_train.shape[1]
    NB_CLASSES = vertex_train.shape[1]

    variable_input = Input(shape=(INPUT_DIM,), name='Pair_Input')

    mid = Dense(NODE_DIM, name='Dense_1')(variable_input)
    mid = BatchNormalization(name='Batch_Normalization_1')(mid)
    mid = Activation('relu', name='Activation_ReLU_1')(mid)
    mid = Dense(NODE_DIM, name='Dense_2')(mid)
    mid = BatchNormalization(name='Batch_Normalization_2')(mid)
    mid = Activation('relu', name='Activation_ReLU_3')(mid)
    mid = Dense(NODE_DIM, name='Dense_3')(mid)
    mid = BatchNormalization(name='Batch_Normalization_3')(mid)

    cla = Dense(NB_CLASSES)(mid)
    vertex_output = Activation('softmax', name='Vertex_Output')(cla)

    position_output = Dense(1, name='Position_Output')(mid)

    model = Model(inputs=variable_input, outputs=[vertex_output, position_output])

    model.summary()

    return model


def PairModelB(x_train, vertex_train, NODE_DIM=256):

    INPUT_DIM = x_train.shape[1]
    NB_CLASSES = vertex_train.shape[1]

    variable_input = Input(shape=(INPUT_DIM,), name='Pair_Input')
    position_input = Input(shape=(1,), name='Position_Input')

    mid = Dense(NODE_DIM, name='Dense_1')(variable_input)
    mid = BatchNormalization(name='Batch_Normalization_1')(mid)
    mid = Activation('relu', name='Activation_ReLU_1')(mid)
    mid = Dense(NODE_DIM, name='Dense_2')(mid)
    mid = BatchNormalization(name='Batch_Normalization_2')(mid)
    mid = Activation('relu', name='Activation_ReLU_3')(mid)
    mid = Dense(NODE_DIM, name='Dense_3')(mid)
    mid = BatchNormalization(name='Batch_Normalization_3')(mid)

    mid = Concatenate()([mid, position_input])

    cla = Dense(NB_CLASSES)(mid)
    vertex_output = Activation('softmax', name='Vertex_Output')(cla)

    model = Model(inputs=[variable_input, position_input], outputs=vertex_output)

    model.summary()

    return model


def PairModelC(x_train, vertex_train, NODE_DIM=256):

    INPUT_DIM = x_train.shape[1]
    NB_CLASSES = vertex_train.shape[1]

    variable_input = Input(shape=(INPUT_DIM,), name='Pair_Input')

    mid = Dense(NODE_DIM, name='Dense_1')(variable_input)
    mid = BatchNormalization(name='Batch_Normalization_1')(mid)
    mid = Activation('relu', name='Activation_ReLU_1')(mid)
    mid = Dense(NODE_DIM, name='Dense_2')(mid)
    mid = BatchNormalization(name='Batch_Normalization_2')(mid)
    mid = Activation('relu', name='Activation_ReLU_3')(mid)
    mid = Dense(NODE_DIM, name='Dense_3')(mid)
    mid = BatchNormalization(name='Batch_Normalization_3')(mid)

    position_output = Dense(1, name='Position_Output')(mid)

    mid = Concatenate()([mid, position_output])
    
    cla = Dense(NB_CLASSES)(mid)
    vertex_output = Activation('softmax', name='Vertex_Output')(cla)

    model = Model(inputs=variable_input, outputs=[vertex_output, position_output])

    model.summary()

