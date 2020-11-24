from tensorflow.keras.layers import Input, Dense, RNN, Activation, TimeDistributed, BatchNormalization, Reshape, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope

from . import layer


def VLSTMModelSimple(pair, tracks, UNITS=256):

    MAX_TRACK_NUM = tracks.shape[1]
    INPUT_DIM = tracks.shape[2]
    PAIR_DIM = pair.shape[1]

    pair_input = Input(shape=(PAIR_DIM,), name='Pair_Input')

    track_input = Input(shape=(None, INPUT_DIM), name='Input')
    track_embedd = TimeDistributed(Dense(UNITS, name='Embedding_Dense', activation='relu'))(encoder_input)

    init_state = Dense(UNITS, name='Init_State_Dense_1')(pair_input)
    init_state = BatchNormalization(name='Init_State_BatchNorm_1')(init_state)
    init_state = Activation('relu', name='Init_State_Activation_1')(init_state)
    init_state = Dense(UNITS, name='Init_State_Dense_2')(init_state)
    init_state = BatchNormalization(name='Init_State_BatchNorm_2')(init_state)
    init_state = Activation('relu', name='Init_State_Activation_2')(init_state)

    cell = layer.VLSTMCellSimple(UNITS, 1)

    rnn = RNN(cell, return_sequences=True, name='Vertex_LSTM_Simple')(track_embedd, initial_state=[init_state, init_state])
    
    model = Model(inputs=[pair_input, track_input], outputs=rnn)

    model.summary()

    return model


def AttentionVLSTMModel(pair, tracks, ENCODER_UNITS=256, DECODER_UNITS=256):

    MAX_TRACK_NUM = tracks.shape[1]
    INPUT_DIM = tracks.shape[2]
    PAIR_DIM = pair.shape[1]

    pair_input = Input(shape=(PAIR_DIM,), name='Pair_Input')

    encoder_input = Input(shape=(MAX_TRACK_NUM, INPUT_DIM), name='Encoder_Input')
    encoder_embedd = TimeDistributed(Dense(ENCODER_UNITS, name='Encoder_Embedding_Dense', activation='relu'))(encoder_input)

    decoder_input = Input(shape=(None, INPUT_DIM), name='Decoder_Input')
    decoder_embedd = TimeDistributed(Dense(DECODER_UNITS, name='Decoder_Embedding_Dense', activation='relu'))(decoder_input)

    encoder_init_state_f = Dense(ENCODER_UNITS, name='Encoder_Forward_Dense_1')(pair_input)
    ecnoder_init_state_f = BatchNormalization(name='Encoder_Forward_BatchNorm_1')(encoder_init_state_f)
    encoder_init_state_f = Activation('relu', name='Encoder_Forward_Activation_1')(encoder_init_state_f)
    encoder_init_state_f = Dense(ENCODER_UNITS, name='Encoder_Forward_Dense_2')(encoder_init_state_f)
    ecnoder_init_state_f = BatchNormalization(name='Encoder_Forward_BatchNorm_2')(encoder_init_state_f)
    encoder_init_state_f = Activation('relu', name='Encoder_Forward_Activation_2')(encoder_init_state_f)

    encoder_init_state_b = Dense(ENCODER_UNITS, name='Encoder_Backward_Dense_1')(pair_input)
    ecnoder_init_state_b = BatchNormalization(name='Encoder_Backward_BatchNorm_1')(encoder_init_state_b)
    encoder_init_state_b = Activation('relu', name='Encoder_Backward_Activation_1')(encoder_init_state_b)
    encoder_init_state_b = Dense(ENCODER_UNITS, name='Encoder_Backward_Dense_2')(encoder_init_state_b)
    ecnoder_init_state_b = BatchNormalization(name='Encoder_Backward_BatchNorm_2')(encoder_init_state_b)
    encoder_init_state_b = Activation('relu', name='Encoder_Backward_Activation_2')(encoder_init_state_b)

    decoder_init_state = Dense(DECODER_UNITS, name='Decoder_Dense_1')(pair_input)
    denoder_init_state = BatchNormalization(name='Decoder_BatchNorm_1')(decoder_init_state)
    decoder_init_state = Activation('relu', name='Decoder_Activation_1')(decoder_init_state)
    decoder_init_state = Dense(DECODER_UNITS, name='Decoder_Dense_2')(decoder_init_state)
    denoder_init_state = BatchNormalization(name='Decoder_BatchNorm_2')(decoder_init_state)
    decoder_init_state = Activation('relu', name='Decoder_Activation_2')(decoder_init_state)

    vlstm_cell_f = layer.VLSTMCellEncoder(ENCODER_UNITS)
    vlstm_cell_b = layer.VLSTMCellEncoder(ENCODER_UNITS)
    encoder_f = RNN(vlstm_cell_f, return_sequences=True, name="Encoder_Forward_VLSTM", go_backwards=False)
    encoder_b = RNN(vlstm_cell_b, return_sequences=True, name="Encoder_Backward_VLSTM", go_backwards=True)

    with CustomObjectScope({"VLSTMCellEncoder": layer.VLSTMCellEncoder}):
        biencoder = Bidirectional(encoder_f, backward_layer=encoder_b)(encoder_embedd, initial_state=[encoder_init_state_f, encoder_init_state_f, encoder_init_state_b, encoder_init_state_b])

    biencoder = Reshape(target_shape=(MAX_TRACK_NUM*ENCODER_UNITS*2,))(biencoder)

    # DCODER_UNITS, ENCODER_UNITS, DECODER_OUTPUT, MAX_TRACK_NUM
    attentionvlstm_cell = layer.AttentionVLSTMCell(DECODER_UNITS, ENCODER_UNITS*2, 1, MAX_TRACK_NUM)
    
    decoder, attention = RNN(attentionvlstm_cell, return_sequences=True, name='Decoder_Attention_VLSTM')(decoder_embedd, initial_state=[biencoder, decoder_init_state])
    
    model = Model(inputs=[pair_input, encoder_input, decoder_input], outputs=decoder)

    model.summary()

    return model

