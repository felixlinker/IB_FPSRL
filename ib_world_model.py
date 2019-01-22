from keras import backend as K, optimizers
from keras.models import Sequential, load_model
from keras.layers import RNN, Layer
import numpy as np
from argparse import ArgumentParser
from misc.dicts import load_cfg
import os


class S_RNNCell(Layer):
    def __init__(self, units, state_size, z_dim, a_dim, **kwargs):
        self.units = units
        self.state_size = state_size
        self.output_size = units
        self.z_dim = z_dim
        self.a_dim = a_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2  # Only batch size and a vector dimension
        assert input_shape[1] == (self.z_dim + self.a_dim)
        self.A_weights = self.add_weight(name='A', shape=(self.state_size, self.state_size), initializer='uniform')
        self.B_weights = self.add_weight(name='B', shape=(self.state_size, self.state_size), initializer='uniform')
        self.C_weights = self.add_weight(name='C', shape=(self.z_dim, self.state_size), initializer='uniform')
        self.D_weights = self.add_weight(name='D', shape=(self.a_dim, self.state_size), initializer='uniform')
        self.E_weights = self.add_weight(name='E', shape=(self.state_size, self.units), initializer='uniform')
        super().build(input_shape)

    def call(self, inputs, states):
        si_prev = states[0]
        z = inputs[:,:self.z_dim]
        a = inputs[:,self.z_dim:]
        s = K.tanh(K.dot(si_prev, self.A_weights) + K.dot(z, self.C_weights))
        si = K.tanh(K.dot(s, self.B_weights) + K.dot(a, self.D_weights))
        y = K.dot(si, self.E_weights)
        return y, [si]


def generate_world_model(cfg, clean = False):
    write_to = cfg['model_output_file']
    if os.path.isfile(write_to) and not clean:
        return load_model(write_to)

    training_data = np.load(cfg['data_output_file'])
    training_input = np.concatenate((training_data['z'], training_data['a']), axis=2)
    training_output = training_data['y']

    STATE_DIM = cfg['state_dim']

    Z_DIM = np.shape(training_data.dtype[0])[1]  # includes self-input
    assert 0 < Z_DIM
    A_DIM = np.shape(training_data.dtype[1])[1]
    assert 0 < A_DIM
    Y_DIM = np.shape(training_data.dtype[2])[1]
    assert 0 < Y_DIM

    UNROLL_PAST = cfg['past_window']
    assert 0 < UNROLL_PAST
    UNROLL_FUTURE = cfg['future_window']
    assert 0 <= UNROLL_FUTURE
    for i in range(len(training_data.dtype)):
        assert np.shape(training_data.dtype[i])[0] == UNROLL_PAST + UNROLL_FUTURE

    cell = S_RNNCell(Y_DIM, STATE_DIM, Z_DIM, A_DIM)
    model = Sequential()
    model.add(RNN(cell, return_sequences=True))

    print('Starting training')
    for lr in map(lambda exp: 0.05 * (0.5 ** exp), range(10)):
        opt = optimizers.RMSprop(lr=lr)
        model.compile(optimizer=opt, loss='mean_squared_error')
        model.fit(training_input, training_output, verbose=1)

    print('Serializing trained model')
    dirs, _ = os.path.split(write_to)
    try:
        os.makedirs(dirs)
    except FileExistsError:
        pass
    model.save(write_to)

    return model


if __name__ == '__main__':
    parser = ArgumentParser(description='Create and train a neural net for a world model')
    parser.add_argument('cfg_file')
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg_file)
    generate_world_model(cfg, args.clean)
