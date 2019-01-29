from keras import backend as K, optimizers
from keras.models import Sequential, load_model
from keras.layers import RNN, Layer
import numpy as np
from argparse import ArgumentParser
from misc.dicts import load_data_cfg
from misc.files import ensure_can_write
from misc.args import parse_cfg_args
import os
from gen_dataset import generate_dataset
import eval_world_model as evaluation


class S_RNNCell(Layer):
    def __init__(self, units, state_size, self_input, z_dim, a_dim, **kwargs):
        self.units = units
        self.state_size = (units, state_size)
        self.self_input = self_input
        self.output_size = units
        self.z_dim = z_dim
        self.a_dim = a_dim
        super().__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'units': self.units,
            'state_size': self.state_size[1],
            'self_input': self.self_input,
            'z_dim': self.z_dim,
            'a_dim': self.a_dim
        })
        return base_config

    def build(self, input_shape):
        assert len(input_shape) == 2  # Only batch size and a vector dimension
        assert input_shape[1] == (self.z_dim + self.a_dim)
        self.A_weights = self.add_weight(name='A', shape=(self.state_size[1], self.state_size[1]), initializer='uniform')
        self.B_weights = self.add_weight(name='B', shape=(self.state_size[1], self.state_size[1]), initializer='uniform')
        self.C_weights = self.add_weight(name='C', shape=(self.z_dim + (1 if self.self_input else 0), self.state_size[1]), initializer='uniform')
        self.D_weights = self.add_weight(name='D', shape=(self.a_dim, self.state_size[1]), initializer='uniform')
        self.E_weights = self.add_weight(name='E', shape=(self.state_size[1], self.units), initializer='uniform')
        self.bias = self.add_weight(name='bias', shape=(self.state_size[1],), initializer='uniform')
        super().build(input_shape)

    def call(self, inputs, states):
        y_prev = states[0]
        si_prev = states[1]
        z = inputs[:,:self.z_dim]
        if self.self_input:
            z = K.concatenate((z, y_prev), axis=1)
        a = inputs[:,self.z_dim:]
        s = K.tanh(K.dot(si_prev, self.A_weights) + K.dot(z, self.C_weights))
        si = K.tanh(K.dot(s, self.B_weights) + K.dot(a, self.D_weights) + self.bias)
        y = K.dot(si, self.E_weights)
        return y, [y, si]


def load_training_data(cfg, strict_clean, validation_split):
    training_data = generate_dataset(cfg, strict_clean, strict_clean)
    training_input = np.concatenate((training_data['z'], training_data['a']), axis=2)
    training_output = training_data['y']
    shuffled_indices = np.arange(len(training_data))
    np.random.shuffle(shuffled_indices)
    training_input = training_input[shuffled_indices]
    training_output = training_output[shuffled_indices]

    validation_split_i = int(len(training_input) * (1 - validation_split))
    training_input, validation_input = np.split(training_input, [ validation_split_i ])
    training_output, validation_output = np.split(training_output, [ validation_split_i ])
    return training_data, training_input, validation_input, training_output, validation_output


def generate_world_model(cfg, clean = False, strict_clean = False):
    write_to = cfg['model_output_file']
    gen_cfg = cfg['generation']
    learning_cfg = cfg['learning']
    if os.path.isfile(write_to) and not (clean or strict_clean):
        return load_model(write_to, custom_objects={ 'S_RNNCell': S_RNNCell })

    SEED = gen_cfg['seed']
    np.random.seed(SEED)

    VALIDATION_SPLIT = learning_cfg['validation_split']
    assert 0 < VALIDATION_SPLIT and VALIDATION_SPLIT < 1

    training_data, \
    training_input, validation_input, \
    training_output, validation_output = load_training_data(cfg, strict_clean, VALIDATION_SPLIT)

    STATE_DIM = learning_cfg['state_dim']
    SELF_INPUT = learning_cfg['self_input']

    Z_DIM = np.shape(training_data.dtype[0])[1]  # includes self-input
    assert 0 < Z_DIM
    A_DIM = np.shape(training_data.dtype[1])[1]
    assert 0 < A_DIM
    Y_DIM = np.shape(training_data.dtype[2])[1]
    assert 0 < Y_DIM

    UNROLL_PAST = gen_cfg['past_window']
    assert 0 < UNROLL_PAST
    UNROLL_FUTURE = gen_cfg['future_window']
    assert 0 <= UNROLL_FUTURE
    for i in range(len(training_data.dtype)):
        assert np.shape(training_data.dtype[i])[0] == UNROLL_PAST + UNROLL_FUTURE

    cell = S_RNNCell(Y_DIM, STATE_DIM, SELF_INPUT, Z_DIM, A_DIM)
    model = Sequential()
    model.add(RNN(cell, return_sequences=True))

    print('Starting training')
    base_lr = learning_cfg['learning_rate']
    steps = learning_cfg['learning_rate_steps']
    for lr in map(lambda exp: base_lr * (0.5 ** exp), range(steps)):
        opt = optimizers.RMSprop(lr=lr)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_error'])
        model.fit(training_input, training_output,
            verbose=1,
            batch_size=learning_cfg['batch_size'],
            epochs=learning_cfg['epochs']
        )

    evaluation.evaluate_world_model(cfg, model=model, eval_input=validation_input, eval_output=validation_output)

    print('Serializing trained model')
    ensure_can_write(write_to)
    model.save(write_to)

    return model


if __name__ == '__main__':
    generate_world_model(*parse_cfg_args(load_data_cfg))
