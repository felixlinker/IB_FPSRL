from typing import Tuple
from functools import reduce
from keras import layers, backend
from keras.models import Model, load_model
from keras.layers import Dense, Input, ZeroPadding1D
from tensorflow import Tensor
import numpy as np
from argparse import ArgumentParser
from misc.dicts import load_cfg
import os

def generate_world_model(cfg: dict, clean: bool = False) -> Model:
    write_to = cfg['model_output_file']
    if os.path.isfile(write_to) and not clean:
        return load_model(write_to)

    training_data = np.load(cfg['data_output_file'])
    training_input = [np.zeros((len(training_data), 1))] + [ e for t in zip(
        np.swapaxes(training_data['z'], 0, 1),
        np.swapaxes(training_data['a'], 0, 1)
    ) for e in t ]
    training_output = list(np.swapaxes(training_data['y'], 0, 1))

    Z_DIM = np.shape(training_data.dtype[0])[1]
    assert 0 < Z_DIM
    A_DIM = np.shape(training_data.dtype[1])[1]
    assert 0 < A_DIM

    UNROLL_PAST = cfg['past_window']
    assert 0 < UNROLL_PAST
    UNROLL_FUTURE = cfg['future_window']
    assert 0 <= UNROLL_FUTURE
    for i in range(3):
        assert np.shape(training_data.dtype[i])[0] == UNROLL_PAST + UNROLL_FUTURE

    # note: paper doesn't mention specific activation functions
    s_t_layer = Dense(1, activation='sigmoid')
    si_t_layer= Dense(1, activation='sigmoid')
    y_t_layer = Dense(1, activation='sigmoid')

    def build_layer_block(s_t_input: Tensor, z_t_input: Tensor, a_t_input: Tensor) -> Tensor:
        s_t = s_t_layer(layers.concatenate([s_t_input, z_t_input]))
        si_t = si_t_layer(layers.concatenate([s_t, a_t_input]))
        y_t = y_t_layer(si_t)
        return (y_t, si_t)

    PLayers = Tuple[list, list, Tensor]
    def red_p_layers(i_list: list, o_list: list, o: Tensor) -> PLayers:
        z_t_input = Input(shape=(Z_DIM,))
        a_t_input = Input(shape=(A_DIM,))
        i_list.extend([z_t_input, a_t_input])
        y_t, si_t = build_layer_block(o, z_t_input, a_t_input)
        o_list.append(y_t)
        return (i_list, o_list, si_t)

    initial_input = Input(shape=(1,))
    (inputs, outputs, s0) = reduce(
        lambda aggregator, _: red_p_layers(*aggregator),
        range(UNROLL_PAST),
        ([initial_input], [], initial_input)
    )

    FLayers = Tuple[list, list, Tensor]
    def red_f_layers(i_list: list, o_list: list, o: Tensor) -> FLayers:
        z_t_input = Input(shape=(Z_DIM,))
        a_t_input = Input(shape=(3,))
        i_list.extend([z_t_input, a_t_input])
        y_t, si_t = build_layer_block(o, z_t_input, a_t_input)
        o_list.append(y_t)
        return (i_list, o_list, si_t)

    (inputs, outputs, _) = reduce(
        lambda aggregator, _: red_f_layers(*aggregator),
        range(UNROLL_FUTURE),
        (inputs, outputs, s0)
    )

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='sgd', loss='mean_squared_error')
    print("Starting training")
    model.fit(training_input, training_output, verbose=1)
    print("Serializing trained model")

    dirs, _ = os.path.split(write_to)
    try:
        os.makedirs(dirs)
    except FileExistsError:
        pass
    model.save(write_to)

    return model

if __name__ == "__main__":
    parser = ArgumentParser(description='Create and train a neural net for a world model')
    parser.add_argument('cfg_file')
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg_file)
    generate_world_model(cfg, args.clean)
