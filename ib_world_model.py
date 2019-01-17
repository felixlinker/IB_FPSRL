from typing import Tuple
from functools import reduce
from keras import layers, backend
from keras.models import Model
from keras.layers import Dense, Input, ZeroPadding1D
import numpy
from argparse import ArgumentParser
import json

parser = ArgumentParser(description='Create and train a neural net for a world model')
parser.add_argument('cfg_file')
args = parser.parse_args()

with open(args.cfg_file) as fp:
    cfg = json.load(fp)

Z_DIM = cfg['z_dim']  # > 1 assumed
A_DIM = cfg['a_dim']  # > 1 assumed

UNROLL_PAST = cfg['unroll_p']  # > 1 assumed
UNROLL_FUTURE = cfg['unroll_f']  # > 1 assumed

# note: paper doesn't mention specific activation functions
s_t_layer = Dense(1, activation='sigmoid')
si_t_layer= Dense(1, activation='sigmoid')
y_t_layer = Dense(1, activation='sigmoid')

PLayers = Tuple[list, list, Dense]
def red_p_layers(i_list: list, o_list: list, o: Dense) -> PLayers:
    z_t_input = Input(shape=(Z_DIM,))
    i_list.append(z_t_input)
    z_t_input = layers.concatenate([o, z_t_input])
    s_t = s_t_layer(z_t_input)
    a_t_input = Input(shape=(A_DIM,))
    i_list.append(a_t_input)
    si_t = si_t_layer(layers.concatenate([s_t, a_t_input]))
    y_t = y_t_layer(si_t)
    o_list.append(y_t)
    return (i_list, o_list, si_t)

initial_input = Input(shape=(1,))
(inputs, outputs, s0) = reduce(
    lambda aggregator, _: red_p_layers(*aggregator),
    range(UNROLL_PAST),
    ([initial_input], [], initial_input)
)

FLayers = Tuple[list, list, Dense]
def red_f_layers(i_list: list, o_list: list, o: Dense) -> FLayers:
    z_t_input = Input(shape=(Z_DIM,))
    i_list.append(z_t_input)
    z_t_input = layers.concatenate([o, z_t_input])
    s_t = s_t_layer(z_t_input)
    a_t_input = Input(shape=(3,))
    i_list.append(a_t_input)
    si_t = si_t_layer(layers.concatenate([s_t, a_t_input]))
    y_t = y_t_layer(si_t)
    o_list.append(y_t)
    return (i_list, o_list, si_t)

(inputs, outputs, _) = reduce(
    lambda aggregator, _: red_f_layers(*aggregator),
    range(UNROLL_FUTURE),
    (inputs, outputs, s0)
)

model = Model(inputs=inputs, outputs=outputs)
