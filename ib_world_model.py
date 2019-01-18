from typing import Tuple
from functools import reduce
from keras import layers, backend
from keras.models import Model
from keras.layers import Dense, Input, ZeroPadding1D
from tensorflow import Tensor
import numpy
from argparse import ArgumentParser
import json

parser = ArgumentParser(description='Create and train a neural net for a world model')
parser.add_argument('cfg_file')
args = parser.parse_args()

with open(args.cfg_file) as fp:
    cfg = json.load(fp)

Z_DIM = cfg['z_dim']
assert 0 < Z_DIM
A_DIM = cfg['a_dim']
assert 0 < A_DIM

UNROLL_PAST = cfg['unroll_p']
assert 1 < UNROLL_PAST
UNROLL_FUTURE = cfg['unroll_f']
assert 1 < UNROLL_FUTURE

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
