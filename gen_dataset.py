import sys
import os  # allow import of IDS from submodule
sys.path.insert(0, os.path.join(os.getcwd(), 'industrialbenchmark/industrial_benchmark_python'))
import IDS
import numpy as np
from itertools import islice
import argparse
from misc.dicts import load_data_cfg
from misc.files import ensure_can_write
from misc.args import parse_cfg_args
from functools import reduce

A_DIM = 3  # IB constant


class BenchmarkGenerator:
    '''
    Generates observable states as well as rewards from applying action vectors
    to the industrial benchmark. Call instances of this class like a function
    to generate them.
    '''
    def __init__(self, z_dim: int, init_setpoint: int, seed: int = None):
        '''
        Parameters
        ----------
        z_dim : int
            Index in range [0, 4]. How many elements of the observable state
            should be included in the output?
        init_setpoint : int
            Initial setpoint of the industrial benchmark,
        seed : int or None
            Seed to be supplied to the industrial benchmark
        '''
        assert 0 <= z_dim and z_dim <= 4
        self.z_dim = z_dim
        self.benchmark = IDS.IDS(p=init_setpoint, inital_seed=seed)

    def get_state(self):
        '''
        Returns
        -------
        np.ndarray
            Array of shape `(self.z_dim,)` holding the current observable state
            of the industrial benchmark
        '''
        return self.benchmark.visibleState()[:self.z_dim]  # p, v, g, h

    def get_rewards(self):
        '''
        Returns
        -------
        np.ndarray
            Array of shape `(2,)` with fuel costs and consumption costs of the
            last tranisition of the industrial benchmark
        '''
        state = self.benchmark.visibleState()
        return np.array(state[4:6])  # f, c

    def __call__(self, action_vector):
        '''
        Apply an action to the industrial benchmark thus transitioning one step.

        Parameters
        ----------
        action_vector : numpy.ndarray
            Array of shape `(3,)` to be applied to the industrial benchmark;
            array must have entries for velocity, gain and shift in this order
            in the range of [-1, 1]

        Returns
        -------
        np.ndarray
            New state
        np.ndarray
            Step rewards
        '''
        assert np.shape(action_vector) == (A_DIM,)
        self.benchmark.step(action_vector)
        return (self.get_state(), self.get_rewards())


class BenchmarkTrajectory:
    '''
    Generates random industrial benchmark trajectories of fixed lengths mapped
    to a set output variable.
    '''
    def __init__(self, z_dim: int, y_index: int, length: int, init_setpoint: int, seed: int = None):
        '''
        Parameters
        ----------
        z_dim : int
            Index in range [0, 4]. How many elements of the observable state
            should be included in the output?
        y_index : int
            Index in range [0, 1]. Which element of the outut should the rewards
            be mapped to?
        length : int
            Length of the trajectory to generate
        init_setpoint : int
            Initial setpoint of the industrial bencharmk
        seed : int or None
            Seed for random number generation
        '''
        self.z_dim = z_dim
        self.y_index = y_index
        self.benchmark = BenchmarkGenerator(z_dim, init_setpoint, seed)
        self.can_supply = length
        if seed is not None:
            np.random.seed(seed)

    def empty(self) -> bool:
        return self.can_supply <= 0

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        '''
        Applies a random action to the benchmark.

        Returns
        -------
        np.ndarray
            New state
        np.ndarray
            Action applied
        float
            Rewards in this step
        '''
        if self.empty():
            raise StopIteration
        actions = 2 * np.random.rand(A_DIM) -1  # elements in [-1, 1]
        s = self.benchmark.get_state()
        _, rewards = self.benchmark(actions)
        self.can_supply -= 1
        return (s, actions, rewards[self.y_index])

    def to_array(self) -> np.ndarray:
        '''
        Turn all steps left in this trajectory into a numpy record array.
        The record will have three columns: `'z'` will holds states, `'a'` holds
        the actions applied to them and `'y'` holds the reward yielded from that
        application.

        Returns
        -------
        np.ndarray
            Trajectory
        '''
        return np.array(list(self), dtype=[
            ('z', 'f4', self.z_dim),
            ('a', 'f4', A_DIM),
            ('y', 'f4', 1)
        ])


def generate_dataset(cfg: dict, clean: bool = False, strict_clean: bool = False) -> np.ndarray:
    '''
    Generate a dataset for a world model to train on. Loads the dataset if at
    the path given in the configuration dict there already is a dataset.

    Data will be provided as a record array with three columns. Each row
    represents a time series to be trained. The length of the times series is
    determined by the config dict. The `'z'` column holds the states at each
    point in time, `'a'` holds the action applied in that point in time and
    `'y'` holds the rewards yielded from that.

    Returns
    -------
    np.ndarray
        Record array of time series
    '''
    write_to = cfg['data_output_file']
    gen_cfg = cfg['generation']
    data_cfg = cfg['data']
    if os.path.isfile(write_to) and not (clean or strict_clean):
        return np.load(write_to)

    SEED = gen_cfg['seed']
    np.random.seed(SEED)

    PAST_LENGTH = gen_cfg['past_window']
    assert 0 < PAST_LENGTH
    FUTURE_LENGTH = gen_cfg['future_window']
    assert 0 <= FUTURE_LENGTH
    WINDOW_LENGTH = PAST_LENGTH + FUTURE_LENGTH

    BLOCKSIZE_MIN = gen_cfg['min_block_size']
    assert 0 < BLOCKSIZE_MIN
    assert WINDOW_LENGTH <= BLOCKSIZE_MIN
    BLOCKSIZE_MAX = gen_cfg['max_block_size']
    assert BLOCKSIZE_MIN <= BLOCKSIZE_MAX
    def random_blocksize():
        return int(BLOCKSIZE_MIN + (BLOCKSIZE_MAX - BLOCKSIZE_MIN) * np.random.rand())

    HYPERVARS = gen_cfg['init_setpoints']
    N_HYPERVARS = len(HYPERVARS)
    assert 0 < N_HYPERVARS
    TRAJECTORIES = gen_cfg['trajectories_num']
    assert 0 < TRAJECTORIES
    TRAJECTORY_LENGTH = gen_cfg['trajectories_length']
    assert 0 < TRAJECTORY_LENGTH

    Z_DIM = data_cfg['z_dim']
    assert 0 < Z_DIM

    OUTPUT_FUEL = data_cfg['output_fuel']
    OUTPUT_CONSUMPTION = data_cfg['output_consumption']
    assert OUTPUT_FUEL or OUTPUT_CONSUMPTION, 'at least one output must be given'
    assert not OUTPUT_FUEL or not OUTPUT_CONSUMPTION, 'only one output is supported'

    # Time series will be generated on blocks. We generate a trajectory and
    # split it into blocks of random size. Than we use a method of sliding
    # window indices to get all the time series on every block. Pre-calculate
    # all block sizes here. Details to this approach can be found in Duell,
    # Udluft, Sterzing, 2012, chapter 29.3.4.
    block_sizes = [ [ [] for _ in range(TRAJECTORIES) ] for _ in range(N_HYPERVARS) ]
    global_windows_num = 0
    for hypervar_trajectories in block_sizes:
        for trajectory_sizes in hypervar_trajectories:
            datapoints_num = TRAJECTORY_LENGTH
            block_size = random_blocksize()
            while 0 < datapoints_num - block_size:
                trajectory_sizes.append(block_size)
                global_windows_num += block_size - (PAST_LENGTH + FUTURE_LENGTH) + 1
                datapoints_num -= block_size
                block_size = random_blocksize()

    # Output array
    data_blocks = np.empty(
        (global_windows_num,), dtype=[
            ('z', 'f4', (WINDOW_LENGTH, Z_DIM)),
            ('a', 'f4', (WINDOW_LENGTH, A_DIM)),
            ('y', 'f4', (WINDOW_LENGTH, 1))
        ]
    )

    block_i = 0
    for i, h_num in enumerate(HYPERVARS):
        for j in range(TRAJECTORIES):
            env = BenchmarkTrajectory(Z_DIM, (0 if OUTPUT_FUEL else 1), TRAJECTORY_LENGTH, h_num, SEED)
            blocks = env.to_array()

            slice_i = 0
            for blocksize in block_sizes[i][j]:
                # get block-many data points
                these_blocks = blocks[slice_i:slice_i + blocksize]
                slice_i += blocksize
                z_inputs = these_blocks['z']
                a_inputs = these_blocks['a']
                outputs = these_blocks['y'].reshape(-1,1)

                if len(z_inputs) < WINDOW_LENGTH:
                    break
                # Number of time series in this iteration
                windows_num = min(len(z_inputs), blocksize) - WINDOW_LENGTH +1

                # Create index matrix for windows
                windows = np.arange(WINDOW_LENGTH) + np.arange(windows_num).reshape(windows_num, 1)
                p_windows, f_windows = np.hsplit(windows, [PAST_LENGTH])
                z_windows = np.append(z_inputs[p_windows], np.zeros(z_inputs.shape)[f_windows], axis=1)
                a_windows = a_inputs[windows]

                output_windows = outputs[windows]

                for k in range(windows_num):
                    data_blocks[block_i] = (z_windows[k], a_windows[k], output_windows[k])
                    block_i += 1

    assert block_i == global_windows_num

    ensure_can_write(write_to)
    np.save(write_to, data_blocks)

    return data_blocks

if __name__ == '__main__':
    generate_dataset(*parse_cfg_args(load_data_cfg))
