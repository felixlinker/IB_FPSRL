import sys
import os  # allow import of IDS from submodule
sys.path.insert(0, os.path.join(os.getcwd(), 'industrialbenchmark/industrial_benchmark_python'))
import IDS
import numpy as np
from itertools import islice
import argparse
from misc.dicts import load_cfg
from misc.files import ensure_can_write
from functools import reduce

A_DIM = 3  # IB constant


class BenchmarkTrajectory:
    def __init__(self, z_dim: int, length: int, hypervars: int, seed: int):
        self.z_dim = z_dim
        self.benchmark = IDS.IDS(p=hypervars, inital_seed=seed)
        self.can_supply = length
        np.random.seed(seed)

    def empty(self) -> bool:
        return self.can_supply <= 0

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self.empty():
            raise StopIteration
        actions = 2 * np.random.rand(A_DIM) -1  # elements in [-1, 1]
        s = self.get_setpoint()
        self.benchmark.step(actions)
        self.can_supply -= 1
        return (s, actions, self.get_rewards())
        return np.concatenate((s, actions, self.get_rewards()))

    def get_setpoint(self) -> float:
        return self.benchmark.visibleState()[:self.z_dim]  # p, v, g, h

    def get_rewards(self) -> np.ndarray:
        state = self.benchmark.visibleState()
        return np.array(state[4:6])  # f, c


def generate_dataset(cfg: dict, clean: bool = False) -> np.ndarray:
    write_to = cfg['data_output_file']
    gen_cfg = cfg['generation']
    data_cfg = cfg['data']
    if os.path.isfile(write_to) and not clean:
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
            env = BenchmarkTrajectory(Z_DIM, TRAJECTORY_LENGTH, h_num, SEED)

            for blocksize in block_sizes[i][j]:
                data_block = islice(env, blocksize)  # get block-many data points
                z_inputs, a_inputs, outputs = map(np.array, reduce(
                    lambda aggregator, t: (aggregator[0] + [t[0]], aggregator[1] + [t[1]], aggregator[2] + [t[2]]),
                    data_block,
                    ([], [], [])
                ))
                outputs = outputs[:,(0 if OUTPUT_FUEL else 1)].reshape(len(outputs), 1)  # else: OUTPUT_CONSUMPTION

                if len(z_inputs) < WINDOW_LENGTH:
                    break
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
    parser = argparse.ArgumentParser(description='Create a .npz file containing training data blocks')
    parser.add_argument('cfg_file')
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg_file)
    generate_dataset(cfg, args.clean)
