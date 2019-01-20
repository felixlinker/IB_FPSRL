import sys
import os  # allow import of IDS from submodule
sys.path.insert(0, os.path.join(os.getcwd(), 'industrialbenchmark\\industrial_benchmark_python'))
import IDS
import numpy as np
from itertools import islice
import argparse
import json


class BenchmarkTrajectory:
    def __init__(self, length: int, hypervars: int, seed: int):
        self.benchmark = IDS.IDS(p=hypervars, inital_seed=seed)
        self.can_supply = length

    def empty(self) -> bool:
        return self.can_supply <= 0

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self.empty():
            raise StopIteration
        actions = 2 * np.random.rand(3) -1  # elements in [-1, 1]
        s = self.get_setpoint()
        self.benchmark.step(actions)
        self.can_supply -= 1
        return np.concatenate(([s], actions, self.get_rewards()))

    def get_setpoint(self) -> float:
        return self.benchmark.visibleState()[0]

    def get_rewards(self) -> np.ndarray:
        state = self.benchmark.visibleState()
        return np.array(state[4:6])  # f, c


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a .npz file containing training data blocks')
    parser.add_argument('cfg_file')
    args = parser.parse_args()

    with open(args.cfg_file) as fp:
        cfg = json.load(fp)

    PAST_LENGTH = cfg.get('past_window', 30)
    assert 0 < PAST_LENGTH
    FUTURE_LENGTH = cfg.get('future_window', 30)
    assert 0 < FUTURE_LENGTH
    WINDOW_LENGTH = PAST_LENGTH + FUTURE_LENGTH

    BLOCKSIZE_MIN = cfg.get('min_block_size', 90)
    assert 0 < BLOCKSIZE_MIN
    assert WINDOW_LENGTH <= BLOCKSIZE_MIN
    BLOCKSIZE_MAX = cfg.get('max_block_size', 120)
    assert BLOCKSIZE_MIN <= BLOCKSIZE_MAX
    def random_blocksize():
        return int(BLOCKSIZE_MIN + (BLOCKSIZE_MAX - BLOCKSIZE_MIN) * np.random.rand())

    HYPERVARS = cfg.get('init_setpoints', range(10, 101, 10))
    N_HYPERVARS = len(HYPERVARS)
    assert 0 < N_HYPERVARS
    TRAJECTORIES = cfg.get('trajectories_num', 10)
    assert 0 < TRAJECTORIES
    TRAJECTORY_LENGTH = cfg.get('trajectories_length', 1000)
    assert 0 < TRAJECTORY_LENGTH

    OUTPUT_FUEL = cfg.get('output_fuel', False)
    OUTPUT_CONSUMPTION = cfg.get('output_consumption', False)
    assert OUTPUT_FUEL or OUTPUT_CONSUMPTION, 'at least one output must be given'
    assert not OUTPUT_FUEL or not OUTPUT_CONSUMPTION, 'only one output is supported'

    SELF_INPUT_FUEL = cfg.get('self_input_fuel', False)
    SELF_INPUT_CONSUMPTION = cfg.get('self_input_consumption', False)

    SEED = cfg['seed']

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
            ('z', 'f4', (WINDOW_LENGTH, 2)),
            ('a', 'f4', (WINDOW_LENGTH, 3)),
            ('y', 'f4', (WINDOW_LENGTH, 1))
        ]
    )

    block_i = 0
    for i, h_num in enumerate(HYPERVARS):
        for j in range(TRAJECTORIES):
            env = BenchmarkTrajectory(TRAJECTORY_LENGTH, h_num, SEED)

            for blocksize in block_sizes[i][j]:
                data_block = np.array(list(islice(env, blocksize)))  # get block-many data points
                inputs, outputs = np.hsplit(data_block, [4])  # split at index for f

                if len(data_block) < WINDOW_LENGTH:
                    break
                windows_num = min(len(data_block), blocksize) - WINDOW_LENGTH +1

                # Create index matrix for windows
                windows = np.arange(WINDOW_LENGTH) + np.arange(windows_num).reshape(windows_num, 1)
                p_windows, f_windows = np.hsplit(windows, [PAST_LENGTH])
                p_windows = inputs[p_windows]
                p_z_windows, p_a_windows = np.split(p_windows, [1], axis=2)
                f_windows = inputs[f_windows]
                f_z_windows = np.zeros((windows_num, FUTURE_LENGTH, 1))
                _, f_a_windows = np.split(f_windows, [1], axis=2)
                z_windows = np.append(p_z_windows, f_z_windows, axis=1)
                a_windows = np.append(p_a_windows, f_a_windows, axis=1)

                output_windows = outputs[windows]
                fuel_output, consumption_output = np.split(output_windows, 2, axis=2)
                if OUTPUT_FUEL:
                    output_windows = fuel_output
                    to_insert = 0 if not SELF_INPUT_FUEL else fuel_output.reshape(windows_num, WINDOW_LENGTH)
                    z_windows = np.insert(z_windows, 1, to_insert, axis=2)
                elif OUTPUT_CONSUMPTION:
                    output_windows = consumption_output
                    to_insert = 0 if not SELF_INPUT_CONSUMPTION else consumption_output.reshape(windows_num, WINDOW_LENGTH)
                    z_windows = np.insert(z_windows, 1, to_insert, axis=2)

                for k in range(windows_num):
                    data_blocks[block_i] = (z_windows[k], a_windows[k], output_windows[k])
                    block_i += 1

    assert block_i == global_windows_num

    write_to = cfg['data_output_file']
    dirs, _ = os.path.split(write_to)
    try:
        os.makedirs(dirs)
    except FileExistsError:
        pass
    np.save(write_to, data_blocks)
