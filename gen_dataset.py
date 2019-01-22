import sys
import os  # allow import of IDS from submodule
sys.path.insert(0, os.path.join(os.getcwd(), 'industrialbenchmark/industrial_benchmark_python'))
import IDS
import numpy as np
from itertools import islice
import argparse
from misc.dicts import load_cfg

class BenchmarkTrajectory:
    z_dim = 4
    a_dim = 3
    y_dim = 2

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
        actions = 2 * np.random.rand(self.a_dim) -1  # elements in [-1, 1]
        s = self.get_setpoint()
        self.benchmark.step(actions)
        self.can_supply -= 1
        return np.concatenate((s, actions, self.get_rewards()))

    def get_setpoint(self) -> float:
        return self.benchmark.visibleState()[:self.z_dim]  # p, v, g, h

    def get_rewards(self) -> np.ndarray:
        state = self.benchmark.visibleState()
        return np.array(state[4:4 + self.y_dim])  # f, c


def generate_dataset(cfg: dict, clean: bool = False) -> np.ndarray:
    write_to = cfg['data_output_file']
    if os.path.isfile(write_to) and not clean:
        return np.load(write_to)

    PAST_LENGTH = cfg['past_window']
    assert 0 < PAST_LENGTH
    FUTURE_LENGTH = cfg['future_window']
    assert 0 <= FUTURE_LENGTH
    WINDOW_LENGTH = PAST_LENGTH + FUTURE_LENGTH

    BLOCKSIZE_MIN = cfg['min_block_size']
    assert 0 < BLOCKSIZE_MIN
    assert WINDOW_LENGTH <= BLOCKSIZE_MIN
    BLOCKSIZE_MAX = cfg['max_block_size']
    assert BLOCKSIZE_MIN <= BLOCKSIZE_MAX
    def random_blocksize():
        return int(BLOCKSIZE_MIN + (BLOCKSIZE_MAX - BLOCKSIZE_MIN) * np.random.rand())

    HYPERVARS = cfg['init_setpoints']
    N_HYPERVARS = len(HYPERVARS)
    assert 0 < N_HYPERVARS
    TRAJECTORIES = cfg['trajectories_num']
    assert 0 < TRAJECTORIES
    TRAJECTORY_LENGTH = cfg['trajectories_length']
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

    has_self_input = (OUTPUT_FUEL and SELF_INPUT_FUEL) or (OUTPUT_CONSUMPTION and SELF_INPUT_CONSUMPTION)
    data_blocks = np.empty(
        (global_windows_num,), dtype=[
            ('z', 'f4', (WINDOW_LENGTH, BenchmarkTrajectory.z_dim + (1 if has_self_input else 0))),
            ('a', 'f4', (WINDOW_LENGTH, BenchmarkTrajectory.a_dim)),
            ('y', 'f4', (WINDOW_LENGTH, 1))
        ]
    )

    block_i = 0
    for i, h_num in enumerate(HYPERVARS):
        for j in range(TRAJECTORIES):
            env = BenchmarkTrajectory(TRAJECTORY_LENGTH, h_num, SEED)

            for blocksize in block_sizes[i][j]:
                data_block = np.array(list(islice(env, blocksize)))  # get block-many data points
                inputs, outputs = np.hsplit(data_block, [env.z_dim + env.a_dim])  # split at index for f

                if len(data_block) < WINDOW_LENGTH:
                    break
                windows_num = min(len(data_block), blocksize) - WINDOW_LENGTH +1

                # Create index matrix for windows
                windows = np.arange(WINDOW_LENGTH) + np.arange(windows_num).reshape(windows_num, 1)
                p_windows, f_windows = np.hsplit(windows, [PAST_LENGTH])
                p_windows = inputs[p_windows]
                p_z_windows, p_a_windows = np.split(p_windows, [env.z_dim], axis=2)
                f_windows = inputs[f_windows]
                f_z_windows = np.zeros((windows_num, FUTURE_LENGTH, env.z_dim))
                _, f_a_windows = np.split(f_windows, [env.z_dim], axis=2)
                z_windows = np.append(p_z_windows, f_z_windows, axis=1)
                a_windows = np.append(p_a_windows, f_a_windows, axis=1)

                output_windows = outputs[windows]
                fuel_output, consumption_output = np.split(output_windows, 2, axis=2)
                if OUTPUT_FUEL:
                    output_windows = fuel_output
                    if SELF_INPUT_FUEL:
                        z_windows = np.insert(
                            z_windows,
                            env.z_dim,
                            fuel_output.reshape(windows_num, WINDOW_LENGTH),
                            axis=2
                        )
                elif OUTPUT_CONSUMPTION:
                    output_windows = consumption_output
                    if SELF_INPUT_FUEL:
                        z_windows = np.insert(
                            z_windows,
                            env.z_dim,
                            consumption_output.reshape(windows_num, WINDOW_LENGTH),
                            axis=2
                        )

                for k in range(windows_num):
                    data_blocks[block_i] = (z_windows[k], a_windows[k], output_windows[k])
                    block_i += 1

    assert block_i == global_windows_num

    dirs, _ = os.path.split(write_to)
    try:
        os.makedirs(dirs)
    except FileExistsError:
        pass
    np.save(write_to, data_blocks)

    return data_blocks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a .npz file containing training data blocks')
    parser.add_argument('cfg_file')
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg_file)
    generate_dataset(cfg, args.clean)
