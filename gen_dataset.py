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


if __name__ == "__main__":
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

    OUTPUT_FUEL = cfg['output_fuel']
    OUTPUT_COST = cfg['output_cost']
    assert OUTPUT_FUEL or OUTPUT_COST  # we need at least one output

    SEED = cfg.get('seed', None)

    block_num = 0
    data_blocks = {}
    for i, h_num in enumerate(HYPERVARS):
        for k in range(TRAJECTORIES):
            env = BenchmarkTrajectory(TRAJECTORY_LENGTH, h_num, SEED)

            while not env.empty():
                blocksize = random_blocksize()
                data_block = np.array(list(islice(env, blocksize)))  # get block-many data points
                inputs, outputs = np.hsplit(data_block, [4])  # split at index for f

                if len(data_block) < WINDOW_LENGTH:
                    break
                windows_num = min(len(data_block), blocksize) - WINDOW_LENGTH +1

                # Create index matrix for windows
                windows = np.arange(WINDOW_LENGTH) + np.arange(windows_num).reshape(windows_num, 1)
                p_windows, f_windows = np.hsplit(windows, [PAST_LENGTH])
                p_windows = inputs[p_windows]
                p_windows = p_windows.reshape(len(p_windows), -1)  # flatten inner lists
                f_windows = inputs[f_windows]
                f_windows[:,:,0] = 0  # set first value of all future data points to zero
                f_windows = f_windows.reshape(len(f_windows), -1)  # flatten inner lists
                input_windows = np.concatenate((p_windows, f_windows), axis=1)

                output_windows = outputs[windows]
                fuel_output, cost_output = np.split(output_windows, 2, axis=2)
                if not OUTPUT_FUEL:  # => OUTPUT_COST
                    output_windows = cost_output.reshape(len(cost_output), -1)
                elif not OUTPUT_COST:  # => OUTPUT_FUEL
                    output_windows = fuel_output.reshape(len(fuel_output), -1)
                else:  # => OUTPUT_COST && OUTPUT_FUEL
                    output_windows = output_windows.reshape(len(output_windows), -1)

                data_blocks[f'input_{block_num}'] = input_windows
                data_blocks[f'output_{block_num}'] = output_windows
                block_num += 1

    np.savez(cfg['output'], **data_blocks)
