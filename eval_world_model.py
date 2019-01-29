import ib_world_model as generation
from gen_dataset import BenchmarkTrajectory
from misc.dicts import load_data_cfg
from misc.args import parse_cfg_args
from misc.files import ensure_can_write
import numpy as np
import json
import matplotlib.pyplot as plt


def evaluate_world_model(cfg, model = None, eval_input = None, eval_output = None):
    gen_cfg = cfg['generation']
    data_cfg = cfg['data']
    learning_cfg = cfg['learning']

    if model is None:
        model = generation.generate_world_model(cfg)

    if eval_input is None or eval_output is None:
        SEED = gen_cfg['seed']
        np.random.seed(SEED)

        VALIDATION_SPLIT = learning_cfg['validation_split']
        assert 0 < VALIDATION_SPLIT and VALIDATION_SPLIT < 1

        _, _, eval_input, _, eval_output = generation.load_training_data(cfg, False, VALIDATION_SPLIT)

    loss, metrics = model.evaluate(eval_input, eval_output, verbose=1)

    write_to = learning_cfg['validation_output_file']
    ensure_can_write(write_to)
    with open(write_to, 'w') as fp:
        json.dump(
            { 'mean_squared_error': loss, 'mean_absolute_error': metrics },
            fp, indent=4
        )

    write_to = learning_cfg['validation_example_file']
    def gen_plot_file(suffix):
        fname = write_to.split('.')
        fname.insert(-1, suffix)
        fname = '.'.join(fname)
        ensure_can_write(fname)
        return fname

    Z_DIM = data_cfg['z_dim']
    assert 0 < Z_DIM

    OUTPUT_FUEL = data_cfg['output_fuel']

    time_series_len = model.input_shape[1]
    for p in gen_cfg['init_setpoints']:
        env = BenchmarkTrajectory(Z_DIM, (0 if OUTPUT_FUEL else 1), time_series_len, p)
        trajectory = env.to_array()
        inputs = np.concatenate((trajectory['z'], trajectory['a']), axis=1)
        outputs = trajectory['y']
        predictions = model.predict(np.array([ inputs ]))[0]

        plt.plot(predictions, label='predicted')
        plt.plot(outputs, label='actual')
        plt.xlabel('t')
        plt.legend()
        plt.savefig(gen_plot_file(f'p{p}'))
        plt.gcf().clear()


if __name__ == "__main__":
    evaluate_world_model(*parse_cfg_args(load_data_cfg, False))
