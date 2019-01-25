import json

DEFAULT_CFG = {
    'generation': {
        'seed': 0,
        'init_setpoints': range(10, 101, 10),
        'past_window': 10,
        'future_window': 50,
        'min_block_size': 90,
        'max_block_size': 120,
        'trajectories_num': 10,
        'trajectories_length': 1000
    },
    'data': {
        'z_dim': 4,
        'a_dim': 3,
        'y_dim': 1,
        'output_fuel': False,
        'output_consumption': False,
    },
    'data_output_file': '',
    'learning': {
        'state_dim': 20,
        'self_input': True,
        'epochs': 3,
        'batch_size': 64,
        'learning_rate': 0.05,
        'learning_rate_steps': 10,
        'validation_split': 0.3
    },
    'model_output_file': ''
}

def load_cfg(path: str) -> dict:
    with open(path) as fp:
        cfg = json.load(fp)
    return setdefaults(cfg, DEFAULT_CFG)

def setdefaults(dest: dict, source: dict) -> dict:
    for k, v in source.items():
        if type(v) == dict:
            setdefaults(dest.setdefault(k, dict()), v)
        else:
            dest.setdefault(k, v)
    return dest
