import json

DEFAULT_DATA_CFG = {
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

DEFAULT_POLICY_CFG = {
    'fuel_cfg': '',
    'consumption_cfg': '',
    'rules_num': 20,
    'future_rewards_window': 7,
    'future_rewards_weight': 0.5
}


def load_cfg(path: str, default_cfg: dict) -> dict:
    with open(path) as fp:
        cfg = json.load(fp)
    return setdefaults(cfg, default_cfg)


def load_data_cfg(path: str) -> dict:
    return load_cfg(path, DEFAULT_DATA_CFG)


def load_policy_cfg(path: str) -> dict:
    return load_cfg(path, DEFAULT_POLICY_CFG)


def setdefaults(dest: dict, source: dict) -> dict:
    for k, v in source.items():
        if type(v) == dict:
            setdefaults(dest.setdefault(k, dict()), v)
        else:
            dest.setdefault(k, v)
    return dest
