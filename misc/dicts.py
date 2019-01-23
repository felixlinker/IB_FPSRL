import json

DEFAULT_CFG = {
    'seed': 0,
    'init_setpoints': range(10, 101, 10),
    'past_window': 10,
    'future_window': 50,
    'min_block_size': 90,
    'max_block_size': 120,
    'trajectories_num': 10,
    'trajectories_length': 1000,
    'state_dim': 20,
    'output_fuel': False,
    'self_input_fuel': False,
    'output_consumption': False,
    'self_input_consumption': False,
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
