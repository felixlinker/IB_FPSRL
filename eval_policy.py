from gen_dataset import BenchmarkGenerator
from policy import Policy
import ib_policy as generation
from misc.dicts import load_policy_cfg, load_data_cfg
from misc.args import parse_cfg_args
from misc.files import ensure_can_write
import numpy as np
import json


class ActionApplier:
    def __init__(self, z_dim):
        self.z_dim = z_dim

    def __call__(self, init_setpoint, trajectory):
        env = BenchmarkGenerator(self.z_dim, init_setpoint)
        # we only care about the last result as this is what the world model does
        _, (fuel, consumption) = list(map(env.__call__, trajectory))[-1]
        return consumption + 3 * fuel


class PolicyCost:
    def __init__(self, z_dim):
        self.z_dim = z_dim
        self.batch_applier = ActionApplier(z_dim)

    def __call__(self, trajectory_batch):
        setpoints, _, action_batch = np.split(trajectory_batch, [1, self.z_dim], axis=2)
        setpoints = setpoints.reshape(action_batch.shape[0], -1)[:,0]
        return np.fromiter(
            map(lambda t: self.batch_applier(*t), zip(setpoints, action_batch)),
            dtype=float
        )


class RandomPolicyGenerator:
    def __init__(self, input_dim, rules_num):
        policy = Policy(input_dim, rules_num)
        min_bounds, max_bounds = policy.bounds()
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

    def get(self):
        rand = np.random.rand(*np.shape(self.max_bounds))
        return rand * (self.max_bounds - self.min_bounds) + self.min_bounds


def evaluate_policy(cfg, policy = None):
    policy_cfg = cfg['policy']
    evaluation_cfg = cfg['evaluation']
    fuel_cfg = load_data_cfg(cfg['fuel_cfg'])
    gen_cfg = fuel_cfg['generation']

    if policy is None:
        policy = generation.generate_policy(cfg)

    Z_DIM = fuel_cfg['data']['z_dim']
    T_SERIES_LEN = gen_cfg['past_window'] + gen_cfg['future_window']

    RULES_NUM = policy_cfg['rules_num']
    FUTURE_WINDOW_SIZE = policy_cfg['future_rewards_window']
    INITIAL_WEIGHT = policy_cfg['future_rewards_weight']
    EVAL_SETPOINTS = policy_cfg['eval_setpoints']

    RANDOM_POLICIES_NUM = cfg['evaluation']['random_policies']

    cost_function = PolicyCost(Z_DIM)
    policy_args = { 'input_dim': Z_DIM, 'rules_num': RULES_NUM }
    evaluater = generation.PolicyEvaluater(cost_function, EVAL_SETPOINTS,
        FUTURE_WINDOW_SIZE, INITIAL_WEIGHT, T_SERIES_LEN, RANDOM_POLICIES_NUM,
        policy_args)

    random_generator = RandomPolicyGenerator(**policy_args)
    random_policies = np.array([ random_generator.get() for _ in range(RANDOM_POLICIES_NUM) ])
    random_cost = np.mean(evaluater(random_policies))

    policy_cost = evaluater(np.array([ policy ]))[0]

    write_to = evaluation_cfg['validation_output_file']
    ensure_can_write(write_to)
    with open(write_to, 'w') as fp:
        json.dump(
            { 'mean_random_performance': random_cost, 'policy_performance': policy_cost },
            fp, indent=4
        )


if __name__ == "__main__":
    evaluate_policy(*parse_cfg_args(load_policy_cfg, False))
