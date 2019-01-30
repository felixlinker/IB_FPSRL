from ib_world_model import generate_world_model
from gen_dataset import BenchmarkGenerator
import eval_policy as evaluation
import numpy as np
from functools import reduce
from misc.dicts import load_policy_cfg, load_data_cfg
from misc.args import parse_cfg_args
from misc.files import ensure_can_write
import argparse
import pyswarms as ps
import os
from multiprocessing.pool import Pool
from policy import load_policy, Policy, TrajectoryGenerator, TrajectoryCosts


def mk_weights(weight_horizon, initial_weight):
    def mk_weight(x):
        return initial_weight ** (1 / (x + 1))
    return (np.vectorize(mk_weight))(np.arange(weight_horizon))


# The following two methods will be used to generates trajectories for policies
# concurrently.
def init_trajectory_generator(policy_args, generator_args):
    '''
    Initialize a global trajectory generator for a policy.

    Parameters
    ----------
    policy_args : dict
        Arguments to instantiate `Policy`
    generator_args : dict
        Arguments without `policy` to instantiate the `TrajectoryGenerator`
    '''
    global trajectory_generator
    policy = Policy(**policy_args)
    trajectory_generator = TrajectoryGenerator(policy=policy, **generator_args)


def map_trajectories(policy_cfg):
    '''
    Uses the global trajectory generator to generate a trajectory for a policy
    configuration vector.

    Parameters
    ----------
    policy_cfg : np.ndarray
        Policy configuration vector

    Returns
    -------
    np.ndarray
        Array of trajectories
    '''
    global trajectory_generator
    return trajectory_generator(policy_cfg)


class PolicyEvaluater:
    '''
    Evaluates batches of policies mapping them to their mean costs for a set of
    initial benchmark setpoints. This class is a context manager and supports
    parallel evaluation of the policies.
    '''
    def __init__(self, cost_function, eval_setpoints, eval_window, eval_weight, time_series_len, data_points_num, policy_args):
        '''
        Parameters
        ----------
        cost_function : function
            Function that maps a batch of trajectories as a time series to their
            final cost.
        eval_setpoints : Iterable[float]
            Iterable returning the initial sepoints for which each policy
            configuration should be evaluated.
        eval_window : int
            Number of steps a policy should be evaluated into the future.
        eval_weight : float
            Each step into the future for policy evaluation will be weighed less
            and less. This is the base weight and should be >= 1.
        time_series_len : int
            What is the length of a time series that the cost function predicts?
        data_points_num : int
            How many policies does this class approximately need to evaluate?
            This argument will be used to optimize concurrent behavior.
        policy_args : dict
            This class will need to instantiate `Policy`. These are the
            arguments to the constructor of `Policy`.
        '''
        self.cost_function = cost_function
        assert 1 <= eval_window
        self.eval_window_indices = np.arange(eval_window).reshape((eval_window, 1)) + np.arange(time_series_len)
        assert 1 <= eval_weight
        self.eval_window_weights = mk_weights(eval_window, eval_weight)
        self.time_series_len = time_series_len
        self.p = Pool(
            initializer=init_trajectory_generator,
            initargs=[ policy_args, {
                'initial_setpoints': eval_setpoints,
                'trajectory_len': eval_window + time_series_len - 1,
            } ],
            maxtasksperchild=data_points_num  # don't recreate workers
        )

    def __call__(self, policy_cfg_batch):
        '''
        Parameters
        ----------
        policy_cfg_batch : np.ndarray
            Array were each row is a policy configuration

        Returns
        -------
        np.ndarray
            Array where each index corresponds to the respective policy's costs
        '''
        # Map each policy cfg to a number of trajectories
        trajectories = np.array(self.p.map(map_trajectories, policy_cfg_batch))
        # Map each trajectory to a matrix of size with as many rows as
        # eval_window and as many colums as the world model requires it
        trajectories = trajectories[:,:,self.eval_window_indices]
        trajectories_shape = np.shape(trajectories)
        # reshape the matrix such that we can get the costs as batches
        batch = trajectories.reshape(-1, self.time_series_len, trajectories_shape[-1])
        costs = self.cost_function(batch)
        # reverse the reshape to batches but exclude last two dimensions as these
        # have been mapped to costs
        costs = costs.reshape(trajectories_shape[:-2])
        # weigh each cost depending on its distance to the initial we consider
        costs *= self.eval_window_weights
        # sum costs for each trajectory
        costs = np.sum(costs, axis=2)
        # average costs for each policy
        return np.mean(costs, axis=1)

    def __enter__(self, *args, **kwargs):
        self.p.__enter__(*args, **kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        self.p.__exit__(*args, **kwargs)


def generate_policy(cfg, clean = False, strict_clean = False):
    '''
    Generates a policy based on a world model. Loads the policy if at
    the path given in the configuration dict there already is a policy.

    Returns
    -------
    np.ndarray
        Policy tuple. First element is the state dimension, second element is
        the number of rules per partial polciy, third element the actual policy
        weights.
    '''
    write_to = cfg['policy_output_file']
    if os.path.isfile(write_to) and not (clean or strict_clean):
        return load_policy(*np.load(write_to))

    policy_cfg = cfg['policy']
    pso_cfg = cfg['particle_swarm']

    fuel_cfg = load_data_cfg(cfg['fuel_cfg'])
    consumption_cfg = load_data_cfg(cfg['consumption_cfg'])

    T_0_INDEX = fuel_cfg['generation']['past_window'] - 1
    assert T_0_INDEX == consumption_cfg['generation']['past_window'] - 1

    Z_DIM = fuel_cfg['data']['z_dim']
    assert Z_DIM == consumption_cfg['data']['z_dim']
    assert 0 < Z_DIM

    RULES_NUM = policy_cfg['rules_num']
    assert 0 < RULES_NUM

    PARTICLES_NUM = pso_cfg['particles']
    assert 0 < PARTICLES_NUM

    PSO_ITERS = pso_cfg['iterations']
    assert 0 < PSO_ITERS

    PSO_PARAMS = pso_cfg['hyperparameters']

    FUTURE_WINDOW_SIZE = policy_cfg['future_rewards_window']
    assert 0 < FUTURE_WINDOW_SIZE

    INITIAL_WEIGHT = policy_cfg['future_rewards_weight']
    assert 1 <= INITIAL_WEIGHT

    EVAL_SETPOINTS = policy_cfg['eval_setpoints']
    assert 0 < len(EVAL_SETPOINTS)

    # Load the world model and create a cost function with it
    fuel_model = generate_world_model(fuel_cfg, strict_clean, strict_clean)
    consumption_model = generate_world_model(consumption_cfg, strict_clean, strict_clean)
    cost_function = TrajectoryCosts(fuel_model, consumption_model, T_0_INDEX)

    # arguments to constructor of Policy
    policy_args = { 'input_dim': Z_DIM, 'rules_num': RULES_NUM }
    sample_policy = Policy(**policy_args)

    optimizer = ps.single.LocalBestPSO(
        n_particles=PARTICLES_NUM,
        dimensions=len(sample_policy),
        options=PSO_PARAMS,
        bounds=sample_policy.bounds()
    )

    with PolicyEvaluater(cost_function, EVAL_SETPOINTS, FUTURE_WINDOW_SIZE,
        INITIAL_WEIGHT, T_0_INDEX + 1, PSO_ITERS * PARTICLES_NUM, policy_args) as evaluater:
        _, policy_weights = optimizer.optimize(
            evaluater,
            print_step=int(0.1 * PSO_ITERS),
            iters=PSO_ITERS,
            verbose=3
        )

    evaluation.evaluate_policy(cfg, policy_weights)

    sample_policy.update(policy_weights)
    print(sample_policy)

    ensure_can_write(write_to)
    np.save(write_to, (Z_DIM, RULES_NUM, policy_weights))

    return sample_policy


if __name__ == "__main__":
    generate_policy(*parse_cfg_args(load_policy_cfg))
