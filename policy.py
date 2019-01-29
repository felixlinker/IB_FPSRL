import numpy as np


class Policy:
    SCALE_V = 1.0
    SCALE_G = 10.0
    SCALE_H = 5.75

    def __init__(self, input_dim, rules_num):
        self.v_policy = PartialPolicy(input_dim, rules_num)
        self.g_policy = PartialPolicy(input_dim, rules_num)
        self.h_policy = PartialPolicy(input_dim, rules_num)

    def __len__(self):
        return int(len(self.v_policy) + len(self.g_policy) + len(self.h_policy))

    def update(self, policy_cfg):
        i = len(self.v_policy)
        j = i + len(self.g_policy)
        v_cfg, g_cfg, h_cfg = np.split(policy_cfg, [ i, j ])
        self.v_policy.update(v_cfg)
        self.g_policy.update(g_cfg)
        self.h_policy.update(h_cfg)

    def bounds(self):
        v_lower_bounds, v_upper_bounds = self.v_policy.bounds()
        g_lower_bounds, g_upper_bounds = self.g_policy.bounds()
        h_lower_bounds, h_upper_bounds = self.h_policy.bounds()
        return (
            np.concatenate([ v_lower_bounds, g_lower_bounds, h_lower_bounds ]),
            np.concatenate([ v_upper_bounds, g_upper_bounds, h_upper_bounds ])
        )

    def __call__(self, state):
        def minmax(x):
            return max(0, min(100, x))
        s, v, g, h, _, _, _ = state
        v_d, g_d, h_d = self.v_policy(state), self.g_policy(state), self.h_policy(state)
        return [
            s,
            minmax(v + self.SCALE_V * v_d),
            minmax(g + self.SCALE_G * g_d),
            minmax(h + self.SCALE_H * h_d),
            v_d,
            g_d,
            h_d
        ]

    def __repr__(self):
        representation = 'Rules for velocity change:\n' + repr(self.v_policy) + '\n'
        representation += 'Rules for gain change:\n' + repr(self.g_policy) + '\n'
        representation += 'Rules for shift change:\n' + repr(self.h_policy) + '\n'
        return representation


def squares(x):
    return x * x
squares = np.vectorize(squares)


class PartialPolicy:
    def __init__(self, input_dim, rules_num):
        self.input_shape = (input_dim,)
        self.rules_num = rules_num
        self.policy_means = np.zeros((rules_num, input_dim))
        self.policy_deviations = np.zeros((rules_num, input_dim))
        self.policy_actions = np.zeros((rules_num,))
        self.policy_slope = np.zeros(())

    def __call__(self, state):
        state = np.resize(state, self.input_shape)  # TODO: should I really use resize?
        input_state_vec = state * np.ones(self.rules_num).reshape((self.rules_num, 1))  # shape (rules_num, input_dim)
        activations = squares(self.policy_means - input_state_vec)  # shape (rules_num, input_dim)
        activations /= 2 * squares(self.policy_deviations)  # shape (rules_num, input_dim)
        activations = np.exp(-1 * activations)  # shape (rules_num, input_dim)
        activations = np.prod(activations, axis=1)  # shape (rules_num,)
        actions = activations * self.policy_actions  # shape (rules_num,)
        output = np.sum(actions) / np.sum(activations)  # shape ()
        return np.tanh(self.policy_slope * output)  # shape ()

    def __len__(self):
        return int(
            np.prod(self.policy_means.shape) + np.prod(self.policy_deviations.shape)
            + np.prod(self.policy_actions.shape) + np.prod(self.policy_slope.shape)
        )

    def update(self, policy_cfg):
        assert np.shape(policy_cfg) == (len(self),)
        i = np.prod(self.policy_means.shape)            # index after means
        j = i + np.prod(self.policy_deviations.shape)   # index after deviations
        k = j + np.prod(self.policy_actions.shape)      # index after actions
        means, deviations, actions, slope = np.split(policy_cfg, [ i, j, k ])
        self.policy_means = np.reshape(means, self.policy_means.shape)
        self.policy_deviations = np.reshape(deviations, self.policy_deviations.shape)
        self.policy_actions = np.reshape(actions, self.policy_actions.shape)
        self.policy_slope = np.reshape(slope, self.policy_slope.shape)

    def bounds(self):
        return (
            np.concatenate((  # lower:
                np.zeros(self.policy_means.shape).flatten(),
                np.zeros(self.policy_deviations.shape).flatten(),
                -1 * np.ones(self.policy_actions.shape).flatten(),
                -2 * np.ones(self.policy_slope.shape).flatten(),
            )),
            np.concatenate((  # upper:
                100 * np.ones(self.policy_means.shape).flatten(),
                100 * np.ones(self.policy_deviations.shape).flatten(),
                np.ones(self.policy_actions.shape).flatten(),
                2 * np.ones(self.policy_slope.shape).flatten()
            ))
        )

    def __repr__(self):
        representation = ''
        for r_i in range(self.policy_means.shape[0]):
            m_repr = ', '.join(map(repr, zip(self.policy_means[r_i], self.policy_deviations[r_i])))
            representation += f'R{r_i}: IF s IS m({m_repr}) THEN {self.policy_actions[r_i]}\n'
        return representation

INIT_A = (50.0, 50.0, 50.0, 0.0, 0.0, 0.0)


class TrajectoryGenerator:
    def __init__(self, initial_setpoints, trajectory_len, policy):
        self.initial_setpoints = initial_setpoints
        self.trajectory_len = trajectory_len
        self.trajectory_iter = 0
        self.policy = policy

    def __iter__(self):
        return self

    def set_init_state(self, state):
        self.state = state
        self.trajectory_iter = self.trajectory_len

    def __next__(self):
        if self.trajectory_iter == 0:
            raise StopIteration
        else:
            self.state = self.policy(self.state)
            self.trajectory_iter -= 1
            return self.state

    def __len__(self):
        return self.trajectory_len

    def generate_trajectory(self, setpoint):
        self.set_init_state((setpoint,) + INIT_A)
        return list(self)

    def __call__(self, policy_cfg):
        self.policy.update(policy_cfg)
        return np.array(list(map(self.generate_trajectory, self.initial_setpoints)))


class TrajectoryCosts:
    def __init__(self, fuel_model, consumption_model, map_to_t):
        assert fuel_model.input_shape == consumption_model.input_shape
        assert len(fuel_model.inputs) == 1
        self.fuel_model = fuel_model
        self.consumption_model = consumption_model
        self.map_to_t = map_to_t
        self.pad_time_series = fuel_model.input_shape[1] - (map_to_t + 1)

    def _pad_time_series(self, batch):
        batch_size, t_len, size = np.shape(batch)
        assert t_len == self.map_to_t + 1
        zeros = np.zeros((batch_size, self.pad_time_series, size))
        return np.concatenate((batch, zeros), axis=1)

    def __call__(self, batch):
        batch = self._pad_time_series(batch)
        consumption = self.consumption_model.predict(batch)[:,self.map_to_t]
        fuel = self.fuel_model.predict(batch)[:,self.map_to_t]
        return consumption + 3 * fuel