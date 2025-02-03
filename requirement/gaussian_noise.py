from typing import Union
import random
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    get_variable, TensorType
from ray.rllib.utils.schedules.piecewise_schedule import PiecewiseSchedule



tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()


class GaussianNoise(Exploration):

    def __init__(self,
                 action_space,
                 *,
                 framework: str,
                 model: ModelV2,
                 random_timesteps=1000,
                 stddev=0.1,
                 initial_scale=1.0,
                 final_scale=0.02,
                 scale_timesteps=10000,
                 scale_schedule=None,
                 **kwargs):

        assert framework is not None
        super().__init__(
            action_space, model=model, framework=framework, **kwargs)

        self.random_timesteps = random_timesteps
        self.random_exploration = Random(
            action_space, model=self.model, framework=self.framework, **kwargs)
        self.stddev = stddev
        # The `scale` annealing schedule.
        self.scale_schedule = scale_schedule or PiecewiseSchedule(
            endpoints=[(random_timesteps, initial_scale),
                       (random_timesteps + scale_timesteps, final_scale)],
            outside_value=final_scale,
            framework=self.framework)

        # The current timestep value (tf-var or python int).
        self.last_timestep = get_variable(
            0, framework=self.framework, tf_name="timestep")

        # Build the tf-info-op.
        if self.framework in ["tf", "tfe"]:
            self._tf_info_op = self.get_info()

    @override(Exploration)
    def get_exploration_action(self,
                               *,
                               action_distribution: ActionDistribution,
                               timestep: Union[int, TensorType],
                               explore: bool = True,
                               policy, state):
        # Adds IID Gaussian noise for exploration, TD3-style.
        if self.framework == "torch":
            return self._get_torch_exploration_action(action_distribution,
                                                      explore, timestep, policy, state)
        else:
            return self._get_tf_exploration_action_op(action_distribution,
                                                      explore, timestep)

    def _get_tf_exploration_action_op(self, action_dist, explore, timestep):
        ts = timestep if timestep is not None else self.last_timestep

        # The deterministic actions (if explore=False).
        deterministic_actions = action_dist.deterministic_sample()

        # Take a Gaussian sample with our stddev (mean=0.0) and scale it.
        gaussian_sample = self.scale_schedule(ts) * tf.random.normal(
            tf.shape(deterministic_actions), stddev=self.stddev)

        # Stochastic actions could either be: random OR action + noise.
        random_actions, _ = \
            self.random_exploration.get_tf_exploration_action_op(
                action_dist, explore)
        stochastic_actions = tf.cond(
            pred=tf.convert_to_tensor(ts <= self.random_timesteps),
            true_fn=lambda: random_actions,
            false_fn=lambda: tf.clip_by_value(
                deterministic_actions + gaussian_sample,
                self.action_space.low * tf.ones_like(deterministic_actions),
                self.action_space.high * tf.ones_like(deterministic_actions))
        )

        # Chose by `explore` (main exploration switch).
        batch_size = tf.shape(deterministic_actions)[0]
        action = tf.cond(
            pred=tf.constant(explore, dtype=tf.bool)
            if isinstance(explore, bool) else explore,
            true_fn=lambda: stochastic_actions,
            false_fn=lambda: deterministic_actions)
        logp = tf.zeros(shape=(batch_size,), dtype=tf.float32)

        # Increment `last_timestep` by 1 (or set to `timestep`).
        if self.framework in ["tf2", "tfe"]:
            if timestep is None:
                self.last_timestep.assign_add(1)
            else:
                self.last_timestep.assign(timestep)
            return action, logp
        else:
            assign_op = (
                tf1.assign_add(self.last_timestep, 1) if timestep is None else
                tf1.assign(self.last_timestep, timestep))
            with tf1.control_dependencies([assign_op]):
                return action, logp

    def _get_torch_exploration_action(self, action_dist, explore, timestep, policy, state):
        # Set last timestep or (if not given) increase by one.
        global action
        self.last_timestep = timestep if timestep is not None else \
            self.last_timestep + 1
        # Apply exploration.
        if explore:
            # Random exploration phase.
            if self.last_timestep <= self.random_timesteps:
                action, _ = \
                    self.random_exploration.get_torch_exploration_action(
                        action_dist, explore=True)

                acc_action = torch.Tensor([action[0][0]])
                lc_action = action[0][1]

                if lc_action > 0.333:
                    lc_action = torch.Tensor([1])
                elif lc_action <= 0.333 and lc_action >= -0.333:
                    lc_action = torch.Tensor([0])
                elif lc_action < -0.333:
                    lc_action = torch.Tensor([-1])

                # Concatenation A_acc + A_lc
                action = torch.cat([acc_action, lc_action]).reshape((1,2))

            else:
                det_actions = action_dist.deterministic_sample()

                # Critic Copy
                weights = policy.get_weights()

                weights_val = []
                for key, val in weights.items():
                    weights_val.append(val)

                qval_wei = []
                qval_bias = []
                for i in range(6, 12):
                    if i % 2 == 0:
                        qval_wei.append(weights_val[i])
                    else:
                        qval_bias.append(weights_val[i])

                tqval_wei = []
                tqval_bias = []
                for i in range(12, 18):
                    if i % 2 == 0:
                        tqval_wei.append(weights_val[i])
                    else:
                        tqval_bias.append(weights_val[i])

                # Using Softmax
                if len(det_actions[0]) == 4:
                    acc_det_actions = det_actions[0][0]
                    LC_det_actions = det_actions[0][1:]

                    # Exploit action select - Softmax
                    if torch.max(LC_det_actions) == LC_det_actions[0]:
                        LC_det_actions = torch.Tensor([1,0,0])
                    elif torch.max(LC_det_actions) == LC_det_actions[1]:
                        LC_det_actions = torch.Tensor([0,1,0])
                    elif torch.max(LC_det_actions) == LC_det_actions[2]:
                        LC_det_actions = torch.Tensor([0,0,1])

                    # Exploration: Epsilon Greedy for Lane Change action
                    epsilon = self.scale_schedule(self.last_timestep)
                    possible_actions = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    random_actions = random.choice(possible_actions)
                    LC_det_actions = torch.where(
                        torch.empty(
                            torch.Tensor(3).size(),).uniform_().to(self.device) < epsilon,
                        random_actions, LC_det_actions)

                    # Exploration: Guassian Noise for Acceleration action
                    scale = self.scale_schedule(self.last_timestep)
                    gaussian_sample = scale * torch.normal(
                        mean=torch.zeros(acc_det_actions.size()), std=self.stddev).to(
                        self.device)
                    acc_det_actions = torch.Tensor([acc_det_actions + gaussian_sample])

                    # Concatenation A_acc + A_lc
                    det_actions = torch.cat([acc_det_actions, LC_det_actions]).reshape((1,4))

                    # Action Bound - Action Space
                    action = torch.min(
                        torch.max(
                            det_actions,
                            torch.tensor(
                                self.action_space.low,
                                dtype=torch.float32,
                                device=self.device)),
                        torch.tensor(
                            self.action_space.high,
                            dtype=torch.float32,
                            device=self.device))

                # Using Hard Boundary
                elif len(det_actions[0]) == 2:
                    acc_det_actions = det_actions[0][0]
                    LC_det_actions = det_actions[0][1]

                    # Exploit action select - Hard Boundary
                    if LC_det_actions > 0.333:
                        LC_det_actions = torch.Tensor([1])
                    elif LC_det_actions <= 0.333 and LC_det_actions >= -0.333:
                        LC_det_actions = torch.Tensor([0])
                    elif LC_det_actions < -0.333:
                        LC_det_actions = torch.Tensor([-1])

                    # Exploration: Epsilon Greedy for Lane Change action
                    epsilon = self.scale_schedule(self.last_timestep)
                    possible_actions = torch.Tensor([[-1], [0], [1]])
                    random_actions = random.choice(possible_actions)
                    prob = torch.empty(
                        torch.Tensor(1).size(), ).uniform_().to(self.device)
                    LC_det_actions = torch.where(
                        prob < epsilon,
                        random_actions, LC_det_actions)

                    # Exploration: Guassian Noise for Acceleration action
                    scale = self.scale_schedule(self.last_timestep)
                    gaussian_sample = scale * torch.normal(
                        mean=torch.zeros(acc_det_actions.size()), std=self.stddev).to(
                        self.device)
                    acc_det_actions = torch.Tensor([acc_det_actions + gaussian_sample])

                    # Concatenation A_acc + A_lc
                    det_actions = torch.cat([acc_det_actions, LC_det_actions]).reshape((1,2))

                    if self.last_timestep > 900000:
                        ip1 = torch.cat((state[0], acc_det_actions, torch.Tensor([-1])))
                        ip2 = torch.cat((state[0], acc_det_actions, torch.Tensor([0])))
                        ip3 = torch.cat((state[0], acc_det_actions, torch.Tensor([1])))
                        ip_list = [ip1, ip2, ip3]

                        q_val_list = []
                        tq_val_list = []
                        for i in ip_list:
                            out = F.relu(F.linear(i, torch.from_numpy(qval_wei[0]),
                                                  torch.from_numpy(qval_bias[0])))
                            out = F.relu(F.linear(out, torch.from_numpy(qval_wei[1]),
                                                  torch.from_numpy(qval_bias[1])))
                            out = torch.tanh(F.linear(out, torch.from_numpy(qval_wei[2]),
                                                      torch.from_numpy(qval_bias[2])))
                            q_val_list.append(out)

                            tout = F.relu(F.linear(i, torch.from_numpy(qval_wei[0]),
                                                   torch.from_numpy(qval_bias[0])))
                            tout = F.relu(F.linear(tout, torch.from_numpy(qval_wei[1]),
                                                   torch.from_numpy(qval_bias[1])))
                            tout = torch.tanh(F.linear(tout, torch.from_numpy(qval_wei[2]),
                                                       torch.from_numpy(qval_bias[2])))
                            tq_val_list.append(tout)

                        for i in [0, 1, 2]:
                            if max(q_val_list) == q_val_list[i]:
                                lc = i - 1
                                Q_action = torch.cat((acc_det_actions, torch.Tensor([lc])))

                    # Action Bound - Action Space
                    action = torch.min(
                        torch.max(
                            det_actions,
                            torch.tensor(
                                self.action_space.low,
                                dtype=torch.float32,
                                device=self.device)),
                        torch.tensor(
                            self.action_space.high,
                            dtype=torch.float32,
                            device=self.device))

        else:
            det_actions = action_dist.deterministic_sample()

            if len(det_actions[0]) == 2:
                acc_det_actions = det_actions[0][0]
                LC_det_actions = det_actions[0][1]

                # Exploit action select - Hard Boundary
                if LC_det_actions > 0.333:
                    LC_det_actions = torch.Tensor([1])
                elif LC_det_actions <= 0.333 and LC_det_actions >= -0.333:
                    LC_det_actions = torch.Tensor([0])
                elif LC_det_actions < -0.333:
                    LC_det_actions = torch.Tensor([-1])

                action = torch.cat([acc_det_actions, LC_det_actions]).reshape((1, 2))

            elif len(det_actions[0]) == 4:
                acc_det_actions = det_actions[0][0]
                LC_det_actions = det_actions[0][1:]

                # Exploit action select - Softmax
                if torch.max(LC_det_actions) == LC_det_actions[0]:
                    LC_det_actions = torch.Tensor([1, 0, 0])
                elif torch.max(LC_det_actions) == LC_det_actions[1]:
                    LC_det_actions = torch.Tensor([0, 1, 0])
                elif torch.max(LC_det_actions) == LC_det_actions[2]:
                    LC_det_actions = torch.Tensor([0, 0, 1])

                action = torch.cat([acc_det_actions, LC_det_actions]).reshape((1, 2))

        logp = torch.zeros(
            (action.size()[0], ), dtype=torch.float32, device=self.device)

        return action, logp

    @override(Exploration)
    def get_info(self, sess=None):
        """Returns the current scale value.

        Returns:
            Union[float,tf.Tensor[float]]: The current scale value.
        """
        if sess:
            return sess.run(self._tf_info_op)
        scale = self.scale_schedule(self.last_timestep)
        return {"cur_scale": scale}
