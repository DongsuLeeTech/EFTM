import functools
import gym
import numpy as np
import time
from typing import Callable, Dict, List, Optional, Tuple, Union



import ray
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.trajectory_view import get_trajectory_view
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.schedules import ConstantSchedule, PiecewiseSchedule
from ray.rllib.utils.torch_ops import convert_to_non_torch_type, \
    convert_to_torch_tensor
from ray.rllib.utils.tracking_dict import UsageTrackingDict
from ray.rllib.utils.types import AgentID, ModelGradients, ModelWeights, \
    TensorType, TrainerConfigDict

torch, _ = try_import_torch()


@DeveloperAPI
class TorchPolicy(Policy):
    """Template for a PyTorch policy and loss to use with RLlib.

    This is similar to TFPolicy, but for PyTorch.

    Attributes:
        observation_space (gym.Space): observation space of the policy.
        action_space (gym.Space): action space of the policy.
        config (dict): config of the policy.
        model (TorchModel): Torch model instance.
        dist_class (type): Torch action distribution class.
    """

    @DeveloperAPI
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            config: TrainerConfigDict,
            *,
            model: ModelV2,
            loss: Callable[[Policy, ModelV2, type, SampleBatch], TensorType],
            action_distribution_class: TorchDistributionWrapper,
            action_sampler_fn: Callable[[TensorType, List[TensorType]], Tuple[
                TensorType, TensorType]] = None,
            action_distribution_fn: Optional[Callable[[
                Policy, ModelV2, TensorType, TensorType, TensorType
            ], Tuple[TensorType, type, List[TensorType]]]] = None,
            max_seq_len: int = 20,
            get_batch_divisibility_req: Optional[int] = None):

        self.framework = "torch"
        super().__init__(observation_space, action_space, config)
        if torch.cuda.is_available() and ray.get_gpu_ids():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = model.to(self.device)
        self.exploration = self._create_exploration()
        self.unwrapped_model = model  # used to support DistributedDataParallel
        self._loss = loss
        self._optimizers = force_list(self.optimizer())

        self.dist_class = action_distribution_class
        self.action_sampler_fn = action_sampler_fn
        self.action_distribution_fn = action_distribution_fn

        # If set, means we are using distributed allreduce during learning.
        self.distributed_world_size = None

        self.max_seq_len = max_seq_len
        self.batch_divisibility_req = \
            get_batch_divisibility_req(self) if get_batch_divisibility_req \
            else 1

    @override(Policy)
    @DeveloperAPI
    def compute_actions(
            self,
            obs_batch: Union[List[TensorType], TensorType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Union[List[TensorType], TensorType] = None,
            prev_reward_batch: Union[List[TensorType], TensorType] = None,
            info_batch: Optional[Dict[str, list]] = None,
            episodes: Optional[List["MultiAgentEpisode"]] = None,
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep

        with torch.no_grad():
            seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)
            input_dict = self._lazy_tensor_dict({
                SampleBatch.CUR_OBS: np.asarray(obs_batch),
                "is_training": False,
            })
            if prev_action_batch is not None:
                input_dict[SampleBatch.PREV_ACTIONS] = \
                    np.asarray(prev_action_batch)
            if prev_reward_batch is not None:
                input_dict[SampleBatch.PREV_REWARDS] = \
                    np.asarray(prev_reward_batch)
                # print(prev_reward_batch)
            state_batches = [
                convert_to_torch_tensor(s, self.device)
                for s in (state_batches or [])
            ]
            actions, state_out, extra_fetches, logp = \
                self._compute_action_helper(
                    input_dict, state_batches, seq_lens, explore, timestep)

            # Action-logp and action-prob.
            if logp is not None:
                logp = convert_to_non_torch_type(logp)
                extra_fetches[SampleBatch.ACTION_PROB] = np.exp(logp)
                extra_fetches[SampleBatch.ACTION_LOGP] = logp

            return convert_to_non_torch_type((actions, state_out,
                                              extra_fetches))

    @override(Policy)
    def compute_actions_from_trajectories(
            self,
            trajectories: List["Trajectory"],
            other_trajectories: Optional[Dict[AgentID, "Trajectory"]] = None,
            explore: bool = None,
            timestep: Optional[int] = None,
            **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep

        with torch.no_grad():
            # Create a view and pass that to Model as `input_dict`.
            input_dict = self._lazy_tensor_dict(
                get_trajectory_view(
                    self.model, trajectories, is_training=False))
            # TODO: (sven) support RNNs w/ fast sampling.
            state_batches = []
            seq_lens = None

            actions, state_out, extra_fetches, logp = \
                self._compute_action_helper(
                    input_dict, state_batches, seq_lens, explore, timestep)

            # Leave outputs as is (torch.Tensors): Action-logp and action-prob.
            if logp is not None:
                extra_fetches[SampleBatch.ACTION_PROB] = torch.exp(logp)
                extra_fetches[SampleBatch.ACTION_LOGP] = logp

            return actions, state_out, extra_fetches

    def _compute_action_helper(self, input_dict, state_batches, seq_lens,
                               explore, timestep):
        """Shared forward pass logic (w/ and w/o trajectory view API).

        Returns:
            Tuple:
                - actions, state_out, extra_fetches, logp.
        """
        if self.action_sampler_fn:
            action_dist = dist_inputs = None
            state_out = []
            actions, logp = self.action_sampler_fn(
                self,
                self.model,
                input_dict[SampleBatch.CUR_OBS],
                explore=explore,
                timestep=timestep)
        else:
            # Call the exploration before_compute_actions hook.
            self.exploration.before_compute_actions(
                explore=explore, timestep=timestep)
            if self.action_distribution_fn:
                dist_inputs, dist_class, state_out = \
                    self.action_distribution_fn(
                        self,
                        self.model,
                        input_dict[SampleBatch.CUR_OBS],
                        explore=explore,
                        timestep=timestep,
                        is_training=False)
            else:
                dist_class = self.dist_class
                dist_inputs, state_out = self.model(input_dict, state_batches,
                                                    seq_lens)

            if not (isinstance(dist_class, functools.partial)
                    or issubclass(dist_class, TorchDistributionWrapper)):
                raise ValueError(
                    "`dist_class` ({}) not a TorchDistributionWrapper "
                    "subclass! Make sure your `action_distribution_fn` or "
                    "`make_model_and_action_dist` return a correct "
                    "distribution class.".format(dist_class.__name__))
            action_dist = dist_class(dist_inputs, self.model)

            # Get the exploration action from the forward results.
            actions, logp = \
                self.exploration.get_exploration_action(
                    action_distribution=action_dist,
                    timestep=timestep,
                    explore=explore)
                    # policy=self,
                    # state=input_dict['obs'])

        input_dict[SampleBatch.ACTIONS] = actions

        # Add default and custom fetches.
        extra_fetches = self.extra_action_out(input_dict, state_batches,
                                              self.model, action_dist)

        # Action-dist inputs.
        if dist_inputs is not None:
            extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs

        return actions, state_out, extra_fetches, logp

    @override(Policy)
    @DeveloperAPI
    def compute_log_likelihoods(
            self,
            actions: Union[List[TensorType], TensorType],
            obs_batch: Union[List[TensorType], TensorType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Optional[Union[List[TensorType],
                                              TensorType]] = None,
            prev_reward_batch: Optional[Union[List[
                TensorType], TensorType]] = None) -> TensorType:

        if self.action_sampler_fn and self.action_distribution_fn is None:
            raise ValueError("Cannot compute log-prob/likelihood w/o an "
                             "`action_distribution_fn` and a provided "
                             "`action_sampler_fn`!")

        with torch.no_grad():
            input_dict = self._lazy_tensor_dict({
                SampleBatch.CUR_OBS: obs_batch,
                SampleBatch.ACTIONS: actions
            })
            if prev_action_batch is not None:
                input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
            if prev_reward_batch is not None:
                input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch
                # print(prev_reward_batch)
            seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)

            # Exploration hook before each forward pass.
            self.exploration.before_compute_actions(explore=False)

            # Action dist class and inputs are generated via custom function.
            if self.action_distribution_fn:
                dist_inputs, dist_class, _ = self.action_distribution_fn(
                    policy=self,
                    model=self.model,
                    obs_batch=input_dict[SampleBatch.CUR_OBS],
                    explore=False,
                    is_training=False)
            # Default action-dist inputs calculation.
            else:
                dist_class = self.dist_class
                dist_inputs, _ = self.model(input_dict, state_batches,
                                            seq_lens)

            action_dist = dist_class(dist_inputs, self.model)
            log_likelihoods = action_dist.logp(input_dict[SampleBatch.ACTIONS])
            return log_likelihoods

    @override(Policy)
    @DeveloperAPI
    def learn_on_batch(
            self, postprocessed_batch: SampleBatch) -> Dict[str, TensorType]:
        # Get batch ready for RNNs, if applicable.
        pad_batch_to_sequences_of_same_size(
            postprocessed_batch,
            max_seq_len=self.max_seq_len,
            shuffle=False,
            batch_divisibility_req=self.batch_divisibility_req)

        train_batch = self._lazy_tensor_dict(postprocessed_batch)
        loss_out = force_list(
            self._loss(self, self.model, self.dist_class, train_batch))
        # Call Model's custom-loss with Policy loss outputs and train_batch.
        if self.model:
            loss_out = self.model.custom_loss(loss_out, train_batch)
        assert len(loss_out) == len(self._optimizers)
        # assert not any(torch.isnan(l) for l in loss_out)
        fetches = self.extra_compute_grad_fetches()

        # Loop through all optimizers.
        grad_info = {"allreduce_latency": 0.0}
        for i, opt in enumerate(self._optimizers):
            # Erase gradients in all vars of this optimizer.
            opt.zero_grad()
            # Recompute gradients of loss over all variables.
            loss_out[i].backward(retain_graph=(i < len(self._optimizers) - 1))
            grad_info.update(self.extra_grad_process(opt, loss_out[i]))

            if self.distributed_world_size:
                grads = []
                for param_group in opt.param_groups:
                    for p in param_group["params"]:
                        if p.grad is not None:
                            grads.append(p.grad)

                start = time.time()
                if torch.cuda.is_available():
                    # Sadly, allreduce_coalesced does not work with CUDA yet.
                    for g in grads:
                        torch.distributed.all_reduce(
                            g, op=torch.distributed.ReduceOp.SUM)
                else:
                    torch.distributed.all_reduce_coalesced(
                        grads, op=torch.distributed.ReduceOp.SUM)

                for param_group in opt.param_groups:
                    for p in param_group["params"]:
                        if p.grad is not None:
                            p.grad /= self.distributed_world_size

                grad_info["allreduce_latency"] += time.time() - start

            # Step the optimizer.
            opt.step()

        grad_info["allreduce_latency"] /= len(self._optimizers)
        grad_info.update(self.extra_grad_info(train_batch))
        return dict(fetches, **{LEARNER_STATS_KEY: grad_info})

    @override(Policy)
    @DeveloperAPI
    def compute_gradients(self,
                          postprocessed_batch: SampleBatch) -> ModelGradients:
        train_batch = self._lazy_tensor_dict(postprocessed_batch)
        loss_out = force_list(
            self._loss(self, self.model, self.dist_class, train_batch))
        assert len(loss_out) == len(self._optimizers)
        fetches = self.extra_compute_grad_fetches()

        grad_process_info = {}
        grads = []
        for i, opt in enumerate(self._optimizers):
            opt.zero_grad()
            loss_out[i].backward()
            grad_process_info = self.extra_grad_process(opt, loss_out[i])

            # Note that return values are just references;
            # calling zero_grad will modify the values
            for param_group in opt.param_groups:
                for p in param_group["params"]:
                    if p.grad is not None:
                        grads.append(p.grad.data.cpu().numpy())
                    else:
                        grads.append(None)

        grad_info = self.extra_grad_info(train_batch)
        grad_info.update(grad_process_info)
        return grads, dict(fetches, **{LEARNER_STATS_KEY: grad_info})

    @override(Policy)
    @DeveloperAPI
    def apply_gradients(self, gradients: ModelGradients) -> None:
        # TODO(sven): Not supported for multiple optimizers yet.
        assert len(self._optimizers) == 1
        for g, p in zip(gradients, self.model.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g).to(self.device)

        self._optimizers[0].step()

    @override(Policy)
    @DeveloperAPI
    def get_weights(self) -> ModelWeights:
        return {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }

    @override(Policy)
    @DeveloperAPI
    def set_weights(self, weights: ModelWeights) -> None:
        weights = convert_to_torch_tensor(weights, device=self.device)
        self.model.load_state_dict(weights)

    @override(Policy)
    @DeveloperAPI
    def is_recurrent(self) -> bool:
        return len(self.model.get_initial_state()) > 0

    @override(Policy)
    @DeveloperAPI
    def num_state_tensors(self) -> int:
        return len(self.model.get_initial_state())

    @override(Policy)
    @DeveloperAPI
    def get_initial_state(self) -> List[TensorType]:
        return [
            s.cpu().detach().numpy() for s in self.model.get_initial_state()
        ]

    @override(Policy)
    @DeveloperAPI
    def get_state(self) -> Union[Dict[str, TensorType], List[TensorType]]:
        state = super().get_state()
        state["_optimizer_variables"] = []
        for i, o in enumerate(self._optimizers):
            state["_optimizer_variables"].append(o.state_dict())
        return state

    @override(Policy)
    @DeveloperAPI
    def set_state(self, state: object) -> None:
        state = state.copy()  # shallow copy
        # Set optimizer vars first.
        optimizer_vars = state.pop("_optimizer_variables", None)
        if optimizer_vars:
            assert len(optimizer_vars) == len(self._optimizers)
            for o, s in zip(self._optimizers, optimizer_vars):
                o.load_state_dict(s)
        # Then the Policy's (NN) weights.
        super().set_state(state)

    @DeveloperAPI
    def extra_grad_process(self, optimizer: "torch.optim.Optimizer",
                           loss: TensorType):
        """Called after each optimizer.zero_grad() + loss.backward() call.

        Called for each self._optimizers/loss-value pair.
        Allows for gradient processing before optimizer.step() is called.
        E.g. for gradient clipping.

        Args:
            optimizer (torch.optim.Optimizer): A torch optimizer object.
            loss (TensorType): The loss tensor associated with the optimizer.

        Returns:
            Dict[str, TensorType]: An dict with information on the gradient
                processing step.
        """
        return {}

    @DeveloperAPI
    def extra_compute_grad_fetches(self) -> Dict[str, any]:
        """Extra values to fetch and return from compute_gradients().

        Returns:
            Dict[str, any]: Extra fetch dict to be added to the fetch dict
                of the compute_gradients call.
        """
        return {LEARNER_STATS_KEY: {}}  # e.g, stats, td error, etc.

    @DeveloperAPI
    def extra_action_out(
            self, input_dict: Dict[str, TensorType],
            state_batches: List[TensorType], model: TorchModelV2,
            action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:
        """Returns dict of extra info to include in experience batch.

        Args:
            input_dict (Dict[str, TensorType]): Dict of model input tensors.
            state_batches (List[TensorType]): List of state tensors.
            model (TorchModelV2): Reference to the model object.
            action_dist (TorchDistributionWrapper): Torch action dist object
                to get log-probs (e.g. for already sampled actions).

        Returns:
            Dict[str, TensorType]: Extra outputs to return in a
                compute_actions() call (3rd return value).
        """
        return {}

    @DeveloperAPI
    def extra_grad_info(self,
                        train_batch: SampleBatch) -> Dict[str, TensorType]:
        """Return dict of extra grad info.

        Args:
            train_batch (SampleBatch): The training batch for which to produce
                extra grad info for.

        Returns:
            Dict[str, TensorType]: The info dict carrying grad info per str
                key.
        """
        return {}

    @DeveloperAPI
    def optimizer(
            self
    ) -> Union[List["torch.optim.Optimizer"], "torch.optim.Optimizer"]:
        """Custom the local PyTorch optimizer(s) to use.

        Returns:
            Union[List[torch.optim.Optimizer], torch.optim.Optimizer]:
                The local PyTorch optimizer(s) to use for this Policy.
        """
        if hasattr(self, "config"):
            return torch.optim.Adam(
                self.model.parameters(), lr=self.config["lr"])
        else:
            return torch.optim.Adam(self.model.parameters())

    @override(Policy)
    @DeveloperAPI
    def export_model(self, export_dir: str) -> None:
        """TODO(sven): implement for torch.
        """
        raise NotImplementedError

    @override(Policy)
    @DeveloperAPI
    def export_checkpoint(self, export_dir: str) -> None:
        """TODO(sven): implement for torch.
        """
        raise NotImplementedError

    @override(Policy)
    @DeveloperAPI
    def import_model_from_h5(self, import_file: str) -> None:
        """Imports weights into torch model."""
        return self.model.import_from_h5(import_file)

    def _lazy_tensor_dict(self, postprocessed_batch):
        train_batch = UsageTrackingDict(postprocessed_batch)
        train_batch.set_get_interceptor(
            functools.partial(convert_to_torch_tensor, device=self.device))
        return train_batch


# TODO: (sven) Unify hyperparam annealing procedures across RLlib (tf/torch)
#   and for all possible hyperparams, not just lr.
@DeveloperAPI
class LearningRateSchedule:
    """Mixin for TFPolicy that adds a learning rate schedule."""

    @DeveloperAPI
    def __init__(self, lr, lr_schedule):
        self.cur_lr = lr
        if lr_schedule is None:
            self.lr_schedule = ConstantSchedule(lr, framework=None)
        else:
            self.lr_schedule = PiecewiseSchedule(
                lr_schedule, outside_value=lr_schedule[-1][-1], framework=None)

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(LearningRateSchedule, self).on_global_var_update(global_vars)
        self.cur_lr = self.lr_schedule.value(global_vars["timestep"])

    @override(TorchPolicy)
    def optimizer(self):
        for opt in self._optimizers:
            for p in opt.param_groups:
                p["lr"] = self.cur_lr
        return self._optimizers


@DeveloperAPI
class EntropyCoeffSchedule:
    """Mixin for TorchPolicy that adds entropy coeff decay."""

    @DeveloperAPI
    def __init__(self, entropy_coeff, entropy_coeff_schedule):
        self.entropy_coeff = entropy_coeff

        if entropy_coeff_schedule is None:
            self.entropy_coeff_schedule = ConstantSchedule(
                entropy_coeff, framework=None)
        else:
            # Allows for custom schedule similar to lr_schedule format
            if isinstance(entropy_coeff_schedule, list):
                self.entropy_coeff_schedule = PiecewiseSchedule(
                    entropy_coeff_schedule,
                    outside_value=entropy_coeff_schedule[-1][-1],
                    framework=None)
            else:
                # Implements previous version but enforces outside_value
                self.entropy_coeff_schedule = PiecewiseSchedule(
                    [[0, entropy_coeff], [entropy_coeff_schedule, 0.0]],
                    outside_value=0.0,
                    framework=None)

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(EntropyCoeffSchedule, self).on_global_var_update(global_vars)
        self.entropy_coeff = self.entropy_coeff_schedule.value(
            global_vars["timestep"])
