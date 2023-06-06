#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import numpy as np
import torch
from gym import spaces
from habitat_baselines.common.tensor_dict import TensorDict
from torch.utils.data import BatchSampler, SubsetRandomSampler


class RolloutStorage:
    r"""Class for storing rollout information for RL trainers.

    Modification:
    - remove double buffers
    - remove the arg @discrete_actions and processing for ActionSpace
    - support truncated episodes

    Reference:
        habitat-lab/habitat_baselines/common/rollout_storage.py
    """

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=0,
    ):
        self.buffers = TensorDict()

        self.buffers["observations"] = TensorDict()
        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )

        # Dict action space should be preprocessed in the wrapper.
        if isinstance(action_space, spaces.Box):
            action_shape = action_space.shape
            action_dtype = torch.float32
        elif isinstance(action_space, spaces.Discrete):
            action_shape = (1,)
            action_dtype = torch.long
        else:
            raise TypeError(type(action_space))

        self.buffers["actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape, dtype=action_dtype
        )
        self.buffers["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape, dtype=action_dtype
        )

        # @masks indicates whether the current step is not after done.
        # The first step of an episode is False, otherwise True
        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.buffers["next_value_preds"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )
        self.buffers["truncated_masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.num_envs = num_envs
        self.numsteps = numsteps
        self.step_idx = 0

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device, non_blocking=True))

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        next_value_preds=None,
        truncated_masks=None,
    ):
        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
            next_value_preds=next_value_preds,
            truncated_masks=truncated_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        if len(next_step) > 0:
            self.buffers.set(self.step_idx + 1, next_step, strict=False)

        if len(current_step) > 0:
            self.buffers.set(self.step_idx, current_step, strict=False)

    def advance(self):
        self.step_idx += 1

    def after_update(self):
        self.buffers[0] = self.buffers[self.step_idx]
        self.step_idx = 0

    def is_full(self):
        return self.step_idx >= self.numsteps

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.buffers["value_preds"][self.step_idx] = next_value
            gae = 0
            for step in reversed(range(self.step_idx)):
                # if self.buffers["truncated_masks"][step + 1]:
                #     print(self.buffers["next_value_preds"][step + 1])
                #     assert not self.buffers["masks"][step + 1]
                delta = (
                    self.buffers["rewards"][step]
                    + gamma
                    * (
                        self.buffers["value_preds"][step + 1]
                        * self.buffers["masks"][step + 1]
                        + self.buffers["next_value_preds"][step + 1]
                        * self.buffers["truncated_masks"][step + 1]
                    )
                    - self.buffers["value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["returns"][step] = (
                    gae + self.buffers["value_preds"][step]
                )
        else:
            self.buffers["returns"][self.step_idx] = next_value
            for step in reversed(range(self.step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * (
                        self.buffers["returns"][step + 1]
                        * self.buffers["masks"][step + 1]
                        + self.buffers["next_value_preds"][step + 1]
                        * self.buffers["truncated_masks"][step + 1]
                    )
                    + self.buffers["rewards"][step]
                )

    def get_advantages(self, use_normalized_advantage, eps=1e-5):
        advantages: torch.Tensor = (
            self.buffers["returns"][: self.step_idx]
            - self.buffers["value_preds"][: self.step_idx]
        )

        if not use_normalized_advantage:
            return advantages

        adv_mean = advantages.mean()
        adv_std = advantages.std()

        return (advantages - adv_mean) / (adv_std + eps)

    @torch.no_grad()
    def feed_forward_generator(self, advantages, mini_batch_size):
        batch_size = self.num_envs * self.step_idx
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True,
        )

        batch = self.buffers[0 : self.step_idx]
        batch["advantages"] = advantages[0 : self.step_idx]
        batch = batch.map(lambda v: v.flatten(0, 1))
        for indices in sampler:
            yield batch[indices]

    @torch.no_grad()
    def recurrent_generator(self, advantages, num_mini_batch) -> TensorDict:
        # num_environments = advantages.size(1)
        num_environments = self.num_envs
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.step_idx, inds]
            batch["advantages"] = advantages[0 : self.step_idx, inds]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]
            yield batch.map(lambda v: v.flatten(0, 1))

class BilinearRolloutStorage:
    r"""Class for storing rollout information for RL trainers.

    Modification:
    - remove double buffers
    - remove the arg @discrete_actions and processing for ActionSpace
    - support truncated episodes

    Reference:
        habitat-lab/habitat_baselines/common/rollout_storage.py
    """

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=0,
    ):
        self.buffers = TensorDict()

        self.buffers["observations"] = TensorDict()
        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["goal_dep_recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        self.buffers["goal_ind_recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )

        # Dict action space should be preprocessed in the wrapper.
        if isinstance(action_space, spaces.Box):
            action_shape = action_space.shape
            action_dtype = torch.float32
        elif isinstance(action_space, spaces.Discrete):
            action_shape = (1,)
            action_dtype = torch.long
        else:
            raise TypeError(type(action_space))

        self.buffers["actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape, dtype=action_dtype
        )
        self.buffers["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape, dtype=action_dtype
        )

        self.buffers["grasped"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=action_dtype
        )

        self.buffers["prev_grasped"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=action_dtype
        )

        # @masks indicates whether the current step is not after done.
        # The first step of an episode is False, otherwise True
        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.buffers["next_value_preds"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )
        self.buffers["truncated_masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.num_envs = num_envs
        self.numsteps = numsteps
        self.step_idx = 0

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device, non_blocking=True))

    def insert(
        self,
        next_observations=None,
        next_goal_dep_recurrent_hidden_states=None,
        next_goal_ind_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        next_value_preds=None,
        truncated_masks=None,
    ):
        next_step = dict(
            observations=next_observations,
            goal_dep_recurrent_hidden_states=next_goal_dep_recurrent_hidden_states,
            goal_ind_recurrent_hidden_states=next_goal_ind_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
            next_value_preds=next_value_preds,
            truncated_masks=truncated_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )
        if next_observations and "is_grasped" in next_observations:
            next_step["prev_grasped"] = next_observations["is_grasped"]
            current_step["grasped"] = next_observations["is_grasped"]

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        if len(next_step) > 0:
            self.buffers.set(self.step_idx + 1, next_step, strict=False)

        if len(current_step) > 0:
            self.buffers.set(self.step_idx, current_step, strict=False)

    def advance(self):
        self.step_idx += 1

    def after_update(self):
        self.buffers[0] = self.buffers[self.step_idx]
        self.step_idx = 0

    def is_full(self):
        return self.step_idx >= self.numsteps

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.buffers["value_preds"][self.step_idx] = next_value
            gae = 0
            for step in reversed(range(self.step_idx)):
                # if self.buffers["truncated_masks"][step + 1]:
                #     print(self.buffers["next_value_preds"][step + 1])
                #     assert not self.buffers["masks"][step + 1]
                delta = (
                    self.buffers["rewards"][step]
                    + gamma
                    * (
                        self.buffers["value_preds"][step + 1]
                        * self.buffers["masks"][step + 1]
                        + self.buffers["next_value_preds"][step + 1]
                        * self.buffers["truncated_masks"][step + 1]
                    )
                    - self.buffers["value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["returns"][step] = (
                    gae + self.buffers["value_preds"][step]
                )
        else:
            self.buffers["returns"][self.step_idx] = next_value
            for step in reversed(range(self.step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * (
                        self.buffers["returns"][step + 1]
                        * self.buffers["masks"][step + 1]
                        + self.buffers["next_value_preds"][step + 1]
                        * self.buffers["truncated_masks"][step + 1]
                    )
                    + self.buffers["rewards"][step]
                )

    def get_advantages(self, use_normalized_advantage, eps=1e-5):
        advantages: torch.Tensor = (
            self.buffers["returns"][: self.step_idx]
            - self.buffers["value_preds"][: self.step_idx]
        )

        if not use_normalized_advantage:
            return advantages

        adv_mean = advantages.mean()
        adv_std = advantages.std()

        return (advantages - adv_mean) / (adv_std + eps)

    @torch.no_grad()
    def feed_forward_generator(self, advantages, mini_batch_size):
        batch_size = self.num_envs * self.step_idx
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True,
        )

        batch = self.buffers[0 : self.step_idx]
        batch["advantages"] = advantages[0 : self.step_idx]
        batch = batch.map(lambda v: v.flatten(0, 1))
        for indices in sampler:
            yield batch[indices]

    @torch.no_grad()
    def recurrent_generator(self, advantages, num_mini_batch) -> TensorDict:
        # num_environments = advantages.size(1)
        num_environments = self.num_envs
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.step_idx, inds]
            batch["advantages"] = advantages[0 : self.step_idx, inds]
            batch["goal_dep_recurrent_hidden_states"] = batch[
                "goal_dep_recurrent_hidden_states"
            ][0:1]
            batch["goal_ind_recurrent_hidden_states"] = batch[
                "goal_ind_recurrent_hidden_states"
            ][0:1]
            yield batch.map(lambda v: v.flatten(0, 1))

class MultiheadRolloutStorage:
    r"""Class for storing rollout information for RL trainers.

    Modification:
    - remove double buffers
    - remove the arg @discrete_actions and processing for ActionSpace
    - support truncated episodes

    Reference:
        habitat-lab/habitat_baselines/common/rollout_storage.py
    """

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        gaussian_action_space,
        categorical_action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=0,
    ):
        self.buffers = TensorDict()

        self.buffers["observations"] = TensorDict()
        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1, dtype=torch.float16)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["gaussian_action_log_probs"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )
        
        self.buffers["categorical_action_log_probs"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )

        # Dict action space should be preprocessed in the wrapper.
        if isinstance(gaussian_action_space, spaces.Box):
            gaussian_action_shape = gaussian_action_space.shape
            gaussian_action_dtype = torch.float32
        elif isinstance(gaussian_action_space, spaces.Discrete):
            gaussian_action_shape = (1,)
            gaussian_action_dtype = torch.long
        else:
            raise TypeError(type(gaussian_action_space))
        
        # Dict action space should be preprocessed in the wrapper.
        if isinstance(categorical_action_space, spaces.Box):
            categorical_action_shape = gaussian_action_space.shape
            categorical_action_dtype = torch.float32
        elif isinstance(categorical_action_space, spaces.Discrete):
            categorical_action_shape = (1,)
            categorical_action_dtype = torch.long
        else:
            raise TypeError(type(categorical_action_space))

        self.buffers["gaussian_actions"] = torch.zeros(
            numsteps + 1, num_envs, *gaussian_action_shape, dtype=gaussian_action_dtype
        )
        self.buffers["categorical_actions"] = torch.zeros(
            numsteps + 1, num_envs, *categorical_action_shape, dtype=categorical_action_dtype
        )
        self.buffers["gaussian_flag"] = torch.ones(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )
        # @masks indicates whether the current step is not after done.
        # The first step of an episode is False, otherwise True
        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.buffers["next_value_preds"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )
        self.buffers["truncated_masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.num_envs = num_envs
        self.numsteps = numsteps
        self.step_idx = 0

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device, non_blocking=True))

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        gaussian_actions=None,
        gaussian_action_log_probs=None,
        categorical_actions=None,
        categorical_action_log_probs=None,
        gaussian_flag=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        next_value_preds=None,
        truncated_masks=None,
    ):
        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            masks=next_masks,
            next_value_preds=next_value_preds,
            truncated_masks=truncated_masks,
        )

        current_step = dict(
            gaussian_actions=gaussian_actions,
            gaussian_action_log_probs=gaussian_action_log_probs,
            categorical_actions=categorical_actions,
            categorical_action_log_probs=categorical_action_log_probs,
            gaussian_flag=gaussian_flag,
            value_preds=value_preds,
            rewards=rewards,
        )
        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        if len(next_step) > 0:
            self.buffers.set(self.step_idx + 1, next_step, strict=False)

        if len(current_step) > 0:
            self.buffers.set(self.step_idx, current_step, strict=False)

    def advance(self):
        self.step_idx += 1

    def after_update(self):
        self.buffers[0] = self.buffers[self.step_idx]
        self.step_idx = 0

    def is_full(self):
        return self.step_idx >= self.numsteps

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.buffers["value_preds"][self.step_idx] = next_value
            gae = 0
            for step in reversed(range(self.step_idx)):
                # if self.buffers["truncated_masks"][step + 1]:
                #     print(self.buffers["next_value_preds"][step + 1])
                #     assert not self.buffers["masks"][step + 1]
                delta = (
                    self.buffers["rewards"][step]
                    + gamma
                    * (
                        self.buffers["value_preds"][step + 1]
                        * self.buffers["masks"][step + 1]
                        + self.buffers["next_value_preds"][step + 1]
                        * self.buffers["truncated_masks"][step + 1]
                    )
                    - self.buffers["value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["returns"][step] = (
                    gae + self.buffers["value_preds"][step]
                )
        else:
            self.buffers["returns"][self.step_idx] = next_value
            for step in reversed(range(self.step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * (
                        self.buffers["returns"][step + 1]
                        * self.buffers["masks"][step + 1]
                        + self.buffers["next_value_preds"][step + 1]
                        * self.buffers["truncated_masks"][step + 1]
                    )
                    + self.buffers["rewards"][step]
                )

    def get_advantages(self, use_normalized_advantage, eps=1e-5):
        advantages: torch.Tensor = (
            self.buffers["returns"][: self.step_idx]
            - self.buffers["value_preds"][: self.step_idx]
        )

        if not use_normalized_advantage:
            return advantages

        adv_mean = advantages.mean()
        adv_std = advantages.std()

        return (advantages - adv_mean) / (adv_std + eps)

    @torch.no_grad()
    def feed_forward_generator(self, advantages, mini_batch_size):
        batch_size = self.num_envs * self.step_idx
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True,
        )

        batch = self.buffers[0 : self.step_idx]
        batch["advantages"] = advantages[0 : self.step_idx]
        batch = batch.map(lambda v: v.flatten(0, 1))
        for indices in sampler:
            yield batch[indices]

    @torch.no_grad()
    def recurrent_generator(self, advantages, num_mini_batch) -> TensorDict:
        # num_environments = advantages.size(1)
        num_environments = self.num_envs
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.step_idx, inds]
            batch["advantages"] = advantages[0 : self.step_idx, inds]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]
            yield batch.map(lambda v: v.flatten(0, 1))
