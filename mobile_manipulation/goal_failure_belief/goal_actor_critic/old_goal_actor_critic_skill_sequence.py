from ..task_goal_critic.task_success_value_network import (
    TaskGoalActorCriticModel, 
    load_goal_critic_model_only,
    load_goal_actor_critic_model,
) 
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    batch_obs,
)
from mobile_manipulation.utils.common import Timer

import numpy as np
import os
import random
from gym import spaces

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from habitat import skill_config
from habitat.core.environments import get_env_class
from habitat_baselines.common.tensor_dict import TensorDict
import pickle
import datetime
from torch.optim.lr_scheduler import LambdaLR
from mobile_manipulation.common.rollout_storage import RolloutStorage
from mobile_manipulation.utils.env_utils import (
    VectorEnv,
    construct_envs,
    make_env_fn,
)
from mobile_manipulation.utils.wrappers import HabitatActionWrapperV1
from mobile_manipulation.methods.skill import CompositeSkill

MAX_STEPS = 600
NAV_OB_KEYS = ["robot_head_depth", "nav_goal_at_base"]

class TrainGoalActorCritic:

    def __init__(self, observation_space, action_space, device, prefix, composite_skill_config, skill_config, base_policy, lr=2.5e-4, reload_critic=False, reload_actor_critic=False, max_grad_norm=0.5, size_batch_sub_trajs=32, runtype="", max_iterations=1e6) -> None:
        self._goal_actor_critic = TaskGoalActorCriticModel(observation_space, action_space).to(device)
        self._observation_space = observation_space
        self._action_space = action_space
        self._device = device
        self._composite_skill_config = composite_skill_config
        self._skill_config = skill_config
        self._base_policies = []

        current_file_path = os.path.abspath(__file__)
        parent_dir = os.path.abspath(os.path.join(current_file_path, os.pardir))
        critic_dir = os.path.join(os.path.abspath(os.path.join(parent_dir, os.pardir)), "task_goal_critic")
        self._critic_dir = critic_dir
        self._writer = SummaryWriter(os.path.join(parent_dir, f'{runtype}logs'))
        self._num_steps_done = 0
        self._losses = 0
        self._num_sub_traj = 0
        self._saved_model_dir = os.path.join(parent_dir, f'{runtype}models_{prefix}')
        self._max_grad_norm = max_grad_norm
        self._size_batch_sub_trajs = size_batch_sub_trajs
        self._runtype = runtype
        self._num_steps_done = 0
        self._num_updates_done = 0
        if not os.path.exists(self._saved_model_dir):
            os.mkdir(self._saved_model_dir)
        self._latest_path = os.path.join(self._saved_model_dir, f"model.pt")

        if reload_critic:
            self._reload_critic_model(self._critic_dir)
        elif reload_actor_critic:
            self._reload_complete_model(self._saved_model_dir)

        self._lr_lambda = lambda x: max(
            self.percent_done(), 0.0
        )
        self._lr_scheduler = LambdaLR(
            optimizer=self._optimizer, lr_lambda=self._lr_lambda
        )

        self.timer = Timer()  # record fine-grained scopes
        self._optimizer = optim.Adam(self._goal_actor_critic.parameters(), lr=lr)

        # ---------------------------------------------------------------------------- #
        env = make_env_fn(
            self._skill_config,
            get_env_class(self._skill_config.ENV_NAME),
            wrappers=[HabitatActionWrapperV1],
        )
        self.envs = [env]
        env.close()
        # ---------------------------------------------------------------------------- #

        ### Compute observation space and action space for the goal actor critic
        new_obs_space_dict = self._observation_space.to_dict()
        new_obs_space_dict["next_skill"] = spaces.Box(0, 1, (len(skill_ordering),), np.int32)
        self._goal_observation_space = spaces.Dict(new_obs_space_dict)
        self._goal_action_space = spaces.Box(-np.inf, np.inf, (3,), np.int32)

        self._init_envs_and_policies(self._skill_config)
        self._setup_rollouts(self._skill_config)
    
    def _reload_critic_model(self, dir: str):
        self._goal_actor_critic = load_goal_critic_model_only(self._goal_actor_critic, os.path.join(dir, "model.pt"))

    def _reload_complete_model(self, dir: str):
        self._goal_actor_critic = load_goal_actor_critic_model(self._goal_actor_critic, os.path.join(dir, "model.pt"))
    
    def _init_envs_and_policies(self, skill_config: skill_config, auto_reset_done=False):
        r"""Initialize vectorized environments."""
        self.envs = construct_envs(
            skill_config,
            get_env_class(skill_config.ENV_NAME),
            split_dataset=skill_config.get("SPLIT_DATASET", True),
            workers_ignore_signals=False,
            auto_reset_done=auto_reset_done,
            wrappers=[HabitatActionWrapperV1],
        )
        self._base_policies = [CompositeSkill(self._composite_skill_config, env) for env in self.envs]
        self._base_policies = nn.DataParallel(self._base_policies)

    def _setup_rollouts(self, skill_config: skill_config):
        ppo_cfg = skill_config.RL.PPO
        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,  # number of steps for each env
            self.envs.num_envs,
            observation_space=self._goal_observation_space,
            action_space=self._goal_action_space,
            recurrent_hidden_state_size=self._goal_actor_critic._crnn.rnn_hidden_size,
            num_recurrent_layers=self._goal_actor_critic._crnn.num_recurrent_layers,
        )
        self.rollouts.to(self._device)

    def _init_obs_batch(self):
        self._obs_batching_cache = ObservationBatchingCache()
        observations = self.envs.reset()
        self._last_obs = observations
        self._last_obs_batch = batch_obs(
            observations, device=self._device, cache=self._obs_batching_cache
        )

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self._goal_actor_critic.parameters(), self._max_grad_norm
        )

    def _compute_ground_truth_values(self, sub_traj_lengths, discounted_returns, batch_size=1):
        """
        Assume discounted_returns has length batch_size
        """
        ground_truth = np.repeat(np.repeat(discounted_returns, sub_traj_lengths).reshape(-1, 1), repeats=batch_size, axis=1)
        return torch.from_numpy(ground_truth).to(dtype=torch.float32, device=self._device)

    def save_train_info(self):
        # log the performance metrics
        train_loss = self._losses/(self._num_sub_traj)
        self._writer.add_scalar('training_loss', train_loss, self._num_steps_done)
        save_dict = {
            'model_state_dict': self._goal_actor_critic.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'num_steps_done': self._num_steps_done,
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.random.get_rng_state(),
            'py_random_state': random.getstate()
        }
        torch.save(save_dict, os.path.join(self._saved_model_dir, f"model.pt"))

        if self._num_steps_done % 1000 == 0:
            torch.save(save_dict, os.path.join(self._saved_model_dir, f"checkpoint_{self._num_steps_done}.pt"))

        self._losses = 0
        self._num_sub_traj = 0
        self._lr_scheduler.step()

    def is_done(self):
        return self._num_steps_done >= self._skill_config.TOTAL_NUM_STEPS

    def percent_done(self):
        return self._num_steps_done / self._skill_config.TOTAL_NUM_STEPS

    def modify_env_goals(self, goals):
        for i in range(len(goals)):
            self.envs[i].env._env.task.place_goal = goals[i]
    
    def is_current_skill_navigate(self, policy_index):
        return self._base_policies[policy_index].current_skill_name == "NavRLSkill"

    @torch.no_grad()
    def step(self):
        """Take one rollout step."""
        n_envs = self.envs.num_envs

        with self.timer.timeit("sample_goal"):
            goals = self._goal_actor_critic(self._last_obs_batch)
            self.modify_env_goals(goals)
            self._last_obs_batch = batch_obs(
                self._last_obs, device=self._device
            )

        with self.timer.timeit("sample_action"):
            # Assume that observations are stored at rollouts in the last step
            actions = self._base_policies.module.act(self._last_obs_batch)
            actions = actions.to(device="cpu", non_blocking=True)

        with self.timer.timeit("step_env"):
            for i_env, action in zip(range(n_envs), actions.unbind(0)):
                self.envs.async_step_at(i_env, {"action": action.numpy()})

        with self.timer.timeit("update_rollout"):
            self.rollouts.insert(
                next_recurrent_hidden_states=output_batch.get(
                    "rnn_hidden_states"
                ),
                actions=output_batch["action"],
                action_log_probs=output_batch["action_log_probs"],
                value_preds=output_batch["value"],
            )

        with self.timer.timeit("step_env"):
            results = self.envs.wait_step()
            obs, rews, dones, infos = map(list, zip(*results))
            self._num_steps_done += n_envs

            self._last_obs_batch =  batch_obs(
                    obs, device=self._device, cache=self._obs_batching_cache
                )
            self._last_obs = obs

        with self.timer.timeit("update_rollout"):
            self.rollouts.insert(
                next_recurrent_hidden_states=output_batch.get(
                    "rnn_hidden_states"
                ),
                actions=output_batch["action"],
                action_log_probs=output_batch["action_log_probs"],
                value_preds=output_batch["value"],
            )

        with self.timer.timeit("step_env"):
            results = self.envs.wait_step()
            obs, rews, dones, infos = map(list, zip(*results))
            self.num_steps_done += n_envs

        # self.envs.render("human", delay=10)

        # -------------------------------------------------------------------------- #
        # Reset and deal with truncated episodes
        # -------------------------------------------------------------------------- #
        next_value = None
        are_truncated = [False for _ in range(n_envs)]
        ignore_truncated = self.config.RL.get("IGNORE_TRUNCATED", False)

        if any(dones):
            # Check which envs are truncated
            for i_env in range(n_envs):
                if dones[i_env]:
                    self.envs.async_reset_at(i_env)
                    is_truncated = infos[i_env].get(
                        "is_episode_truncated", False
                    )
                    are_truncated[i_env] = is_truncated

            if ignore_truncated:
                are_truncated = [False for _ in range(n_envs)]

            # Estimate values of actual next obs
            if any(are_truncated):
                next_batch = batch_obs(
                    obs,
                    device=self.device,
                    cache=self._obs_batching_cache,
                )
                next_step_batch = self.rollouts.buffers[
                    self.rollouts.step_idx + 1
                ]
                next_step_batch["observations"] = next_batch
                # Only the really truncated episodes have valid results
                next_step_batch["masks"] = torch.ones_like(
                    next_step_batch["masks"]
                )
                next_value = self.actor_critic.get_value(next_step_batch)

            for i_env in range(n_envs):
                if dones[i_env]:
                    obs[i_env] = self.envs.wait_reset_at(i_env)
        # -------------------------------------------------------------------------- #

        with self.timer.timeit("batch_obs"):
            batch = batch_obs(
                obs, device=self.device, cache=self._obs_batching_cache
            )
            rewards = torch.tensor(rews, dtype=torch.float).unsqueeze(1)
            done_masks = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
            not_done_masks = torch.logical_not(done_masks)
            truncated_masks = torch.tensor(
                are_truncated, dtype=torch.bool
            ).unsqueeze(1)

        with self.timer.timeit("update_stats"):
            for i_env in range(n_envs):
                self.episode_rewards[i_env] += rews[i_env]
                if dones[i_env]:
                    episode_info = self._extract_scalars_from_info(
                        infos[i_env]
                    )
                    episode_info["return"] = self.episode_rewards[i_env].item()
                    self.window_episode_stats.append(episode_info)
                    self.episode_rewards[i_env] = 0.0

        with self.timer.timeit("update_rollout"):
            self.rollouts.insert(
                next_observations=batch,
                rewards=rewards,
                next_masks=not_done_masks,
                next_value_preds=next_value,
                truncated_masks=truncated_masks,
            )
            self.rollouts.advance()
    
    def train(self, reload: bool) -> None:
        ppo_cfg = self._skill_config.RL.PPO
        self._init_obs_batch()
        if reload:
            state_dict = torch.load(os.path.join(self._saved_model_dir, f"model.pt"))
            self._goal_actor_critic.load_state_dict(state_dict["model_state_dict"])
            self._optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self._num_steps_done = state_dict["num_steps_done"]

        if ppo_cfg.use_linear_lr_decay:
            min_lr = ppo_cfg.get("min_lr", 0.0)
            min_lr_ratio = min_lr / ppo_cfg.lr
            lr_lambda = lambda x: max(
                1 - self.percent_done() * (1.0 - min_lr_ratio), min_lr_ratio
            )
            lr_scheduler = LambdaLR(
                optimizer=self._optimizer, lr_lambda=lr_lambda
            )

        while not self.is_done():
            # Rollout and collect transitions
            self._goal_actor_critic.eval()
            for _ in range(ppo_cfg.num_steps):
                self.step()

            with self.timer.timeit("update_model"):
                self._goal_actor_critic.train()
                losses, metrics = self.update()
                self.rollouts.after_update()

            # Logging
            episode_metrics = self.get_episode_metrics()
            if self._num_updates_done % self._skill_config.LOG_INTERVAL == 0:
                self.log(episode_metrics)

            # Tensorboard
            metrics.update(**episode_metrics) 
            metrics["lr"] = self._optimizer.param_groups[0]["lr"]

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()

        self._writer.close()
        self.envs.close()
