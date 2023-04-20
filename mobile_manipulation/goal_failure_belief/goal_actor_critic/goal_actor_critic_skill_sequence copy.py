from ..task_goal_critic.task_success_value_network import (
    GoalPredictorPolicy, 
    load_goal_critic_model_only,
    load_goal_actor_critic_model,
) 
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    batch_obs,
)
from mobile_manipulation.utils.common import (
    Timer,
    extract_scalars_from_info,
)

from collections import deque
import numpy as np
import os
import random
from typing import Dict

from gym import spaces
from habitat import Config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from habitat.core.environments import get_env_class
from habitat_baselines.common.tensor_dict import TensorDict
import pickle
import datetime
from torch.optim.lr_scheduler import LambdaLR
from mobile_manipulation.common.registry import mm_registry
from mobile_manipulation.common.rollout_storage import RolloutStorage
from mobile_manipulation.utils.env_utils import (
    VectorEnv,
    construct_envs_multi_config,
    make_env_fn,
)
from mobile_manipulation.utils.wrappers import HabitatActionWrapperV1
from mobile_manipulation.methods.vec_skill import CompositeSkill

MAX_STEPS = 600
NAV_OB_KEYS = ["robot_head_depth", "nav_goal_at_base"]

class TrainGoalActorCritic:

    def __init__(self, observation_space, action_space, device, prefix, composite_task_config, skill_config, skill_sequence, lr=2.5e-4, max_grad_norm=0.5, size_batch_sub_trajs=32, runtype="") -> None:
        
        self._observation_space = observation_space
        self._action_space = action_space
        self._device = device
        self._composite_task_config = composite_task_config
        self._skill_config = skill_config
        self._skill_sequence = skill_sequence
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

        self._lr_lambda = lambda x: max(
            self.percent_done(), 0.0
        )
        # self._lr_scheduler = LambdaLR(
        #     optimizer=self._optimizer, lr_lambda=self._lr_lambda
        # )

        self.timer = Timer()  # record fine-grained scopes

        # ---------------------------------------------------------------------------- #
        env = make_env_fn(
            self._skill_config,
            get_env_class(self._skill_config.ENV_NAME),
            wrappers=[HabitatActionWrapperV1],
        )
        self.envs = [env]
        env.close()
        # ---------------------------------------------------------------------------- #
        self._init_envs_and_policies(self._skill_config)
        
        # Current episode rewards (return)
        self.episode_rewards = torch.zeros(self.envs.num_envs, 1)
        # Recent episode stats (each stat is a dict)
        self.window_episode_stats = deque(
            maxlen=self._skill_config.RL.PPO.reward_window_size
        )

        ### Compute observation space and action space for the goal actor critic
        new_obs_space_dict = self._observation_space
        new_obs_space_dict["next_skill"] = spaces.Box(0, 1, (len(self._skill_sequence),), np.int32)
        # new_obs_space_dict["current_goal"] = spaces.Box(0, 1, (len(self._skill_sequence),), np.int32)
        self._goal_observation_space = new_obs_space_dict
        self._goal_action_space = spaces.Box(-np.inf, np.inf, (3,), np.int32)
        self._goal_actor_critic = GoalPredictorPolicy.setup_policy(
            observation_space=self._goal_observation_space, 
            action_space=self._goal_action_space
        )

        self._setup_rollouts(self._skill_config)
        print("initialized trainer")

    def _init_envs_and_policies(self, auto_reset_done=False):
        r"""Initialize vectorized environments."""
        self.envs = construct_envs_multi_config(
            self._composite_task_config,
            self._skill_config,
            get_env_class(self._skill_config.ENV_NAME),
            split_dataset=self._skill_config.get("SPLIT_DATASET", True),
            workers_ignore_signals=False,
            auto_reset_done=auto_reset_done,
            wrappers=[HabitatActionWrapperV1],
        )
        self._base_policies = [CompositeSkill(self._composite_task_config.SOLUTION, self.envs, i) for i in range(self.envs.num_envs)]
        # self._base_policies = nn.DataParallel(self._base_policies)

    def _setup_rollouts(self, skill_config: Config):
        ppo_cfg = skill_config.RL.PPO
        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,  # number of steps for each env
            self.envs.num_envs,
            observation_space=self._goal_observation_space,
            action_space=self._goal_action_space,
            recurrent_hidden_state_size=self._goal_actor_critic.net.rnn_hidden_size,
            num_recurrent_layers=self._goal_actor_critic.net.num_recurrent_layers,
        )
        self.rollouts.to(self._device)

    def _init_train(self):
        if self.config.LOG_FILE:
            log_dir = os.path.dirname(self.config.LOG_FILE)
            os.makedirs(log_dir, exist_ok=True)
            logger.add_filehandler(self.config.LOG_FILE)

        if self.config.VERBOSE:
            logger.info(f"config:\n {self.config}")
            logger.info("commit id: {}".format(get_git_commit_id()))

        if self.config.CHECKPOINT_FOLDER:
            os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)

        # ---------------------------------------------------------------------------- #
        # Initialization
        # ---------------------------------------------------------------------------- #
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # ---------------------------------------------------------------------------- #
        # NOTE(jigu): workaround from erik, to avoid high gpu memory fragmentation
        env = make_env_fn(
            self.config,
            get_env_class(self.config.ENV_NAME),
            wrappers=[HabitatActionWrapper],
        )
        self.envs = [env]
        self._init_observation_space(self.config)
        self._init_action_space(self.config)
        self._setup_actor_critic(self.config)
        env.close()
        # ---------------------------------------------------------------------------- #

        self._init_envs(self.config)
        self._init_observation_space(self.config)
        self._init_action_space(self.config)
        self._setup_rollouts(self.config)

        if self.config.VERBOSE:
            logger.info(f"actor_critic: {self.actor_critic}")
        logger.info(
            "#parameters: {}".format(
                sum(param.numel() for param in self.actor_critic.parameters())
            )
        )
        logger.info("obs space: {}".format(self.obs_space))
        logger.info("action space: {}".format(self.action_space))

        # ---------------------------------------------------------------------------- #
        # Setup statistic
        # ---------------------------------------------------------------------------- #
        # Current episode rewards (return)
        self.episode_rewards = torch.zeros(self.envs.num_envs, 1)
        # Recent episode stats (each stat is a dict)
        self.window_episode_stats = deque(
            maxlen=self.config.RL.PPO.reward_window_size
        )

        self.t_start = time.time()  # record overall time
        self.timer = Timer()  # record fine-grained scopes
        self.writer = TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=30
        )

        # resumable stats
        self.num_steps_done = 0
        self.num_updates_done = 0
        self.prev_time = 0.0
        self.count_checkpoints = 0
        self.prev_ckpt_step = 0


    def _init_rollouts(self):
        self._obs_batching_cache = ObservationBatchingCache()
        observations = self.envs.reset()
        self._last_obs = observations
        self._stage_successes = np.zeros(self.envs.num_envs)
        batch = batch_obs(
            observations, device=self._device, cache=self._obs_batching_cache
        )
        self.rollouts.buffers["observations"][0] = batch

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self._goal_actor_critic.parameters(), self._max_grad_norm
        )

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
    
    @classmethod
    def _extract_scalars_from_info(cls, info: Dict):
        return extract_scalars_from_info(
            info, blacklist=["terminal_observation"]
        )

    def modify_env_goals(self, goals):
        # TODO: parallelize this
        for i in range(self.envs.num_envs):
            self.envs.call_at(i, "set_goal", {"goal": goals[i]})
        
        modify(self._last_obs, i) # here, use some pre-stored correspondence between the skill name and key in the observation missing

    @torch.no_grad()
    def step(self):
        """Take one rollout step."""
        n_envs = self.envs.num_envs
        
        ### Goal Sampling ###
        with self.timer.timeit("sample_goal"):
            step_batch = self.rollouts.buffers[self.rollouts.step_idx]
            output_batch = self._goal_actor_critic.act(step_batch)
            self.modify_env_goals(output_batch["action"])
        
        with self.timer.timeit("update_rollout"):
            self.rollouts.insert(
                next_recurrent_hidden_states=output_batch.get(
                    "rnn_hidden_states"
                ),
                actions=output_batch["action"],
                action_log_probs=output_batch["action_log_probs"],
                value_preds=output_batch["value"],
            )

        with self.timer.timeit("sample_action"):
            # Assume that observations are stored at rollouts in the last step
            actions = np.zeros((n_envs, self._action_space.n))
            for i in range(n_envs):
                actions[i] = self._base_policies[i].act(self._last_obs[i])
            actions = torch.from_numpy(actions).to(device="cpu", non_blocking=True)

        with self.timer.timeit("step_env"):
            for i_env, action in zip(range(n_envs), actions.unbind(0)):
                self.envs.async_step_at(i_env, {"action": action.numpy()})

        with self.timer.timeit("step_env"):
            results = self.envs.wait_step()
            obs, rews, dones, infos = map(list, zip(*results))
            self._num_steps_done += n_envs
        
        with self.timer.timeit("update_current_skill"):
            self._last_obs = obs
            # TODO: add next_skill 
            for i in range(n_envs):
                self._last_obs[i]["next_skill"] = np.zeros(len(self._skill_sequence))
                self._last_obs[i]["next_skill"][self._base_policies[i].skill_index] = 1

        # -------------------------------------------------------------------------- #
        # Reset and deal with truncated episodes
        # -------------------------------------------------------------------------- #
        next_value = None
        are_truncated = [False for _ in range(n_envs)]
        ignore_truncated = self._skill_config.RL.get("IGNORE_TRUNCATED", False)

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
                    self._last_obs,
                    device=self._device,
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
                next_value = self._goal_actor_critic.get_value(next_step_batch)

            for i_env in range(n_envs):
                if dones[i_env]:
                    obs[i_env] = self.envs.wait_reset_at(i_env)
        # -------------------------------------------------------------------------- #

        with self.timer.timeit("batch_obs"):
            batch = batch_obs(
                self._last_obs, device=self._device, cache=self._obs_batching_cache
            )
            stage_successes = np.zeros(n_envs)
            for i in range(n_envs):
                stage_success = infos["stage_success"]
                stage_successes[i] = np.sum([stage_success[stage] for stage in stage_success])
            diff_stage_successes = stage_successes - self._stage_successes
            rewards = torch.tensor(diff_stage_successes, dtype=torch.float).unsqueeze(1)
            done_masks = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
            not_done_masks = torch.logical_not(done_masks)
            truncated_masks = torch.tensor(
                are_truncated, dtype=torch.bool
            ).unsqueeze(1)
            self._stage_successes = stage_successes

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
    
    def train(self) -> None:
        ppo_cfg = self._skill_config.RL.PPO
        self._init_rollouts()

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

            # # Logging
            # episode_metrics = self.get_episode_metrics()
            # if self._num_updates_done % self._skill_config.LOG_INTERVAL == 0:
            #     self.log(episode_metrics)

            # # Tensorboard
            # metrics.update(**episode_metrics) 
            # metrics["lr"] = self._optimizer.param_groups[0]["lr"]

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()

        self._writer.close()
        self.envs.close()

    def update(self):
        """PPO update."""
        # import pdb; pdb.set_trace()
        ppo_cfg = self._skill_config.RL.PPO
        if ppo_cfg.use_linear_clip_decay:
            clip_param = ppo_cfg.clip_param * max(1 - self.percent_done(), 0.0)
        else:
            clip_param = ppo_cfg.clip_param
        ppo_epoch = ppo_cfg.ppo_epoch

        with torch.no_grad():
            step_batch = self.rollouts.buffers[self.rollouts.step_idx]
            next_value = self._goal_actor_critic.get_value(step_batch)

            # NOTE(jigu): next_value will be stored in the buffer.
            # However, it will be overwritten when next action is taken.
            self.rollouts.compute_returns(
                next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
            )
            advantages = self.rollouts.get_advantages(
                ppo_cfg.use_normalized_advantage
            )

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0

        num_updates = 0
        num_clipped_epoch = [0 for _ in range(ppo_epoch)]
        num_samples_epoch = [0 for _ in range(ppo_epoch)]

        for i_epoch in range(ppo_epoch):
            if ppo_cfg.use_recurrent_generator:
                data_generator = self.rollouts.recurrent_generator(
                    advantages, ppo_cfg.num_mini_batch
                )
            else:
                data_generator = self.rollouts.feed_forward_generator(
                    advantages, ppo_cfg.mini_batch_size
                )

            for batch in data_generator:
                outputs = self.actor_critic.evaluate_actions(
                    batch, batch["actions"]
                )
                values = outputs["value"]  # [B, 1]
                action_log_probs = outputs["action_log_probs"]  # [B, 1]
                dist_entropy = outputs["dist_entropy"]  # [B, A]

                ratio = torch.exp(action_log_probs - batch["action_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
                    * batch["advantages"]
                )
                action_loss = -torch.min(surr1, surr2)

                if ppo_cfg.use_clipped_value_loss:
                    value_pred_clipped = batch["value_preds"] + (
                        values - batch["value_preds"]
                    ).clamp(-clip_param, clip_param)
                    value_losses = (values - batch["returns"]).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - batch["returns"]
                    ).pow(2)
                    value_loss = 0.5 * torch.max(
                        value_losses, value_losses_clipped
                    )
                else:
                    value_loss = 0.5 * (batch["returns"] - values).pow(2)

                action_loss = action_loss.mean()
                value_loss = value_loss.mean()
                dist_entropy = dist_entropy.mean()

                self.optimizer.zero_grad()

                # ppo extra metrics
                num_clipped = torch.logical_or(
                    ratio < 1.0 - clip_param,
                    ratio > 1.0 + clip_param,
                ).float()
                num_clipped_epoch[i_epoch] += num_clipped.sum().item()
                num_samples_epoch[i_epoch] += num_clipped.size(0)

                total_loss = (
                    value_loss * ppo_cfg.value_loss_coef
                    + action_loss
                    - dist_entropy * ppo_cfg.entropy_coef
                )
                total_loss.backward()

                if ppo_cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(),
                        ppo_cfg.max_grad_norm,
                    )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                num_updates += 1

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        loss_dict = dict(
            value_loss=value_loss_epoch,
            action_loss=action_loss_epoch,
            dist_entropy=dist_entropy_epoch,
        )

        metric_dict = dict()
        for i_epoch in range(ppo_epoch):
            clip_ratio = num_clipped_epoch[i_epoch] / max(
                num_samples_epoch[i_epoch], 1
            )
            metric_dict[f"clip_ratio_{i_epoch}"] = clip_ratio

        self._num_updates_done += 1
        return loss_dict, metric_dict
