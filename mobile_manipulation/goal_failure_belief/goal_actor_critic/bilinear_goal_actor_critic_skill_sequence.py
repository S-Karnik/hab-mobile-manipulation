import multiprocessing
from ..task_goal_critic.task_success_value_network import (
    BilinearGoalPredictorPolicy, 
    load_goal_critic_model_only,
    load_goal_actor_critic_model,
) 
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    batch_obs,
    generate_video,
    get_checkpoint_id,
)
from mobile_manipulation.utils.common import (
    Timer,
    extract_scalars_from_info,
    get_git_commit_id,
    get_latest_checkpoint,
)
import asyncio
import concurrent.futures
from collections import deque
from copy import deepcopy
import numpy as np
import os
import random
import time
from typing import Dict
from collections import OrderedDict

from gym import spaces
from habitat import Config, RLEnv, logger
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
from mobile_manipulation.common.rollout_storage import BilinearRolloutStorage
from mobile_manipulation.utils.env_utils import (
    ThreadedVectorEnv,
    VectorEnv,
    construct_envs_multi_config,
    make_env_fn,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from mobile_manipulation.utils.wrappers import HabitatActionWrapperV1
from mobile_manipulation.methods.vec_bilinear_skill import CompositeSkill
from mobile_manipulation.common.registry import (
    vbmm_registry as my_registry,
)

TASK_SEQ_LENGTH = 8
GOAL_SIZE = 3
DENSE_TASK_REWARD = 5.0
FINAL_TASK_REWARD = 10.0

import concurrent.futures

class ReusableThreadPoolExecutor:
    def __init__(self, max_workers):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
    def submit(self, fn, *args, **kwargs):
        return self.executor.submit(fn, *args, **kwargs)

    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        self.executor.shutdown(wait=True)

class TrainBilinearGoalActorCritic:

    envs: VectorEnv
    device: torch.device
    optimizer: torch.optim.Adam

    obs_space: spaces.Space
    _obs_batching_cache: ObservationBatchingCache
    action_space: spaces.Space
    checkpoint_dir: str
    goal_dist_coef: np.float32
    gpu_id: int
    
    def __init__(self, composite_task_config, skill_config, seed=0, goal_dist_coef=0.1, gpu_id=0) -> None:
        self._composite_task_config = composite_task_config
        self._skill_config = skill_config
        self._skill_sequence = composite_task_config.SOLUTION.SKILL_SEQUENCE
        self._base_policies = []

        self._prefix = composite_task_config.TASK_CONFIG.TASK.TYPE
        self.timer = Timer()  # record fine-grained scopes
        current_file_path = os.path.abspath(__file__)
        parent_dir = os.path.abspath(os.path.join(current_file_path, os.pardir))
        self.checkpoint_dir = os.path.join(parent_dir, f'{self._prefix}-models-{seed}-{composite_task_config.FULL_CKPT_NAME}-early-terminate-fixed-thresh-2-init-correct-smooth-2pt5reward')
        self.tensorboard_dir = os.path.join(parent_dir, f'{self._prefix}-tensorboard-{seed}-{composite_task_config.FULL_CKPT_NAME}-early-terminate-fixed-thresh-2-init-correct-smooth-2pt5reward')
        self.checkpoint_dir = os.path.join(parent_dir, f'{self._prefix}-models-{seed}-debug')
        self.tensorboard_dir = os.path.join(parent_dir, f'{self._prefix}-tensorboard-{seed}-debug')
        self.seed = int(seed)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.exists(self.tensorboard_dir):
            os.mkdir(self.tensorboard_dir)
        self.goal_dist_coef = goal_dist_coef
        self.gpu_id = gpu_id
        print("initialized trainer")

    def to_device(self, tensor_dict, device: torch.device):
        """Send a TensorDict to a given device."""
        if isinstance(tensor_dict, torch.Tensor):
            return tensor_dict.to(device)
        return {key: tensor.to(device) for key, tensor in tensor_dict.items()}

    def summarize(self, losses: Dict[str, float], metrics: Dict[str, float]):
        """Summarize scalars in tensorboard."""
        for k, v in losses.items():
            self.writer.add_scalar(f"losses/{k}", v, self.num_steps_done)
        for k, v in metrics.items():
            self.writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)

    def summarize2(self):
        """Summarize histogram and video in tensorboard."""
        self.writer.add_histogram(
            "value_preds",
            self.rollouts.buffers["value_preds"],
            global_step=self.num_steps_done,
        )
        self.writer.add_histogram(
            "discounted_returns",
            self.rollouts.buffers["returns"],
            global_step=self.num_steps_done,
        )

        video_keys = self._skill_config.get("TB_VIDEO_KEYS", [])
        for key in video_keys:
            video_tensor = self.rollouts.buffers["observations"][key]
            self.writer.add_video(
                key,
                video_tensor.permute(1, 0, 4, 2, 3),
                global_step=self.num_steps_done,
                fps=10,
            )

    def should_summarize(self, mult=1) -> bool:
        if self._skill_config.SUMMARIZE_INTERVAL == -1:
            interval = self._skill_config.LOG_INTERVAL
        else:
            interval = self._skill_config.SUMMARIZE_INTERVAL
        return self.num_updates_done % (interval * mult) == 0

    def should_checkpoint(self) -> bool:
        if self._skill_config.NUM_CHECKPOINTS == -1:
            ckpt_freq = self._skill_config.CHECKPOINT_INTERVAL
        else:
            ckpt_freq = (
                self._skill_config.TOTAL_NUM_STEPS // self._skill_config.NUM_CHECKPOINTS
            )
        return self.num_steps_done >= (self.count_checkpoints + 1) * 1000000

    def should_checkpoint2(self) -> bool:
        """Check whether to save (overwrite) the latest checkpoint."""
        if (
            self._skill_config.NUM_CHECKPOINTS == -1
            or self._skill_config.CHECKPOINT_INTERVAL == -1
        ):
            return False
        return self.num_updates_done % (self._skill_config.LOG_INTERVAL * 6) == 0

    def save_checkpoint(self, ckpt_path):
        wall_time = (time.time() - self.t_start) + self.prev_time
        checkpoint = dict(
            config=self._skill_config,
            state_dict=self._goal_actor_critic.state_dict(),
            optim_state=self.optimizer.state_dict(),
            step=self.num_steps_done,
            wall_time=wall_time,
            num_updates_done=self.num_updates_done,
            count_checkpoints=self.count_checkpoints,
        )
        torch.save(checkpoint, ckpt_path)

    def save(self, ckpt_id):
        if not self.checkpoint_dir:
            return
        ckpt_path = os.path.join(
            self.checkpoint_dir, f"ckpt.{ckpt_id}.pth"
        )
        self.save_checkpoint(ckpt_path)
        logger.info(
            f"Saved checkpoint to {ckpt_path} at {self.num_steps_done}th step"
        )

    def resume(self):
        if not self.checkpoint_dir:
            return
        ckpt_path = get_latest_checkpoint(self.checkpoint_dir, False)
        if ckpt_path is None:
            return
        assert os.path.isfile(ckpt_path), ckpt_path
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        logger.info(f"Resume from {ckpt_path}")

        self._goal_actor_critic.load_state_dict(ckpt_dict["state_dict"])
        self.optimizer.load_state_dict(ckpt_dict["optim_state"])

        self.num_steps_done = ckpt_dict["step"]
        self.num_updates_done = ckpt_dict["num_updates_done"]
        self.prev_time = ckpt_dict["wall_time"]
        self.count_checkpoints = ckpt_dict["count_checkpoints"]
        self.prev_ckpt_step = self.num_steps_done

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
        self._saved_actor_critics = {}
        if "default" in self._composite_task_config.FULL_CKPT_NAME:
            from mobile_manipulation.methods.vec_skill import CompositeSkill
        else:
            from mobile_manipulation.methods.vec_bilinear_skill import CompositeSkill
        self._base_policies = [CompositeSkill(self._composite_task_config.SOLUTION, self.envs, i, self._saved_actor_critics) for i in range(self.envs.num_envs)]
        for i in range(self.envs.num_envs):
            self._base_policies[i].to(self.device)
            
        self.thread_executor = ReusableThreadPoolExecutor(self.envs.num_envs)
        
        # self._base_policies = nn.DataParallel(self._base_policies)

    def _init_observation_space(self):
        if isinstance(self.envs, VectorEnv):
            obs_space = self.envs.observation_spaces[0]
        else:
            env: RLEnv = self.envs[0]
            obs_space = env.observation_space
        original_sensors = list(obs_space.spaces)
        for sensor in original_sensors:
            if "rgb" in sensor:
                del obs_space.spaces[sensor]
        self.obs_space = obs_space
        self.obs_space["next_skill"] = spaces.Box(0, 1, (TASK_SEQ_LENGTH,), np.int32)

    def _init_action_space(self):
        self.action_space = spaces.Box(-np.inf, np.inf, (3,), np.float32)

    def _setup_goal_actor_critic(self, config: Config) -> None:
        r"""Set up actor critic for PPO."""
        self._goal_actor_critic = BilinearGoalPredictorPolicy.setup_policy(
            observation_space=self.obs_space, 
            action_space=self.action_space
        )
        self._goal_actor_critic.to(self.device)

        ppo_cfg = config.RL.PPO
        self.optimizer = torch.optim.Adam(
            self._goal_actor_critic.parameters(), lr=ppo_cfg.lr, eps=ppo_cfg.eps
        )

    def _setup_rollouts(self, skill_config: Config):
        ppo_cfg = skill_config.RL.PPO
        self.rollouts = BilinearRolloutStorage(
            ppo_cfg.num_steps,  # number of steps for each env
            self.envs.num_envs,
            observation_space=self.obs_space,
            action_space=self.action_space,
            recurrent_hidden_state_size=self._goal_actor_critic.net.rnn_hidden_size,
            num_recurrent_layers=self._goal_actor_critic.net.num_recurrent_layers,
        )
        self.rollouts.to(self.device)

    def is_navigate(self, x):
        return x == "NavRLSkill"

    def is_reset_arm(self, x):
        return x == "ResetArm"

    def is_next_target(self, x):
        return x == "NextTarget"

    def _init_train(self):
        if self._composite_task_config.LOG_FILE:
            log_dir = os.path.dirname(self._composite_task_config.LOG_FILE)
            os.makedirs(log_dir, exist_ok=True)
            logger.add_filehandler(self._composite_task_config.LOG_FILE)

        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # ---------------------------------------------------------------------------- #
        # Initialization
        # ---------------------------------------------------------------------------- #
        if torch.cuda.is_available():
            print(torch.cuda.device_count())
            self.device = torch.device("cuda", self.gpu_id)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # ---------------------------------------------------------------------------- #
        # NOTE(jigu): workaround from erik, to avoid high gpu memory fragmentation
        env = make_env_fn(
            self._skill_config,
            get_env_class(self._skill_config.ENV_NAME),
            wrappers=[HabitatActionWrapperV1],
        )
        self.envs = [env]
        # import pdb; pdb.set_trace()
        self._init_observation_space()
        self._init_action_space()
        self._setup_goal_actor_critic(self._skill_config)
        env.close()
        # ---------------------------------------------------------------------------- #

        self._init_envs_and_policies()
        self._init_observation_space()
        self._init_action_space()
        self._setup_rollouts(self._skill_config)

        if self._composite_task_config.VERBOSE:
            logger.info(f"actor_critic: {self._goal_actor_critic}")
        logger.info(
            "#parameters: {}".format(
                sum(param.numel() for param in self._goal_actor_critic.parameters())
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
            maxlen=self._skill_config.RL.PPO.reward_window_size
        )

        self.t_start = time.time()  # record overall time
        self.timer = Timer()  # record fine-grained scopes
        self.writer = TensorboardWriter(
            self.tensorboard_dir, flush_secs=30
        )

        # resumable stats
        self.num_steps_done = 0
        self.num_updates_done = 0
        self.prev_time = 0.0
        self.count_checkpoints = 0
        self.prev_ckpt_step = 0

        config_stages = list(self._composite_task_config.TASK_CONFIG.TASK.StageSuccess.GOALS.keys())
        config_stage_aliases = [s for s in self._base_policies[0].skill_sequence if (not self.is_navigate(s)) and (not self.is_reset_arm(s)) and (not self.is_next_target(s))]
        config_stage_alias_dict = {config_stage_aliases[i]: [] for i in range(len(config_stage_aliases))}
        for i in range(len(config_stage_aliases)):
            config_stage_alias_dict[config_stage_aliases[i]].append(config_stages[i])
        self._config_stage_alias_dict = config_stage_alias_dict
        self._positions_in_skill_seq = []
        j = 0
        for i, skill_name in enumerate(self._skill_sequence):
            self._positions_in_skill_seq.append(min(j, TASK_SEQ_LENGTH-1))
            if skill_name in self._config_stage_alias_dict:
                j += 1

    def _init_rollouts(self):
        self._obs_batching_cache = ObservationBatchingCache()
        observations = self.envs.reset()
        self._last_obs = observations
        for i in range(self.envs.num_envs):
            sensors = list(observations[i].keys())
            for key in sensors:
                if "rgb" in key:
                    del observations[i][key]
            self._base_policies[i].reset(observations[i])
            # import pdb; pdb.set_trace()
            observations[i]["next_skill"] = np.zeros(TASK_SEQ_LENGTH)
            observations[i]["next_skill"][self._positions_in_skill_seq[self._base_policies[i]._skill_idx]] = 1
        self._stage_successes = np.zeros(self.envs.num_envs)
        self._returns = np.zeros(self.envs.num_envs)
        self._entropies = np.zeros(self.envs.num_envs)
        self._max_entropies = 0
        self._sum_final_entropies = 0
        self._num_steps_per_env = np.zeros(self.envs.num_envs)
        self._num_completed_envs = 0
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        self.rollouts.buffers["observations"][0] = batch
    
    def eval(self):
        self.device = (
            torch.device("cuda", self._composite_task_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        if "tensorboard" in self._composite_task_config.VIDEO_OPTION:
            assert (
                len(self._composite_task_config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self._composite_task_config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self._composite_task_config.VIDEO_OPTION:
            assert (
                len(self._composite_task_config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"
            os.makedirs(self._composite_task_config.VIDEO_DIR, exist_ok=True)

        if self._composite_task_config.LOG_FILE:
            log_dir = os.path.dirname(self._composite_task_config.LOG_FILE)
            os.makedirs(log_dir, exist_ok=True)
            logger.add_filehandler(self._composite_task_config.LOG_FILE)

        writer = TensorboardWriter(self._composite_task_config.TENSORBOARD_DIR, flush_secs=30)

        if self._skill_config.EVAL.CKPT_PATH:
            ckpt_path = self._skill_config.EVAL.CKPT_PATH
        else:
            ckpt_path = get_latest_checkpoint(
                self.checkpoint_dir, True
            )
        assert os.path.isfile(ckpt_path), ckpt_path
        ckpt_id = get_checkpoint_id(ckpt_path)
        if ckpt_id is None:
            ckpt_id = -1

        # if self._skill_config.EVAL.BATCH_ENVS:
        #     self._batch_eval_checkpoint(ckpt_path, writer, ckpt_id)
        # else:
        #     self._eval_checkpoint(ckpt_path, writer, ckpt_id)
        writer.close()

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self._goal_actor_critic.parameters(), self._max_grad_norm
        )

    def is_done(self):
        return self.num_steps_done >= self._skill_config.TOTAL_NUM_STEPS

    def percent_done(self):
        return self.num_steps_done / self._skill_config.TOTAL_NUM_STEPS
    
    def get_episode_metrics(self):
        if len(self.window_episode_stats) == 0:
            return {}
        # Assume all episodes have the same keys. True for Habitat.
        return {
            k: np.mean([ep_info[k] for ep_info in self.window_episode_stats])
            for k in self.window_episode_stats[0].keys()
        }

    @classmethod
    def _extract_scalars_from_info(cls, info: Dict):
        return extract_scalars_from_info(
            info, blacklist=["terminal_observation"]
        )

    def log(self, metrics: Dict[str, float]):
        wall_time = (time.time() - self.t_start) + self.prev_time
        logger.info(
            "update: {}\tframes: {}\tfps: {:.3f}\tpercent: {:.2f}%".format(
                self.num_updates_done,
                self.num_steps_done,
                self.num_steps_done / wall_time,
                self.percent_done() * 100,
            )
        )
        logger.info(
            "\t".join(
                "{}: {:.3f}s".format(k, v)
                for k, v in self.timer.elapsed_times.items()
            )
        )
        logger.info(
            "  ".join("{}: {:.3f}".format(k, v) for k, v in metrics.items()),
        )

    def summarize(self, losses: Dict[str, float], metrics: Dict[str, float]):
        """Summarize scalars in tensorboard."""
        for k, v in losses.items():
            self.writer.add_scalar(f"losses/{k}", v, self.num_steps_done)
        for k, v in metrics.items():
            self.writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)

    def summarize2(self):
        """Summarize histogram and video in tensorboard."""
        self.writer.add_histogram(
            "value_preds",
            self.rollouts.buffers["value_preds"],
            global_step=self.num_steps_done,
        )
        self.writer.add_histogram(
            "discounted_returns",
            self.rollouts.buffers["returns"],
            global_step=self.num_steps_done,
        )

        video_keys = self._composite_task_config.get("TB_VIDEO_KEYS", [])
        for key in video_keys:
            video_tensor = self.rollouts.buffers["observations"][key]
            self.writer.add_video(
                key,
                video_tensor.permute(1, 0, 4, 2, 3),
                global_step=self.num_steps_done,
                fps=10,
            )

    # def modify_env_goals(self, goal_residuals):
    #     for i in range(self.envs.num_envs):
    #         self._last_obs[i] = self.envs.call_at(i, "set_goal", {"goal_residual": goal_residuals[i].cpu().numpy(), "current_obs": self._last_obs[i]})
    
    def act_in_parallel_multiprocessing(self, i_env, base_policy, last_obs):
        out = base_policy.act(last_obs[i_env])
        entropy = 0 if "entropy" not in out else out["entropy"].item()
        return (i_env, out, entropy)
    
    async def act_in_parallel_coroutine(self, base_policy, obs):
        return base_policy.act(obs)

    async def parallelized_policy_act_asyncio(self):
        # Call act_in_parallel_coroutine concurrently for all environments
        coroutines = [self.act_in_parallel_coroutine(self._base_policies[i_env], self._last_obs[i_env]) for i_env in range(self.envs.num_envs)]
        results = await asyncio.gather(*coroutines)

        # Extract actions and entropies from results
        actions = []
        entropies = []
        for i_env, result in enumerate(results):
            out = result
            entropy = 0 if "entropy" not in out else out["entropy"].item()
            actions.append((i_env, out))
            entropies.append(entropy)

        actions.sort()
        _, actions_only = zip(*actions)
        return actions_only, entropies

    # Parallelize base_policy.act() with one process per environment index
    def parallelized_policy_act_multiprocessing(self):
        actions = []
        n_envs = self.envs.num_envs
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_envs) as executor:
            futures = [executor.submit(self.act_in_parallel_multiprocessing, i_env, self._base_policies[i_env], self._last_obs) for i_env in range(n_envs)]
            for future in concurrent.futures.as_completed(futures):
                action = future.result()
                actions.append(action)
        actions.sort()
        _, actions_only, entropies = zip(*actions)
        return actions_only, entropies

    def modify_env_goals(self, goal_residuals):
        goal_residuals = goal_residuals.cpu().numpy()        
        self._last_obs = self.envs.call(["set_goal"] * self.envs.num_envs, [{"goal_residual": goal_residuals[i], "current_obs": self._last_obs[i]} for i in range(self.envs.num_envs)])

    def act_in_parallel(self, i_env):
        out = self._base_policies[i_env].act(self._last_obs[i_env])
        return (i_env, out)

    # Parallelize base_policy.act() with one thread per environment index
    def parallelized_policy_act(self):
        n_envs = self.envs.num_envs
        futures = [self.thread_executor.submit(self.act_in_parallel, i_env) for i_env in range(n_envs)]
        results = [f.result() for f in futures]
        results.sort()
        _, actions_only = zip(*results)
        return actions_only

    @torch.no_grad()
    def step(self):
        """Take one rollout step."""
        n_envs = self.envs.num_envs
        
        ### Goal Sampling ###
        with self.timer.timeit("sample_goal"):
            step_batch = self.rollouts.buffers[self.rollouts.step_idx]
            output_batch = self._goal_actor_critic.act(step_batch)
            # self.modify_env_goals(output_batch["action"])
            # print(self._last_obs)
        
        with self.timer.timeit("update_rollout"):
            self.rollouts.insert(
                next_goal_dep_recurrent_hidden_states=output_batch.get(
                    "goal_dep_rnn_hidden_states"
                ),
                next_goal_ind_recurrent_hidden_states=output_batch.get(
                    "goal_ind_rnn_hidden_states"
                ),
                actions=output_batch["action"],
                action_log_probs=output_batch["action_log_probs"],
                value_preds=output_batch["value"],
            )
        
        with self.timer.timeit("sample_action"):
            # Assume that observations are stored at rollouts in the last step
            actions = self.parallelized_policy_act()

        with self.timer.timeit("step_env"):
            for i_env, action in zip(range(n_envs), actions):
                self.envs.async_step_at(i_env, action)
            # import pdb; pdb.set_trace()
            results = self.envs.wait_step()
            obs, rews, dones, infos = map(list, zip(*results))
            # import pdb; pdb.set_trace()
            self.num_steps_done += n_envs
        with self.timer.timeit("update_info"):
            self._last_obs = obs
            stage_successes = np.zeros(n_envs)
            for i_env in range(n_envs):
                stage_success = infos[i_env]["stage_success"]
                stage_successes[i_env] = np.sum([stage_success[stage] for stage in stage_success])
                current_skill_name = self._base_policies[i_env].skill_sequence[self._base_policies[i_env]._skill_idx]
                if not (self.is_navigate(current_skill_name) or self.is_reset_arm(current_skill_name) or self.is_next_target(current_skill_name)):
                    self.envs.call(["set_early_terminate"] * self.envs.num_envs, [{"early_terminate": True} for i in range(self.envs.num_envs)])
                    if "composite_reward" in infos[i_env]:
                        rews[i_env] = infos[i_env]['composite_reward']
                else:
                    self.envs.call(["set_early_terminate"] * self.envs.num_envs, [{"early_terminate": False} for i in range(self.envs.num_envs)])
                    rews[i_env] = 0
            diff_stage_successes = stage_successes - self._stage_successes
            self._stage_successes = stage_successes
        
        with self.timer.timeit("update_current_skill"):
            updated_obs = obs
            for i_env in range(n_envs):
                updated_obs[i_env]["next_skill"] = np.zeros(TASK_SEQ_LENGTH)
                updated_obs[i_env]["next_skill"][self._positions_in_skill_seq[self._base_policies[i_env]._skill_idx]] = 1
                reset_env = self._base_policies[i_env].check_if_done(obs[i_env]) and (self._base_policies[i_env]._skill_idx == len(self._skill_sequence) - 1)
                if self._base_policies[i_env]._skill_idx > 0:
                    prev_skill_name = self._base_policies[i_env].skill_sequence[self._base_policies[i_env]._skill_idx-1]
                    if (prev_skill_name == "NextTarget") and (self._stage_successes[i_env] < 4):
                        reset_env = True
                if reset_env:
                    dones[i_env] = True
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
                    updated_obs,
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
                next_value = self._goal_actor_critic.get_value(next_step_batch)

            self._num_steps_per_env += 1

            for i_env in range(n_envs):
                if dones[i_env]:
                    ob = self.envs.wait_reset_at(i_env)
                    self._last_obs[i_env] = ob
                    self._base_policies[i_env].reset(ob)
                    updated_obs[i_env] = ob
                    updated_obs[i_env]["next_skill"] = np.zeros(TASK_SEQ_LENGTH)
                    updated_obs[i_env]["next_skill"][self._positions_in_skill_seq[self._base_policies[i_env]._skill_idx]] = 1
                    self._num_completed_envs += 1
                    self._sum_final_entropies += self._entropies[i_env] / self._num_steps_per_env[i_env]
                    self._num_steps_per_env[i_env] = 0
                    self._entropies[i_env] = 0
                    if (self._stage_successes[i_env] == len(infos[i_env]["stage_success"])):
                        rews[i_env] += FINAL_TASK_REWARD

        # -------------------------------------------------------------------------- #
        self._returns += rews
        if any(dones):
            for i_env in range(n_envs):
                if dones[i_env]:
                    # import pdb; pdb.set_trace()
                    self._returns[i_env] = 0
                    self._stage_successes[i_env] = 0

        with self.timer.timeit("batch_obs"):
            batch = batch_obs(
                updated_obs, device=self.device, cache=self._obs_batching_cache
            )
            rewards = torch.tensor(rews, dtype=torch.float).unsqueeze(1)
            done_masks = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
            not_done_masks = torch.logical_not(done_masks)
            truncated_masks = torch.tensor(
                are_truncated, dtype=torch.bool
            ).unsqueeze(1)

        with self.timer.timeit("update_stats"):
            for i_env in range(n_envs):
                self.episode_rewards[i_env] += diff_stage_successes[i_env]
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

        self._init_train()
        self._init_rollouts()
        # self.resume()

        if ppo_cfg.use_linear_lr_decay:
            min_lr = ppo_cfg.get("min_lr", 0.0)
            min_lr_ratio = min_lr / ppo_cfg.lr
            lr_lambda = lambda x: max(
                1 - self.percent_done() * (1.0 - min_lr_ratio), min_lr_ratio
            )
            lr_scheduler = LambdaLR(
                optimizer=self.optimizer, lr_lambda=lr_lambda
            )

        while not self.is_done():
            # Rollout and collect transitions
            self._goal_actor_critic.eval()
            for i in range(ppo_cfg.num_steps):
                self.step()
            # import pdb; pdb.set_trace()

            with self.timer.timeit("update_model"):
                self._goal_actor_critic.train()
                losses, metrics = self.update()
                self.rollouts.after_update()
            
            # Logging
            episode_metrics = self.get_episode_metrics()
            if self.num_updates_done % self._skill_config.LOG_INTERVAL == 0:
                self.log(episode_metrics)
            # Tensorboard
            metrics.update(**episode_metrics)
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            if self.should_summarize():
                self.summarize(losses, metrics)
            if self.should_summarize(10):
                self.summarize2()

            # Checkpoint
            if self.should_checkpoint():
                self.count_checkpoints += 1
                self.prev_ckpt_step = self.num_steps_done
                self.save(ckpt_id=self.count_checkpoints)
            if self.should_checkpoint2():
                self.save(ckpt_id=-1)

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()

        # Save the last model
        if self.num_steps_done > self.prev_ckpt_step:
            self.count_checkpoints += 1
            self.save(ckpt_id=self.count_checkpoints)

        self.writer.close()
        self.envs.close()

    def update(self):
        """PPO update."""
        # self.rollouts.to("cpu")
        ppo_cfg = self._skill_config.RL.PPO
        if ppo_cfg.use_linear_clip_decay:
            clip_param = ppo_cfg.clip_param * max(1 - self.percent_done(), 0.0)
        else:
            clip_param = ppo_cfg.clip_param
        ppo_epoch = ppo_cfg.ppo_epoch

        with torch.no_grad():
            step_batch = self.rollouts.buffers[self.rollouts.step_idx]
            # for key in step_batch:
            #     step_batch[key] = self.to_device(step_batch[key], self.device)
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
        pick_loss_epoch = 0.0
        place_loss_epoch = 0.0
        pick_loss_smooth_epoch = 0.0
        place_loss_smooth_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0

        num_updates = 0
        num_clipped_epoch = [0 for _ in range(ppo_epoch)]
        num_samples_epoch = [0 for _ in range(ppo_epoch)]

        for i_epoch in range(ppo_epoch):
            if ppo_cfg.use_recurrent_generator:
                data_generator = self.rollouts.recurrent_generator(
                    advantages, 2
                )
            else:
                data_generator = self.rollouts.feed_forward_generator(
                    advantages, ppo_cfg.mini_batch_size
                )

            for batch in data_generator:
                # for key in batch:
                #     batch[key] = self.to_device(batch[key], self.device)
                outputs = self._goal_actor_critic.evaluate_actions(
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
                # import pdb; pdb.set_trace()
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
                grasped_obs = batch["grasped"].detach()
                prev_grasped_obs = batch["prev_grasped"].detach()
                not_grasped_obs = torch.logical_not(grasped_obs)
                prev_not_grasped_obs = torch.logical_not(prev_grasped_obs)
                consecutive_grasped_obs = torch.logical_and(grasped_obs, prev_grasped_obs)
                consecutive_not_grasped_obs = torch.logical_and(not_grasped_obs, prev_not_grasped_obs)
                total_num_observations = len(grasped_obs)
                num_grasped = torch.sum(grasped_obs)
                num_not_grasped = total_num_observations - num_grasped
                pick_goal_dist_loss = (not_grasped_obs * (batch["actions"]).pow(2)).sum() / (num_not_grasped + 1e-10)
                place_goal_dist_loss = (grasped_obs * (batch["actions"])).pow(2).sum() / (num_grasped + 1e-10)
                pick_goal_smoothness = (consecutive_not_grasped_obs * (batch["actions"] - batch["prev_actions"]).pow(2)).sum() / (num_not_grasped + 1e-10)
                place_goal_smoothness = (consecutive_grasped_obs * (batch["actions"] - batch["prev_actions"]).pow(2)).sum() / (num_grasped + 1e-10)
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
                    # + 0.001 * pick_goal_dist_loss
                    # + 0.001 * place_goal_dist_loss
                    + 0.001 * pick_goal_smoothness
                    + 0.001 * place_goal_smoothness
                    # - dist_entropy * 0.001
                )
                total_loss.backward()

                if ppo_cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self._goal_actor_critic.parameters(),
                        ppo_cfg.max_grad_norm,
                    )
                self.optimizer.step()

                pick_loss_epoch += pick_goal_dist_loss.item()
                place_loss_epoch += place_goal_dist_loss.item()
                pick_loss_smooth_epoch += pick_goal_smoothness.item()
                place_loss_smooth_epoch += place_goal_smoothness.item()
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                num_updates += 1
                del batch

        pick_loss_epoch /= num_updates
        place_loss_epoch /= num_updates
        pick_loss_smooth_epoch /= num_updates
        place_loss_smooth_epoch /= num_updates
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        

        loss_dict = dict(
            pick_loss=pick_loss_epoch,
            place_loss=place_loss_epoch,
            pick_loss_smooth=pick_loss_smooth_epoch,
            place_loss_smooth=place_loss_smooth_epoch,
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

        self.num_updates_done += 1
        # self.rollouts.to(self.device)
        return loss_dict, metric_dict
