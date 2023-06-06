import argparse
import json
import os
import os.path as osp
import re

import magnum as mn
import numpy as np
import torch
from habitat import Config, logger
from habitat_baselines.utils.common import batch_obs, generate_video

from habitat_extensions.tasks.rearrange import RearrangeRLEnv
from habitat_extensions.tasks.rearrange.play import get_action_from_key
from habitat_extensions.utils.viewer import OpenCVViewer
from habitat_extensions.utils.visualizations.utils import put_info_on_image
from mobile_manipulation.config import get_config
from mobile_manipulation.methods.skill import CompositeSkill
from mobile_manipulation.utils.common import (
    extract_scalars_from_info,
    get_git_commit_id,
    get_run_name,
)
from mobile_manipulation.utils.wrappers import HabitatActionWrapperV1
from mobile_manipulation.goal_failure_belief.task_goal_critic.value_skill_sequence import TrainSkillValueNetwork

import pickle
from gym import spaces

from tqdm import tqdm
import random

OLD_SKILL_DISCOUNT_FACTOR = 0.9
SKILL_DISCOUNT_FACTOR = 0.995

def preprocess_config(config_path: str, config: Config):
    config.defrost()

    fileName = osp.splitext(osp.basename(config_path))[0]
    runName = get_run_name()
    substitutes = dict(fileName=fileName, runName=runName)

    config.PREFIX = config.PREFIX.format(**substitutes)
    config.BASE_RUN_DIR = config.BASE_RUN_DIR.format(**substitutes)

    for key in ["LOG_FILE", "VIDEO_DIR"]:
        config[key] = config[key].format(
            prefix=config.PREFIX, baseRunDir=config.BASE_RUN_DIR, **substitutes
        )


def update_ckpt_path(config: Config, seed: int):
    config.defrost()
    for k in config:
        if k == "CKPT_PATH":
            ckpt_path = config[k]
            new_ckpt_path = re.sub(r"seed=[0-9]+", f"seed={seed}", ckpt_path)
            print(f"Update {ckpt_path} to {new_ckpt_path}")
            config[k] = new_ckpt_path
        elif isinstance(config[k], Config):
            update_ckpt_path(config[k], seed)
    config.freeze()


def is_navigate(x):
    return x == "NavRLSkill"

def is_reset_arm(x):
    return x == "ResetArm"

def is_next_target(x):
    return x == "NextTarget"

def get_goal_position(env):
    if env.env._env._sim.gripper.is_grasped:
        return env.env._env.task.place_goal
    else:
        return env.env._env.task.pick_goal
    
def update_sensor_resolution(config: Config, height, width):
    config.defrost()
    sensor_names = [
        "THIRD_RGB_SENSOR",
        "RGB_SENSOR",
        "DEPTH_SENSOR",
        "SEMANTIC_SENSOR",
    ]
    for name in sensor_names:
        sensor_cfg = config.TASK_CONFIG.SIMULATOR[name]
        sensor_cfg.HEIGHT = height
        sensor_cfg.WIDTH = width
        print(f"Update {name} resolution")
    config.freeze()

def convert_next_skill_succeeds(next_skill_succeeds, num_post_nav_skills):
    length = len(next_skill_succeeds)
    next_skill_succeeds = np.array(next_skill_succeeds).reshape(length // num_post_nav_skills, num_post_nav_skills)
    future_discounted_returns = np.zeros_like(next_skill_succeeds, dtype=np.float32)
    future_discounted_returns[:, -1] = next_skill_succeeds[:, -1] * 1.0
    for i in range(num_post_nav_skills-2, -1, -1):
        future_discounted_returns[:, i] = next_skill_succeeds[:, i] * 1.0 + OLD_SKILL_DISCOUNT_FACTOR * future_discounted_returns[:, i+1]
    return future_discounted_returns

def compute_ground_truth_values(sub_traj_lengths, discounted_returns):
    """
    Assume discounted_returns has length batch_size
    """
    ground_truth = np.repeat(discounted_returns, sub_traj_lengths).reshape(-1, 1)
    return np.array(ground_truth, dtype=np.float32)

def compute_temporal_differences(sub_traj_lengths, rewards):
    """
    Assume rewards is 1d 
    """
    sub_traj_lengths = np.array(sub_traj_lengths)
    rewards = np.array(rewards, dtype=np.float32)
    total_traj_length = np.sum(sub_traj_lengths)
    td_discounted_returns = np.zeros(total_traj_length, dtype=np.float32)
    cum_sub_traj_lengths = np.cumsum(sub_traj_lengths) - 1
    td_discounted_returns[cum_sub_traj_lengths] = rewards
    for i in range(total_traj_length-2, -1, -1):
        td_discounted_returns[i] += SKILL_DISCOUNT_FACTOR * td_discounted_returns[i+1]
    return td_discounted_returns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="config_path", type=str, required=True)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    # Episodes
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="whether to shuffle test episodes",
    )
    parser.add_argument(
        "--num-episodes", type=int, help="number of episodes to evaluate"
    )
    parser.add_argument(
        "--episode-ids", type=str, help="episodes ids to evaluate"
    )

    # Save
    parser.add_argument("--save-video", choices=["all", "failure"])
    parser.add_argument("--save-log", action="store_true")

    # Viewer
    parser.add_argument(
        "--viewer", action="store_true", help="enable OpenCV viewer"
    )
    parser.add_argument("--viewer-delay", type=int, default=10)
    parser.add_argument(
        "--play", action="store_true", help="enable input control"
    )

    # Policy
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-seed", type=int)

    # Rendering
    parser.add_argument("--render-mode", type=str, default="human")
    parser.add_argument("--render-info", action="store_true")
    parser.add_argument(
        "--no-rgb", action="store_true", help="disable rgb observations"
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="use high resolution for visualization",
    )

    parser.add_argument(
        "--num_epochs_per_update",
        type=int,
        default=10
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------------------- #
    # Configure
    # ---------------------------------------------------------------------------- #
    config = get_config(args.config_path, opts=args.opts)
    preprocess_config(args.config_path, config)
    torch.set_num_threads(1)

    config.defrost()
    if args.split is not None:
        config.TASK_CONFIG.DATASET.SPLIT = args.split
    if not args.shuffle:
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = False
    if args.no_rgb:
        sensors = config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = [
            x for x in sensors if "RGB" not in x
        ]
    config.freeze()

    if args.train_seed is not None:
        update_ckpt_path(config, seed=args.train_seed)

    if args.high_res:
        update_sensor_resolution(config, height=720, width=1080)

    if args.save_log:
        if config.LOG_FILE:
            log_dir = os.path.dirname(config.LOG_FILE)
            os.makedirs(log_dir, exist_ok=True)
            logger.add_filehandler(config.LOG_FILE)
        logger.info(config)
        logger.info("commit id: {}".format(get_git_commit_id()))

    # For reproducibility, just skip other episodes
    if args.episode_ids is not None:
        eval_episode_ids = eval(args.episode_ids)
        eval_episode_ids = [str(x) for x in eval_episode_ids]
    else:
        eval_episode_ids = None

    # ---------------------------------------------------------------------------- #
    # Initialize env
    # ---------------------------------------------------------------------------- #
    env = RearrangeRLEnv(config)
    env = HabitatActionWrapperV1(env)
    env.seed(config.TASK_CONFIG.SEED)
    print("obs space", env.observation_space)
    print("action space", env.action_space)

    # -------------------------------------------------------------------------- #
    # Initialize policy
    # -------------------------------------------------------------------------- #
    policy = CompositeSkill(config.SOLUTION, env)
    policy.to(args.device)

    # -------------------------------------------------------------------------- #
    # Main
    # -------------------------------------------------------------------------- #
    # num_episodes = len(env.habitat_env.episode_iterator.episodes)

    skill_sequence = policy.skill_sequence
    num_post_nav_skills = 0
    for i in range(len(skill_sequence)):
        if skill_sequence[i] == "NavRLSkill":
            num_post_nav_skills += 1

    skill_ordering = []
    for s in policy.skills:
        if (s not in skill_ordering) and (not is_navigate(s)) and (not is_reset_arm(s)):
            skill_ordering.append(s)
    skill_ordering_dict = {s: np.zeros(len(skill_ordering)) for s in skill_ordering}

    for i, s in enumerate(skill_ordering):
        skill_ordering_dict[s][i] = 1

    # Recompute obs space with new next_skill observation
    new_obs_space_dict = env.observation_space.spaces.copy()
    new_obs_space_dict["next_skill"] = spaces.Box(0, 1, (num_post_nav_skills,), np.int32)
    new_obs_space_dict["goal"] = spaces.Box(-np.inf, np.inf, (3,), np.float32)
    new_observation_space = spaces.Dict(new_obs_space_dict)

    reload = True
    skill_value_network = TrainSkillValueNetwork(observation_space=new_observation_space, action_space=env.action_space, prefix=config.TASK_CONFIG.TASK.TYPE, device=args.device, reload=reload)
    num_epochs = 1000
    base_file_path = "mobile_manipulation/goal_failure_belief/belief_train_data/"
    proc_file_path = "mobile_manipulation/goal_failure_belief/belief_train_proc_data/"
    saved_traj_dir = sorted(os.listdir(proc_file_path))
    import time
    num_val = 3
    
    for epoch in tqdm(range(num_epochs)):
        sampled_list = random.sample(saved_traj_dir[:-1*num_val], len(saved_traj_dir)-num_val)
        # sampled_list = saved_traj_dir
        for i in range(len(sampled_list)):
            fname = sampled_list[i]
            with open(os.path.join(proc_file_path, fname), 'rb') as f:
                proc_saved_data = pickle.load(f)
            skill_value_network.train_model(proc_saved_data['ob_trajectories'], proc_saved_data['ground_truths'])
        proc_saved_data = {"ob_trajectories": [], "ground_truths": []}
        for fname in saved_traj_dir[-1*num_val:]:
            with open(os.path.join(proc_file_path, fname), "rb") as f:
                loaded_proc_saved_data = pickle.load(f)
                proc_saved_data['ob_trajectories'] += loaded_proc_saved_data['ob_trajectories']
                proc_saved_data['ground_truths'] += loaded_proc_saved_data['ground_truths']
        skill_value_network.save_train_info(proc_saved_data['ob_trajectories'], proc_saved_data['ground_truths'])
    env.close()

if __name__ == "__main__":
    main()
