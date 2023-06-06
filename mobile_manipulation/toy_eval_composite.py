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
from tqdm import tqdm

import mobile_manipulation.methods.skills
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
from mobile_manipulation.goal_failure_belief.train_failure_belief_init import TrainBeliefClassifier
from mobile_manipulation.goal_failure_belief.energy_goal_sampler import EnergyGoalSampler
from gym import spaces
import pickle

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

def set_goal_position(env, new_goal) -> None:
    if env.env._env._sim.gripper.is_grasped:
        env.env._env.task.place_goal = new_goal
    else:
        env.env._env.task.pick_goal = new_goal


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
    parser.add_argument("--split", type=str, default="val")
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

    eval_episode_ids = ["3"]
    for epoch in tqdm(range(100)):
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
        num_episodes = env.number_of_episodes
        # num_episodes = len(env.habitat_env.episode_iterator.episodes)
        if args.num_episodes is not None:
            num_episodes = args.num_episodes

        done, info = True, {}
        all_episode_stats = []
        episode_reward = 0
        failure_episodes = []

        if args.save_video is not None:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)
        rgb_frames = []

        if args.viewer:
            viewer = OpenCVViewer(config.TASK_CONFIG.TASK.TYPE)

        
        num_epoch_trajs = 50
        failure_thresh = 0.9

        if epoch == 0:
            relevant_ob_keys = ["robot_head_depth", "nav_goal_at_base"]
            skill_ordering = []
            for s in policy.skills:
                if (s not in skill_ordering) and (not is_navigate(s)) and (not is_reset_arm(s)):
                    skill_ordering.append(s)
            skill_ordering_dict = {s: np.zeros(len(skill_ordering)) for s in skill_ordering}

            for i, s in enumerate(skill_ordering):
                skill_ordering_dict[s][i] = 1

            # Recompute obs space with new next_skill observation
            new_obs_space_dict = env.observation_space.spaces.copy()
            new_obs_space_dict["next_skill"] = spaces.Box(0, 1, (len(skill_ordering),), np.int32)
            new_obs_space_dict["goal"] = spaces.Box(-np.inf, np.inf, (3,), np.int32)
            new_observation_space = spaces.Dict(new_obs_space_dict)

            # Compute config stages and aliases
            config_stages = list(config.TASK_CONFIG.TASK.StageSuccess.GOALS.keys())
            config_stage_aliases = [s for s in policy.skill_sequence if (not is_navigate(s)) and (not is_reset_arm(s)) and (not is_next_target(s))]
            config_stage_alias_dict = {config_stage_aliases[i]: config_stages[i] for i in range(len(config_stage_aliases))}

            belief_classifier = TrainBeliefClassifier(observation_space=new_observation_space, action_space=env.action_space, prefix=config.TASK_CONFIG.TASK.TYPE, device=args.device, reload=True, runtype="toy")
            energy_sampler = EnergyGoalSampler(belief_classifier) 

        if epoch % num_epoch_trajs == 0:
            ob_trajectories = []
            next_skill_fails = []

        for i_ep in range(5):
            ob = env.reset()
            policy.reset(ob)

            episode_reward = 0.0
            info = {}
            rgb_frames = []
            episode_id = env.current_episode.episode_id
            scene_id = env.current_episode.scene_id

            # Skip episode and keep reproducibility
            if eval_episode_ids is not None and episode_id not in eval_episode_ids:
                print("Skip episode", episode_id)
                continue

            current_skill = None
            current_skills = []
            post_nav_skills = []
            while True:
                step_action = policy.act(ob)
                if step_action is None:
                    print("Terminate the episode given none action")
                    break

                # -------------------------------------------------------------------------- #
                # Visualization
                # -------------------------------------------------------------------------- #
                if args.viewer or args.save_video:
                    # Add additional info
                    info["values"] = step_action.get("values")
                    info["value"] = step_action.get("value")
                    info["success_probs"] = step_action.get("success_probs")

                    metrics = extract_scalars_from_info(info)
                    if args.render_mode == "human":
                        frame = env.render(
                            "human",
                            info=metrics,
                            overlay_info=False,
                            show_info=args.render_info,
                        )
                    else:
                        frame = env.render(args.render_mode)
                        if args.render_info:
                            frame = put_info_on_image(
                                frame, info=metrics, overlay=False
                            )
                    rgb_frames.append(frame)
                # -------------------------------------------------------------------------- #

                if current_skill != policy.current_skill_name: 
                    if is_navigate(policy.current_skill_name):
                        ob_trajectories.append([])
                        post_nav_skills.append(None)

                    if not is_reset_arm(current_skill):
                        prev_skill = current_skill
                    current_skill = policy.current_skill_name
                    current_skills.append(current_skill)

                if is_navigate(prev_skill) and not is_reset_arm(current_skill):
                    post_nav_skills[-1] = current_skill
                
                if is_navigate(policy.current_skill_name):
                    old_ob = {k: ob[k] for k in relevant_ob_keys}
                    old_ob["next_skill"] = skill_ordering_dict[policy.skill_sequence[policy._skill_idx + 2]]
                    old_ob["goal"] = get_goal_position(env)
                    ob_trajectories[-1].append(old_ob)
                    
                    
                # failure_prob = belief_classifier.run_model_sub_traj(ob_trajectories[-1])[-1].item()
                
                # if failure_prob > failure_thresh:
                #     goal = get_goal_position(env)
                #     new_goal = energy_sampler.sample_goal(ob_trajectories[-1], goal)
                #     set_goal_position(env, new_goal)
                #     for i in range(len(ob_trajectories[-1])):
                #         ob_trajectories[-1][i]["goal"] = new_goal

                ob, reward, done, info = env.step(step_action)
                episode_reward += reward
                if done:
                    break

            # -------------------------------------------------------------------------- #
            # Update stats
            # -------------------------------------------------------------------------- #
            metrics = extract_scalars_from_info(info)
            episode_stats = metrics.copy()
            episode_stats["return"] = episode_reward
            all_episode_stats.append(episode_stats)
            next_skill_fails += [not info["stage_success"][config_stage_alias_dict[s]] for s in post_nav_skills]
        env.close()
        if (epoch + 1) % num_epoch_trajs == 0:
            saved_data = dict(ob_trajectories=ob_trajectories, next_skill_fails=next_skill_fails)
            with open(f"mobile_manipulation/goal_failure_belief/toy_belief_train_data/saved_traj_{epoch}.pkl", "wb") as f:
                pickle.dump(saved_data, f)
            

if __name__ == "__main__":
    main()
