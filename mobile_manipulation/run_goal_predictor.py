import argparse
import json
import os
import os.path as osp
import re
import time
import magnum as mn
import numpy as np
import random
import torch

from habitat import Config, logger
from habitat_baselines.utils.common import batch_obs, generate_video
from mobile_manipulation.goal_failure_belief.goal_actor_critic.goal_actor_critic_skill_sequence_fixed_env import TrainGoalActorCriticFixedEnv

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
from mobile_manipulation.goal_failure_belief.goal_actor_critic.goal_actor_critic_skill_sequence import TrainGoalActorCritic
from mobile_manipulation.goal_failure_belief.goal_actor_critic.bilinear_goal_actor_critic_skill_sequence import TrainBilinearGoalActorCritic
from mobile_manipulation.utils.common import warn

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

def preprocess_skill_config(config: Config, args):
    config_path = args.skill_config_path
    run_type = args.run_type

    config.defrost()

    # placeholders supported in config
    fileName = osp.splitext(osp.basename(config_path))[0]
    runName = get_run_name()
    timestamp = time.strftime("%y%m%d")
    substitutes = dict(
        fileName=fileName,
        runName=runName,
        runType=run_type,
        timestamp=timestamp,
    )

    config.PREFIX = config.PREFIX.format(**substitutes)
    config.BASE_RUN_DIR = config.BASE_RUN_DIR.format(**substitutes)

    for key in ["CHECKPOINT_FOLDER"]:
        config[key] = config[key].format(
            prefix=config.PREFIX, baseRunDir=config.BASE_RUN_DIR, **substitutes
        )

    for key in ["LOG_FILE", "TENSORBOARD_DIR", "VIDEO_DIR"]:
        if key not in config:
            warn(f"'{key}' is missed in the config")
            continue
        if run_type == "train":
            prefix = config.PREFIX
        else:
            prefix = config.EVAL.PREFIX or config.PREFIX
        config[key] = config[key].format(
            prefix=prefix,
            baseRunDir=config.BASE_RUN_DIR,
            **substitutes,
        )
    
    # Support relative path like "@/ckpt.pth"
    config.EVAL.CKPT_PATH = config.EVAL.CKPT_PATH.replace(
        "@", config.CHECKPOINT_FOLDER
    )

    # Override
    if args.split is not None:
        if run_type == "train":
            config.TASK_CONFIG.DATASET.SPLIT = args.split
        else:
            config.EVAL.SPLIT = args.split
            
    config.freeze()

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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="config_path", type=str, required=True)
    parser.add_argument("--cfg-skill", dest="skill_config_path", type=str, required=True)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
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
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu-id", type=int)

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
    skill_config = get_config(args.skill_config_path, opts=args.opts)
    preprocess_config(args.config_path, config)
    preprocess_skill_config(skill_config, args)
    torch.set_num_threads(1)
    seed = args.seed

    config.defrost()
    if args.split is not None:
        config.TASK_CONFIG.DATASET.SPLIT = args.split
    # if not args.shuffle:
    #     config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    #     config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = False
    sensors = config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = [
        x for x in sensors if "RGB" not in x
    ]
    config.TASK_CONFIG.SEED = seed
    config.freeze()

    if args.train_seed is not None:
        update_ckpt_path(config, seed=args.train_seed)

    if args.save_log:
        if config.LOG_FILE:
            log_dir = os.path.dirname(config.LOG_FILE)
            os.makedirs(log_dir, exist_ok=True)
            logger.add_filehandler(config.LOG_FILE)
        logger.info(config)
        logger.info("commit id: {}".format(get_git_commit_id()))
    
    gpu_id = args.gpu_id
    # seed = 101
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.split == "val":
        trainer = TrainGoalActorCriticFixedEnv(composite_task_config=config, skill_config=skill_config, seed=seed, gpu_id=gpu_id)
    else:
        # trainer = TrainGoalActorCritic(composite_task_config=config, skill_config=skill_config, seed=seed, gpu_id=gpu_id)
        trainer = TrainBilinearGoalActorCritic(composite_task_config=config, skill_config=skill_config, seed=seed, gpu_id=gpu_id)
    trainer.train()

if __name__ == "__main__":
    main()
