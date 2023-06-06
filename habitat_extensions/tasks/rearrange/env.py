import copy
from typing import List, Optional

from gym import spaces
import magnum as mn
import numpy as np
from habitat import Config, Dataset, RLEnv
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_extensions.utils.visualizations.utils import (
    draw_border,
    observations_to_image,
    put_info_on_image,
)

# isort: off
from .sim import RearrangeSim
from .task import RearrangeTask
from . import actions, sensors
from . import sub_tasks, composite_tasks, composite_sensors
from .sensors import GripperStatus


@baseline_registry.register_env(name="RearrangeRLEnv-v0")
class RearrangeRLEnv(RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._prev_env_obs = None

        super().__init__(self._core_env_config, dataset=dataset)

    def reset(self):
        observations = super().reset()
        self._prev_env_obs = observations
        # self._prev_env_obs = copy.deepcopy(observations)
        return observations

    def step(self, *args, **kwargs):
        observations, reward, done, info = super().step(*args, **kwargs)
        self._prev_env_obs = observations
        return observations, reward, done, info

    def get_success(self):
        measures = self._env.task.measurements.measures
        success_measure = self._rl_config.SUCCESS_MEASURE
        if success_measure in measures:
            success = measures[success_measure].get_metric()
        else:
            success = False
        if self._rl_config.get("SUCCESS_ON_STOP", False):
            success = success and self._env.task.should_terminate
        return success

    def get_reward(self, observations: Observations):
        metrics = self._env.get_metrics()

        reward = self._rl_config.SLACK_REWARD
        for reward_measure in self._rl_config.REWARD_MEASURES:
            # print(reward_measure, metrics[reward_measure])
            reward += metrics[reward_measure]

        if self.get_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def get_done(self, observations: Observations):
        # NOTE(jigu): episode is over when task.is_episode_active is False,
        # or time limit is passed.
        done = self._env.episode_over

        success = self.get_success()
        end_on_success = self._rl_config.get("END_ON_SUCCESS", True)
        if success and end_on_success:
            done = True

        return done

    def get_info(self, observations: Observations):
        info = self._env.get_metrics()
        info["is_episode_active"] = self._env.task.is_episode_active
        if self._env.task.is_episode_active:
            # The episode can only be truncated if not active
            assert (
                not self._env.task.is_episode_truncated
            ), self._env._elapsed_steps
            info["is_episode_truncated"] = self._env._past_limit()
        else:
            info["is_episode_truncated"] = self._env.task.is_episode_truncated
        info["elapsed_steps"] = self._env._elapsed_steps
        return info

    def get_reward_range(self):
        # Have not found its usage, but required to be implemented.
        return (np.finfo(np.float32).min, np.finfo(np.float32).max)

    def render(self, mode: str = "human", **kwargs) -> np.ndarray:
        if mode == "human":
            obs = self._prev_env_obs.copy()
            info = kwargs.get("info", {})
            show_info = kwargs.get("show_info", True)
            overlay_info = kwargs.get("overlay_info", True)
            render_uuid = kwargs.get("render_uuid", "robot_third_rgb")

            # rendered_frame = self._env.sim.render(render_uuid)
            rendered_frame = self._env.task.render(render_uuid)
            # rendered_frame = obs[render_uuid]

            # gripper status
            measures = self._env.task.measurements.measures
            gripper_status = measures.get("gripper_status", None)
            if gripper_status is None:
                gripper_status = measures.get("gripper_status_v1", None)
            if gripper_status is not None:
                if gripper_status.status == GripperStatus.PICK_CORRECT:
                    rendered_frame = draw_border(
                        rendered_frame, (0, 255, 0), alpha=0.5
                    )
                elif gripper_status.status == GripperStatus.PICK_WRONG:
                    rendered_frame = draw_border(
                        rendered_frame, (255, 0, 0), alpha=0.5
                    )
                elif gripper_status.status == GripperStatus.DROP:
                    rendered_frame = draw_border(
                        rendered_frame, (255, 255, 0), alpha=0.5
                    )

            if show_info:
                rendered_frame = put_info_on_image(
                    rendered_frame, info, overlay=overlay_info
                )
            obs[render_uuid] = rendered_frame

            return observations_to_image(obs)
        else:
            return super().render(mode=mode)
        
    def set_terminate(self, should_terminate: bool) -> None:
        self._env.task._should_terminate = should_terminate

    def grip_desnap(self, should_desnap: bool) -> None:
        self._env._sim.gripper.desnap(should_desnap)

    def get_observation_space(self) -> spaces.Dict:
        return self.observation_space
    
    def get_action_space(self) -> spaces.Dict:
        return self.action_space
    
    def compute_framed_position(self, task, world_position, frame):
        position = mn.Vector3(world_position)

        robot = self._env._sim.robot
        if frame == "world":
            T = mn.Matrix4.identity_init()
        elif frame == "base":
            T = robot.base_T.inverted()
        elif frame == "gripper":
            T = robot.gripper_T.inverted()
        elif frame == "base_t":
            T = mn.Matrix4.translation(-robot.base_T.translation)
        elif frame == "start_base":
            T = task.start_base_T.inverted()
        else:
            raise NotImplementedError(frame)

        position = T.transform_point(position)
        return np.array(position, dtype=np.float32)
    
    def set_goal(self, goal_residual, current_obs):
        if self._env._sim.gripper.is_grasped:
            place_goal = self.task.place_goal
            world_goal = place_goal + goal_residual
            transformed_goal_at_base = self.compute_framed_position(self.task, world_goal, "base")
            transformed_goal_at_gripper = self.compute_framed_position(self.task, world_goal, "gripper")
            current_obs["place_goal_at_base"] = transformed_goal_at_base
            current_obs["place_goal_at_gripper"] = transformed_goal_at_gripper
        else:
            pick_goal = self.task.pick_goal
            world_goal = pick_goal + goal_residual
            transformed_goal_at_base = self.compute_framed_position(self.task, world_goal, "base")
            transformed_goal_at_gripper = self.compute_framed_position(self.task, world_goal, "gripper")
            current_obs["pick_goal_at_base"] = transformed_goal_at_base
            current_obs["pick_goal_at_gripper"] = transformed_goal_at_gripper
        current_obs["nav_goal_at_base"] = transformed_goal_at_base
        return current_obs
    
    def set_goal_per_skill(self, goal_residual, current_obs, is_grasped=None):
        if is_grasped is None:
            is_grasped = self._env._sim.gripper.is_grasped
        if is_grasped:
            place_goal = self.task.place_goal
            world_goal = place_goal + goal_residual
            transformed_goal_at_base = self.compute_framed_position(self.task, world_goal, "base")
            transformed_goal_at_gripper = self.compute_framed_position(self.task, world_goal, "gripper")
            if "place_goal_at_base" in current_obs:
                current_obs["place_goal_at_base"] = transformed_goal_at_base
            if "place_goal_at_gripper" in current_obs:
                current_obs["place_goal_at_gripper"] = transformed_goal_at_gripper
        else:
            pick_goal = self.task.pick_goal
            world_goal = pick_goal + goal_residual
            transformed_goal_at_base = self.compute_framed_position(self.task, world_goal, "base")
            transformed_goal_at_gripper = self.compute_framed_position(self.task, world_goal, "gripper")
            if "pick_goal_at_base" in current_obs:
                current_obs["pick_goal_at_base"] = transformed_goal_at_base
            if "pick_goal_at_gripper" in current_obs:
                current_obs["pick_goal_at_gripper"] = transformed_goal_at_gripper
        if "nav_goal_at_base" in current_obs:
            current_obs["nav_goal_at_base"] = transformed_goal_at_base
        return current_obs
    
    def set_goal_old(self, goal_residual, current_obs):
        if self._env._sim.gripper.is_grasped:
            goal = current_obs["place_goal_at_base"] + goal_residual
            current_obs["place_goal_at_base"] = goal
        else:
            goal = current_obs["place_goal_at_base"] + goal_residual
            current_obs["pick_goal_at_base"] = goal
        current_obs["nav_goal_at_base"] = goal
        return current_obs
    
    @property
    def task(self):
        return self._env.task
    
    @property
    def sim(self):
        return self._env._sim
    
    def get_metrics(self):
        return self._env.get_metrics()
    
    def get_original_pick_goal(self):
        return self._env.task.original_pick_goal

    def get_original_place_goal(self):
        return self._env.task.original_place_goal
    
    def get_task_is_stop_called(self, action):
        return self._env.task.actions[action].is_stop_called

    def get_robot_arm_joint_pos(self):
        return self._env.sim.robot.arm_joint_pos

    def get_policy_action(self, policy, ob):
        return policy.act(ob)

    def pyb_robot_ik(self, ee_tgt_pos, max_iters=100):
        return self._env.sim.pyb_robot.IK(ee_tgt_pos, max_iters)

    def task_tgt_idx_incr(self):
        self._env.task.set_target(self._env.task.tgt_idx + 1)

    def set_task_is_stop_called(self, action, is_stop_called):
        self._env.task.actions[action].is_stop_called = is_stop_called

    def set_ee_tgt_pos(self, action_names, ee_tgt_pos):
        task_actions = self._env.task.actions
        for action_name in action_names:
            if action_name not in task_actions:
                continue
            task_actions[action_name].ee_tgt_pos = ee_tgt_pos

    def set_ee_pyb_joint_states(self):
        self._env.sim.pyb_robot.set_joint_states(self._env.sim.robot.params.arm_init_params)
    
    def set_robot_arm_motor_pos(self, arm_motor_pos):
        self._env.sim.robot.arm_motor_pos = arm_motor_pos

    def compute_new_actions(self, action_args, original_categorical_action_log_probs, gaussian_action_residual, categorical_all_log_probs, gaussian_flag, testing=False):
        if gaussian_flag:
            action_args = action_args + gaussian_action_residual
            return action_args
        x = original_categorical_action_log_probs + categorical_all_log_probs
        y = np.exp(x)
        probs = y / np.sum(y)
        if testing:
            action_idx = np.argmax(probs)
        else:
            action_idx = np.random.choice(len(probs), p=probs)
        action_args = np.array(action_idx, dtype='long')
        return action_args
    
    def set_early_terminate(self, early_terminate):
        self.task.set_early_terminate(early_terminate)
        return
