from habitat.core.registry import registry
import numpy as np

# -------------------------------------------------------------------------- #
# Measure
# -------------------------------------------------------------------------- #
from ..sensors import (
    GripperStatus,
    GripperStatusMeasure,
    GripperToRestingDistance,
    MyMeasure,
    ObjectToGoalDistance,
    PlaceObjectSuccess,
)
from ..task import RearrangeTask


@registry.register_measure
class RearrangePlaceSuccess(MyMeasure):
    cls_uuid = "rearrange_place_success"

    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                GripperToRestingDistance.cls_uuid,
                PlaceObjectSuccess.cls_uuid,
            ],
        )
        self.prev_obj_pos = None
        self.prev_obj_vel = None
        self.prev_ee_pos = None
        self.prev_ee_vel = None
        self.prev_dist = 0
        self.time_step = 0
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        dist = measures[GripperToRestingDistance.cls_uuid].get_metric()
        place_success = measures[PlaceObjectSuccess.cls_uuid].get_metric()
        rest_success = (
            not self._sim.gripper.is_grasped and dist <= self._config.THRESHOLD
        )
        obj_vel_0 = False
        obj_acc_0 = False
        obj_pos = np.array(task.tgt_obj.translation, dtype=np.float32)
        obj_vel = None

        if self.prev_obj_pos is not None:
            obj_vel = obj_pos - self.prev_obj_pos

        obj_acc = None
        if self.prev_obj_vel is not None:
            obj_acc = obj_vel - self.prev_obj_vel
            
        self.prev_obj_pos = obj_pos
        self.prev_obj_vel = obj_vel
        if obj_vel is not None:
            obj_vel_0 = np.linalg.norm(obj_vel) <= self._config.OBJ_REST_THRESHOLD
        if obj_acc is not None:
            obj_acc_0 = np.linalg.norm(obj_acc) <= self._config.OBJ_REST_THRESHOLD

        ee_vel_0 = False
        ee_acc_0 = False
        ee_pos = np.array(self._sim.robot.gripper_pos, dtype=np.float32)
        ee_vel = None

        if self.prev_ee_pos is not None:
            ee_vel = ee_pos - self.prev_ee_pos

        ee_acc = None
        if self.prev_ee_vel is not None:
            ee_acc = ee_vel - self.prev_ee_vel
            
        self.prev_ee_pos = ee_pos
        self.prev_ee_vel = ee_vel
        if ee_vel is not None:
            ee_vel_0 = np.linalg.norm(ee_vel) <= self._config.THRESHOLD
        if ee_acc is not None:
            ee_acc_0 = np.linalg.norm(ee_acc) <= self._config.THRESHOLD
        self._metric = rest_success and place_success and ((obj_vel_0 and obj_acc_0) or (self.time_step == task._max_episode_steps - 1)) and (ee_vel_0 and ee_acc_0)
        # self._metric = rest_success and place_success and ((obj_vel_0 and obj_acc_0)) and (ee_vel_0 and ee_acc_0)
        self.time_step += 1
        self.prev_dist = dist
        
        # if (rest_success and place_success):
        #     print(np.linalg.norm(obj_vel), obj_vel_0, np.linalg.norm(obj_acc), obj_acc_0)
        # if self._metric:
        #     print("****************")
# -------------------------------------------------------------------------- #
# Reward
# -------------------------------------------------------------------------- #
@registry.register_measure
class RearrangePlaceReward(MyMeasure):
    prev_dist_to_goal: float
    cls_uuid = "rearrange_place_reward"

    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        if not kwargs.get("no_dep", False):
            task.measurements.check_measure_dependencies(
                self.uuid,
                [
                    ObjectToGoalDistance.cls_uuid,
                    GripperToRestingDistance.cls_uuid,
                    GripperStatusMeasure.cls_uuid,
                    PlaceObjectSuccess.cls_uuid,
                ],
            )

        self.prev_dist_to_goal = None
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        obj_to_goal_dist = measures[ObjectToGoalDistance.cls_uuid].get_metric()
        gripper_to_resting_dist = measures[
            GripperToRestingDistance.cls_uuid
        ].get_metric()
        gripper_status = measures[GripperStatusMeasure.cls_uuid].status
        # print("gripper_status", gripper_status)

        reward = 0.0

        if gripper_status == GripperStatus.DROP:
            self.prev_dist_to_goal = gripper_to_resting_dist
        elif gripper_status in [
            GripperStatus.PICK_CORRECT,
            GripperStatus.HOLDING_CORRECT,
        ]:
            if self._config.USE_DIFF:
                if self.prev_dist_to_goal is None:
                    diff_dist = 0.0
                else:
                    diff_dist = self.prev_dist_to_goal - obj_to_goal_dist
                    diff_dist = round(diff_dist, 3)
                dist_reward = diff_dist * self._config.DIST_REWARD
            else:
                dist_reward = -obj_to_goal_dist * self._config.DIST_REWARD
            reward += dist_reward
            self.prev_dist_to_goal = obj_to_goal_dist
        elif gripper_status == GripperStatus.NOT_HOLDING:
            if self._config.USE_DIFF:
                if self.prev_dist_to_goal is None:
                    diff_dist = 0.0
                else:
                    diff_dist = (
                        self.prev_dist_to_goal - gripper_to_resting_dist
                    )
                    diff_dist = round(diff_dist, 3)
                dist_reward = diff_dist * self._config.DIST_REWARD
            else:
                dist_reward = (
                    -gripper_to_resting_dist * self._config.DIST_REWARD
                )
            reward += dist_reward
            self.prev_dist_to_goal = gripper_to_resting_dist
        else:
            raise RuntimeError(gripper_status)

        self._metric = reward


@registry.register_measure
class RearrangePlaceRewardV1(RearrangePlaceReward):
    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        obj_to_goal_dist = measures[ObjectToGoalDistance.cls_uuid].get_metric()
        gripper_to_resting_dist = measures[
            GripperToRestingDistance.cls_uuid
        ].get_metric()
        gs_measure = measures[GripperStatusMeasure.cls_uuid]
        n_drop = gs_measure.get_metric()["drop"]
        gripper_status = gs_measure.status
        # print("gripper_status", gripper_status)
        place_success = measures[PlaceObjectSuccess.cls_uuid].get_metric()

        reward = 0.0

        if gripper_status == GripperStatus.DROP:
            if place_success:
                if n_drop == 1:  # first drop
                    reward += self._config.RELEASE_REWARD
            else:
                reward -= self._config.RELEASE_PENALTY
                if self._config.END_DROP:
                    task._is_episode_active = False
                    task._is_episode_truncated = self._config.get(
                        "TRUNCATE_DROP", False
                    )
            self.prev_dist_to_goal = gripper_to_resting_dist
        elif gripper_status == GripperStatus.PICK_CORRECT:
            self.prev_dist_to_goal = obj_to_goal_dist
        elif gripper_status == GripperStatus.HOLDING_CORRECT:
            if self._config.USE_DIFF:
                if self.prev_dist_to_goal is None:
                    diff_dist = 0.0
                else:
                    diff_dist = self.prev_dist_to_goal - obj_to_goal_dist
                    diff_dist = round(diff_dist, 3)
                dist_reward = diff_dist * self._config.DIST_REWARD
            else:
                dist_reward = -obj_to_goal_dist * self._config.DIST_REWARD
            reward += dist_reward
            self.prev_dist_to_goal = obj_to_goal_dist
        elif gripper_status == GripperStatus.NOT_HOLDING:
            if self._config.USE_DIFF:
                if self.prev_dist_to_goal is None:
                    diff_dist = 0.0
                else:
                    diff_dist = (
                        self.prev_dist_to_goal - gripper_to_resting_dist
                    )
                    diff_dist = round(diff_dist, 3)
                dist_reward = diff_dist * self._config.DIST_REWARD
            else:
                dist_reward = (
                    -gripper_to_resting_dist * self._config.DIST_REWARD
                )
            reward += dist_reward
            self.prev_dist_to_goal = gripper_to_resting_dist
        elif gripper_status == GripperStatus.PICK_WRONG:
            # Only for composite reward
            reward -= self._config.PICK_PENALTY
            task._is_episode_active = False
            task._is_episode_truncated = self._config.get(
                "TRUNCATE_PICK_WRONG", False
            )
        else:
            raise RuntimeError(gripper_status)

        self._metric = reward