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
from .art_sensors import GripperStatusMeasureV1
from ..task import RearrangeTask
from habitat.tasks.rearrange.multi_task.composite_task import CompositeTask

@registry.register_measure
class RearrangePlaceSuccess(MyMeasure):
    cls_uuid = "rearrange_place_success"

    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                GripperToRestingDistance.cls_uuid,
                PlaceObjectSuccess.cls_uuid,
                RearrangePlaceReward.cls_uuid,
            ],
        )
        self._obj_movement_timer = self._config.RESET_OBJ_MOVEMENT_TIMER
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        dist = measures[GripperToRestingDistance.cls_uuid].get_metric()
        place_success = measures[PlaceObjectSuccess.cls_uuid].get_metric()
        rest_success = (
            not self._sim.gripper.is_grasped and dist <= self._config.THRESHOLD
        )
        orientation_dist = measures[RearrangePlaceReward.cls_uuid].orientation_dist
        # print(place_success, rest_success, orientation_dist)
        if (place_success) and (self._obj_movement_timer > 0) and (orientation_dist < self._config.OBJ_ROT_THRESHOLD):
            self._obj_movement_timer -= 1
        else:
            self._obj_movement_timer = self._config.RESET_OBJ_MOVEMENT_TIMER
        self._metric = place_success and self._obj_movement_timer == 0 and not self._sim.gripper.is_grasped
        # if self._metric:
        #     print("************ success ************")
        # else:
        #     print(rest_success, place_success, orientation_dist, self._obj_movement_timer)

# -------------------------------------------------------------------------- #
# Reward
# -------------------------------------------------------------------------- #
@registry.register_measure
class RearrangePlaceReward(MyMeasure):
    prev_dist_to_goal: float
    cls_uuid = "rearrange_place_reward"

    def reset_metric(self, *args, task: RearrangeTask, early_terminate: bool = False, **kwargs):
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
        self.prev_success = False
        self.prev_orientation = None
        self.prev_base_pos = None
        self.update_metric(*args, task=task, early_terminate=early_terminate, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, early_terminate: bool = False, **kwargs):
        measures = task.measurements.measures
        obj_to_goal_dist = measures[ObjectToGoalDistance.cls_uuid].get_metric()
        gripper_to_resting_dist = measures[
            GripperToRestingDistance.cls_uuid
        ].get_metric()
        if GripperStatusMeasure.cls_uuid in measures:
            gripper_status = measures[GripperStatusMeasure.cls_uuid].status
        else:
            gripper_status = measures[GripperStatusMeasureV1.cls_uuid].status
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
    def update_metric(self, *args, task: RearrangeTask, early_terminate: bool = True, **kwargs):
        measures = task.measurements.measures
        obj_to_goal_dist = measures[ObjectToGoalDistance.cls_uuid].get_metric()
        gripper_to_resting_dist = measures[
            GripperToRestingDistance.cls_uuid
        ].get_metric()
        if GripperStatusMeasure.cls_uuid in measures:
            gs_measure = measures[GripperStatusMeasure.cls_uuid]
        else:
            gs_measure = measures[GripperStatusMeasureV1.cls_uuid]
        n_drop = gs_measure.get_metric()["drop"]
        gripper_status = gs_measure.status
        # print("gripper_status", gripper_status)
        place_success = measures[PlaceObjectSuccess.cls_uuid].get_metric()
        # print(place_success, obj_to_goal_dist)
        reward = 0.0
        self.orientation_dist = 0
        current_orientation = task.sensor_suite.sensors.get("object_orientation").get_observation(task=task)
        if self.prev_orientation is None:
            self.orientation_dist = 0
            prev_orient_up_dist = 0
        else:
            self.orientation_dist = np.linalg.norm(self.prev_orientation - current_orientation)
            prev_orient_up_dist = np.linalg.norm(self.prev_orientation - np.asarray([0, 0, 1], dtype=np.float32))
        orient_up_dist = np.linalg.norm(current_orientation - np.asarray([0, 0, 1], dtype=np.float32))
        
        self.prev_orientation = current_orientation
        current_base_pos = task._sim.robot.base_pos
        self.prev_base_pos = current_base_pos
        if gripper_status == GripperStatus.DROP:
            if place_success:
                if n_drop == 1:  # first drop
                    reward += self._config.RELEASE_REWARD
                    reward -= self._config.DIST_REWARD * gripper_to_resting_dist
                    # reward -= orient_up_dist
            else:
                reward -= self._config.RELEASE_PENALTY
                if self._config.END_DROP or early_terminate:
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
            # if obj_to_goal_dist < self._config.ORIENTATION_START_THRESHOLD:
            #     reorient_reward = prev_orient_up_dist - orient_up_dist
            #     reward += reorient_reward
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
                # print(f"dist_reward = {dist_reward}")
            else:
                dist_reward = (
                    -gripper_to_resting_dist * self._config.DIST_REWARD
                )
            reward += dist_reward
            if place_success:
                reward += self._config.CONTINUED_SUCCESS_REWARD
                reward -= self.orientation_dist ** 2 * self._config.ROT_REWARD
                # print(f"place_success orientation = {self.orientation_dist ** 2 * self._config.ROT_REWARD}")
            self.prev_dist_to_goal = gripper_to_resting_dist
        elif gripper_status == GripperStatus.PICK_WRONG:
            # Only for composite reward
            reward -= self._config.PICK_PENALTY
            if early_terminate:
                task._is_episode_active = False
                task._is_episode_truncated = self._config.get(
                    "TRUNCATE_PICK_WRONG", False
                )
        else:
            raise RuntimeError(gripper_status)
        self._metric = reward
