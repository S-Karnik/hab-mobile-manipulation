import habitat_sim
import magnum as mn
import numpy as np
from habitat.core.env import Env

# from habitat_extensions.tasks.tidy_house.sim import TidyHouseSim
# from habitat_extensions.tasks.tidy_house.task import TidyHouseTask
from habitat_extensions.tasks.rearrange.sim import RearrangeSim
from habitat_extensions.tasks.rearrange.task import RearrangeTask
from habitat_extensions.utils.geo_utils import wrap_angle
from mobile_manipulation.common.registry import (
    mm_registry as my_registry,
)

from ..skill import Skill

def quaternion_to_pybullet(quaternion):
    # Extract the vector and scalar components of the quaternion
    vector = quaternion.vector
    scalar = quaternion.scalar

    # Arrange the components in the order expected by PyBullet (x, y, z, w)
    target_orientation = [vector.x, vector.y, vector.z, scalar]

    return target_orientation


def euler_to_quaternion(roll, pitch, heading):
    # Convert the roll, pitch, and heading angles to a Magnum Euler angles object
    euler_angles = mn.Vector3(roll, pitch, heading)

    # Create a Quaternion from the Euler angles
    quaternion = mn.Quaternion(euler_angles)

    return quaternion

@my_registry.register_skill
class NavGTSkill(Skill):
    def reset(self, obs, **kwargs):
        self._sim: RearrangeSim = self._rl_env.habitat_env.sim
        self._robot = self._sim.robot
        self._task: RearrangeTask = self._rl_env.habitat_env.task
        self._goal = self._task.goal_to_nav

        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(
            self._sim.robot.base_pos, dtype=np.float32
        )
        path.requested_end = np.array(self._goal[0:3], dtype=np.float32)
        found_path = self._sim.pathfinder.find_path(path)
        assert found_path, "No path is found for episode {}".format(
            self._rl_env.habitat_env.current_episode.episode_id
        )
        self.path = path
        self.path_length = len(path.points)
        print("Path length is {}".format(self.path_length))
        self._path_index = 1

        self._elapsed_steps = 0

    def _compute_velocity(
        self, goal_pos, goal_ori, dist_thresh, ang_thresh, turn_thresh
    ):
        base_invT = self._robot.base_T.inverted()
        base_pos = self._robot.base_pos

        direction_world = goal_pos - base_pos[[0, 2]]
        direction_base = base_invT.transform_vector(
            mn.Vector3(direction_world[0], 0, direction_world[1])
        )
        direction = np.array(direction_base)

        distance = np.linalg.norm(direction)
        should_stop = False

        if distance < dist_thresh:
            lin_vel = 0.0

            if goal_ori is None:
                angle = np.arctan2(-direction[2], direction[0])
            else:
                angle = wrap_angle(goal_ori - self._robot.base_angle)
            if ang_thresh is None or np.abs(angle) <= np.deg2rad(ang_thresh):
                ang_vel = 0.0
                should_stop = True
            else:
                ang_vel = angle / self._sim.timestep
        else:
            angle = np.arctan2(-direction[2], direction[0])
            if np.abs(angle) <= np.deg2rad(turn_thresh):
                lin_vel = distance * np.cos(angle) / self._sim.timestep
            else:
                lin_vel = 0.0
            ang_vel = angle / self._sim.timestep

        # print(lin_vel, ang_vel)
        # print(distance, np.rad2deg(angle))

        return lin_vel, ang_vel, should_stop

    def act(self, obs, **kwargs):
        # # Kinematically set the robot
        # self._sim.robot.base_pos = self.path.points[self._elapsed_steps]
        # self._sim.robot.base_angle = self._goal[-1]
        # self._elapsed_steps += 1
        # return {"action": "EmptyAction"}

        goal_pos = self.path.points[self._path_index][[0, 2]]
        if self._path_index == self.path_length - 1:
            lin_vel, ang_vel, should_stop = self._compute_velocity(
                goal_pos,
                self._goal[-1],
                dist_thresh=self._config.DIST_THRESH,
                ang_thresh=self._config.ANG_THRESH,
                turn_thresh=self._config.TURN_THRESH,
            )
            if should_stop:
                self._path_index += 1
                print("Finish the last waypoint", self._path_index)
        else:
            lin_vel, ang_vel, should_stop = self._compute_velocity(
                goal_pos,
                None,
                # dist_thresh=self._config.DIST_THRESH,
                dist_thresh=0.05,
                ang_thresh=None,
                turn_thresh=self._config.TURN_THRESH,
            )
            if should_stop:
                self._path_index += 1
                print("Advanced to next waypoint", self._path_index)
        step_action = {
            "action": "BaseVelAction",
            "action_args": {"velocity": [lin_vel, ang_vel]},
        }

        self._elapsed_steps += 1
        return step_action

    def is_timeout(self):
        timeout = self._config.get("TIMEOUT", 0)
        if timeout > 0:
            return self._elapsed_steps >= timeout
        else:
            return False

    def should_terminate(self, obs, **kwargs):
        if self._path_index == self.path_length:
            return True
        return self.is_timeout()


@my_registry.register_skill
class ResetArm(Skill):
    def reset(self, obs, **kwargs):
        super().reset(obs, **kwargs)
        sim: RearrangeSim = self._rl_env.habitat_env.sim
        self._robot = sim.robot
        self.ee_tgt_pos = np.array([0.5, 0.0, 1.0])
        self._set_ee_tgt_pos()

        sim.pyb_robot.set_joint_states(self._robot.params.arm_init_params)
        arm_tgt_qpos = sim.pyb_robot.IK(self.ee_tgt_pos, max_iters=100)
        cur_qpos = np.array(self._robot.arm_joint_pos)
        tgt_qpos = np.array(arm_tgt_qpos)
        n_step = np.ceil(np.max(np.abs(tgt_qpos - cur_qpos)) / 0.1)
        n_step = max(1, int(n_step))
        self.plan = np.linspace(cur_qpos, tgt_qpos, n_step)
        self._plan_idx = 0

    def _set_ee_tgt_pos(self):
        # task = self._rl_env.habitat_env.task
        task_actions = self._rl_env.habitat_env.task.actions
        action_names = ["ArmGripperAction", "BaseArmGripperAction"]
        for action_name in action_names:
            if action_name not in task_actions:
                continue
            task_actions[action_name].ee_tgt_pos = self.ee_tgt_pos

    def act(self, obs, **kwargs):
        self._robot.arm_motor_pos = self.plan[self._plan_idx]
        self._plan_idx += 1
        self._elapsed_steps += 1
        # print("reset", np.asarray(self._rl_env.habitat_env.sim.pyb_robot.ee_state[4]), self.ee_tgt_pos)
        return {"action": "EmptyAction"}

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            return True
        return self._plan_idx >= len(self.plan)


@my_registry.register_skill
class NextTarget(Skill):
    def reset(self, obs, **kwargs):
        super().reset(obs, **kwargs)
        task = self._rl_env.habitat_env.task
        task.set_target(task.tgt_idx + 1)

    def act(self, obs, **kwargs):
        self._elapsed_steps += 1
        return {"action": "EmptyAction"}

    def should_terminate(self, obs, **kwargs):
        return True

@my_registry.register_skill
class PlaceGTNav(Skill):
    def reset(self, obs, **kwargs):
        print("gt nav skill *******************")
        super().reset(obs, **kwargs)
        sim: RearrangeSim = self._rl_env.habitat_env.sim
        self._sim = sim
        self._robot = sim.robot
        obj_pos_diff = self._rl_env.task.place_goal - np.asarray(self._rl_env.task.tgt_obj.translation)
        self._goal = np.zeros(4)
        self._goal[:-1] = sim.robot.base_pos + obj_pos_diff
        self._goal[-1] = self._robot.base_ori
        waypoints = []
        waypoints.append(sim.robot.base_pos)  # Add the start point as the first waypoint
        waypoints.append(self._goal[:-1])  # Add the end point as the last waypoint

        # Set the desired number of intermediate waypoints
        num_intermediate_waypoints = 10  # Adjust as needed

        # Generate the intermediate waypoints
        for i in range(1, num_intermediate_waypoints):
            t = i / (num_intermediate_waypoints + 1)  # Adjust the interpolation factor as needed
            intermediate_point = (1 - t) * sim.robot.base_pos + t * self._goal[:-1]
            waypoints.insert(-1, intermediate_point)
        self._path_index = 1

        self.path = waypoints
        self.path_length = len(waypoints)
        self._elapsed_steps = 0

    def _compute_velocity(
        self, goal_pos, goal_ori, dist_thresh, ang_thresh, turn_thresh
    ):
        base_invT = self._robot.base_T.inverted()
        base_pos = self._robot.base_pos

        direction_world = goal_pos - base_pos[[0, 2]]
        direction_base = base_invT.transform_vector(
            mn.Vector3(direction_world[0], 0, direction_world[1])
        )
        direction = np.array(direction_base)

        distance = np.linalg.norm(direction)
        should_stop = False
        # print("dist, dist_thresh", distance, dist_thresh)
        # print("direction", direction, np.arctan2(-direction[2], direction[0]))
        # print(self._rl_env.task.tgt_obj.translation)
        if distance < dist_thresh:
            lin_vel = 0.0

            if goal_ori is None:
                angle = np.arctan2(-direction[2], direction[0])
            else:
                angle = wrap_angle(goal_ori - self._robot.base_ori)
            # print("ang. ang thresh", np.abs(angle), np.deg2rad(ang_thresh))
            if ang_thresh is None or np.abs(angle) <= np.deg2rad(ang_thresh):
                ang_vel = 0.0
                should_stop = True
            else:
                ang_vel = angle / self._sim.timestep
        else:
            angle = np.arctan2(-direction[2], direction[0])
            # print(distance, dist_thresh, angle, turn_thresh, np.deg2rad(turn_thresh))
            if np.abs(angle) <= np.deg2rad(turn_thresh):
                lin_vel = distance * np.cos(angle) / self._sim.timestep
            else:
                lin_vel = 0.0
            ang_vel = angle / self._sim.timestep
        print(lin_vel, ang_vel, self._robot.base_T.translation)

        return lin_vel, ang_vel, should_stop

    def act(self, obs, **kwargs):
        goal_pos = self.path[self._path_index][[0, 2]]
        if self._path_index == self.path_length - 1:
            lin_vel, ang_vel, should_stop = self._compute_velocity(
                goal_pos,
                self._goal[-1],
                dist_thresh=self._config.get("DIST_THRESH", 0.001),
                ang_thresh=self._config.get("ANG_THRESH", 0.5),
                turn_thresh=self._config.get("TURN_THRESH", 5.0),
            )
            if should_stop:
                self._path_index += 1
                print("Finish the last waypoint", self._path_index)
        else:
            lin_vel, ang_vel, should_stop = self._compute_velocity(
                goal_pos,
                None,
                # dist_thresh=self._config.DIST_THRESH,
                dist_thresh=0.001,
                ang_thresh=self._config.get("ANG_THRESH", 0.5),
                turn_thresh=self._config.get("TURN_THRESH", 5.0),
            )
            if should_stop:
                self._path_index += 1
                print("Advanced to next waypoint", self._path_index)
        step_action = {
            "action": "BaseVelAction",
            "action_args": {"velocity": [lin_vel, ang_vel]},
        }
        self._elapsed_steps += 1
        return step_action

    def is_timeout(self):
        timeout = self._config.get("TIMEOUT", 0)
        if timeout > 0:
            return self._elapsed_steps >= timeout
        else:
            return False

    def should_terminate(self, obs, **kwargs):
        if self._path_index == self.path_length:
            return True
        return self.is_timeout()

@my_registry.register_skill
class PlaceTranslate(Skill):
    def reset(self, obs, **kwargs):
        super().reset(obs, **kwargs)
        sim: RearrangeSim = self._rl_env.habitat_env.sim
        self._robot = sim.robot
        self._time_buffer = 40
        self.ee_tgt_pos = np.asarray(self._rl_env.habitat_env.sim.pyb_robot.ee_state[4]) 
        self._set_ee_tgt_pos()
        sim.pyb_robot.set_joint_states(self._robot.params.arm_init_params)
        arm_tgt_qpos = sim.pyb_robot.IK(self.ee_tgt_pos, max_iters=10000)
        cur_qpos = np.array(self._robot.arm_joint_pos)
        tgt_qpos = np.array(arm_tgt_qpos)
        n_step = np.ceil(np.max(np.abs(tgt_qpos - cur_qpos)) / 0.01)
        n_step = max(1, int(n_step))
        self.plan = np.linspace(cur_qpos, tgt_qpos, n_step)
        self._plan_idx = 0
        self._sim = sim

    def _set_ee_tgt_pos(self):
        # task = self._rl_env.habitat_env.task
        task_actions = self._rl_env.habitat_env.task.actions
        action_names = ["ArmGripperAction", "BaseArmGripperAction"]
        for action_name in action_names:
            if action_name not in task_actions:
                continue
            task_actions[action_name].ee_tgt_pos = self.ee_tgt_pos

    def compute_rotation_matrix(self, v1, v2):
        # Normalize the input vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        # Compute the rotation axis
        axis = np.cross(v1_norm, v2_norm)
        axis_norm = axis / np.linalg.norm(axis)

        # Compute the rotation angle
        angle = np.arccos(np.dot(v1_norm, v2_norm))

        # Create the rotation matrix
        rotation_matrix = mn.Matrix4.rotation(mn.Rad(angle), mn.Vector3(*axis_norm))

        return rotation_matrix

    def act(self, obs, **kwargs):
        if self._plan_idx < len(self.plan):
            self._robot.arm_motor_pos = self.plan[self._plan_idx]
        self._plan_idx += 1
        self._elapsed_steps += 1
        # print("reset", np.asarray(self._rl_env.habitat_env.sim.pyb_robot.ee_state[4]), self.ee_tgt_pos, self._plan_idx, len(self.plan))
        # print("place_translate", np.asarray(self._rl_env.habitat_env.sim.pyb_robot.ee_state[4]), self.ee_tgt_pos, self._rl_env.task.tgt_obj.translation, self._rl_env.task.place_goal)
        if self._plan_idx >= len(self.plan):
            current_transform = self._sim.get_agent(0).scene_node.transformation

            # Compute the desired rotation matrix from the quaternion
            ee_rot_mat = self._sim.robot.ee_T.rotation()
            ee_quaternion = mn.Quaternion.from_matrix(ee_rot_mat)
            obj_reorient_vec = self.compute_rotation_matrix(self._rl_env.task.place_goal - self._sim.robot.ee_T.translation, self._rl_env.task.place_goal - np.asarray(self._rl_env.task.tgt_obj.translation))
            matrix4 = mn.Matrix4.from_(mn.Matrix3(ee_quaternion.to_matrix()), mn.Vector3())
            rotation_matrix = matrix4 @ obj_reorient_vec
            new_transform = mn.Matrix4((
                (rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], 0),
                (rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], 0),
                (rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], 0),
                (current_transform.translation[0], current_transform.translation[1], current_transform.translation[2], 1)
            ))
            # print(obj_reorient_vec)
            self._sim.get_agent(0).scene_node.transformation = new_transform

        return {"action": "EmptyAction"}

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            return True
        return self._plan_idx >= len(self.plan) + self._time_buffer

@my_registry.register_skill
class PlaceReorient(Skill):
    def compute_rotation_quaternion(self, v1, v2):
        # Normalize the input vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        # Compute the rotation axis
        axis = np.cross(v1_norm, v2_norm)
        axis_norm = axis / np.linalg.norm(axis)

        # Compute the rotation angle
        angle = np.arccos(np.dot(v1_norm, v2_norm))

        # Create the quaternion representing the rotation
        rotation_quaternion = mn.Quaternion(axis_norm, angle)

        return rotation_quaternion
    
    def compute_rotation_matrix(self, v1, v2):
        # Normalize the input vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        # Compute the rotation axis
        axis = np.cross(v1_norm, v2_norm)
        axis_norm = axis / np.linalg.norm(axis)

        # Compute the rotation angle
        angle = np.arccos(np.dot(v1_norm, v2_norm))

        # Create the rotation matrix
        rotation_matrix = mn.Matrix4.rotation(mn.Rad(angle), mn.Vector3(*axis_norm))

        return rotation_matrix

    def reset(self, obs, **kwargs):
        super().reset(obs, **kwargs)
        sim: RearrangeSim = self._rl_env.habitat_env.sim
        self._robot = sim.robot
        self.ee_tgt_pos = np.asarray(self._rl_env.habitat_env.sim.pyb_robot.ee_state[4])
        self.ee_tgt_pos[-1] += self._rl_env.task.place_goal[-1] - np.asarray(self._rl_env.task.tgt_obj.translation)[-1]
        self._set_ee_tgt_pos()
        sim.pyb_robot.set_joint_states(self._robot.params.arm_init_params)
        # Define your Euler angles
        heading_angle = 80.0
        pitch_angle = 30.0
        roll_angle = 130.0
        euler_angles = np.radians([roll_angle, pitch_angle, heading_angle])
        target_orientation = euler_to_quaternion(euler_angles[0], euler_angles[1], euler_angles[2])
        self._time_buffer = 40
        ee_rot_mat = sim.robot.ee_T.rotation()
        ee_quaternion = mn.Quaternion.from_matrix(ee_rot_mat)
        obj_reorient_vec = self.compute_rotation_matrix(self._rl_env.task.place_goal - np.asarray(self._rl_env.task.tgt_obj.translation), self._rl_env.task.place_goal - sim.robot.ee_T.translation)

        # Apply the desired rotation to the end effector
        matrix4 = mn.Matrix4.from_(mn.Matrix3(ee_quaternion.to_matrix()), mn.Vector3())
        self.ee_tgt_ori = matrix4 @ obj_reorient_vec
        # target_ori = quaternion_to_pybullet(target_ori)

        # import pdb; pdb.set_trace()
        arm_tgt_qpos = sim.pyb_robot.IK(self.ee_tgt_pos, max_iters=10000)
        cur_qpos = np.array(self._robot.arm_joint_pos)
        tgt_qpos = np.array(arm_tgt_qpos)
        n_step = np.ceil(np.max(np.abs(tgt_qpos - cur_qpos)) / 0.01)
        n_step = max(1, int(n_step))
        self.plan = np.linspace(cur_qpos, tgt_qpos, n_step)
        self._plan_idx = 0
        self._sim = sim

    def _set_ee_tgt_pos(self):
        # task = self._rl_env.habitat_env.task
        task_actions = self._rl_env.habitat_env.task.actions
        action_names = ["ArmGripperAction", "BaseArmGripperAction"]
        for action_name in action_names:
            if action_name not in task_actions:
                continue
            task_actions[action_name].ee_tgt_pos = self.ee_tgt_pos

    def act(self, obs, **kwargs):
        if self._plan_idx < len(self.plan):
            self._robot.arm_motor_pos = self.plan[self._plan_idx]
        self._plan_idx += 1
        self._elapsed_steps += 1
        # print("reset", np.asarray(self._rl_env.habitat_env.sim.pyb_robot.ee_state[4]), self.ee_tgt_pos, self._plan_idx, len(self.plan))
        # print(np.linalg.norm(self._rl_env.task.tgt_obj.translation - self._rl_env.task.place_goal))
        # print(self._rl_env.task.tgt_obj.translation, self._rl_env.task.place_goal)
        if self._plan_idx < len(self.plan) + self._time_buffer:
            return {"action": "EmptyAction"}
        # current_transform = self._sim.get_agent(0).scene_node.transformation

        # # Compute the desired rotation matrix from the quaternion
        # ee_rot_mat = self._sim.robot.ee_T.rotation()
        # ee_quaternion = mn.Quaternion.from_matrix(ee_rot_mat)
        # obj_reorient_vec = self.compute_rotation_matrix(self._rl_env.task.place_goal - np.asarray(self._rl_env.task.tgt_obj.translation), self._rl_env.task.place_goal - self._sim.robot.ee_T.translation)
        # matrix4 = mn.Matrix4.from_(mn.Matrix3(ee_quaternion.to_matrix()), mn.Vector3())
        # rotation_matrix = matrix4 @ obj_reorient_vec
        # new_transform = mn.Matrix4((
        #     (rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], 0),
        #     (rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], 0),
        #     (rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], 0),
        #     (current_transform.translation[0], current_transform.translation[1], current_transform.translation[2], 1)
        # ))
        # # print(obj_reorient_vec)
        # self._sim.get_agent(0).scene_node.transformation = new_transform
        return {"action": "ArmGripperAction", "action_args": {
                "arm_action": [0, 0, 0],
                "gripper_action": -1,
            }}

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            return True
        return self._plan_idx >= len(self.plan) + self._time_buffer
