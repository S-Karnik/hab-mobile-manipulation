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
    vmm_registry as my_registry,
)

from ..vec_skill import Skill


@my_registry.register_skill
class ResetArm(Skill):
    def reset(self, obs, **kwargs):
        super().reset(obs, **kwargs)
        self.ee_tgt_pos = np.array([0.5, 0.0, 1.0])
        self._set_ee_tgt_pos()

        self._vec_rl_env.call_at(self._vec_rl_env_idx, "set_ee_pyb_joint_states", {})
        arm_tgt_qpos = self._vec_rl_env.call_at(self._vec_rl_env_idx, "pyb_robot_ik", {"ee_tgt_pos": self.ee_tgt_pos})
        cur_qpos = self._vec_rl_env.call_at(self._vec_rl_env_idx, "get_robot_arm_joint_pos", {})
        tgt_qpos = np.array(arm_tgt_qpos)
        n_step = np.ceil(np.max(np.abs(tgt_qpos - cur_qpos)) / 0.1)
        n_step = max(1, int(n_step))
        self.plan = np.linspace(cur_qpos, tgt_qpos, n_step)
        self._plan_idx = 0

    def _set_ee_tgt_pos(self):
        self._vec_rl_env.call_at(self._vec_rl_env_idx, "set_ee_tgt_pos", 
                                        {"action_names": ["ArmGripperAction", "BaseArmGripperAction"],
                                         "ee_tgt_pos": self.ee_tgt_pos})

    def act(self, obs, **kwargs):
        self._vec_rl_env.call_at(self._vec_rl_env_idx, "set_robot_arm_motor_pos", {"arm_motor_pos": self.plan[self._plan_idx]})
        self._plan_idx += 1
        self._elapsed_steps += 1
        return {"action": "EmptyAction"}

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            return True
        return self._plan_idx >= len(self.plan)


@my_registry.register_skill
class NextTarget(Skill):
    def reset(self, obs, **kwargs):
        super().reset(obs, **kwargs)
        self._vec_rl_env.call_at(self._vec_rl_env_idx, "task_tgt_idx_incr", {})

    def act(self, obs, **kwargs):
        self._elapsed_steps += 1
        return {"action": "EmptyAction"}

    def should_terminate(self, obs, **kwargs):
        return True
