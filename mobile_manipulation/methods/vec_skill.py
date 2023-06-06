from collections import OrderedDict
from typing import Dict

from habitat import Config, RLEnv
from habitat.core.vector_env import VectorEnv

from mobile_manipulation.common.registry import (
    vmm_registry as my_registry,
    vbmm_registry as bmy_registry,
)


class Skill:
    def __init__(self, config: Config, saved_actor_critics: dict, vec_rl_env: VectorEnv, vec_rl_env_idx: int):
        self._config = config
        self._saved_actor_critics = saved_actor_critics
        self._vec_rl_env = vec_rl_env
        self._vec_rl_env_idx = vec_rl_env_idx
        self._obs_space = self._vec_rl_env.observation_spaces[self._vec_rl_env_idx]
        self._action_space = self._vec_rl_env.action_spaces[self._vec_rl_env_idx]
        self.device = None

    def reset(self, obs, **kwargs):
        self._elapsed_steps = 0

    def act(self, obs, **kwargs) -> Dict:
        raise NotImplementedError

    def should_terminate(self, obs, **kwargs):
        raise NotImplementedError

    def is_timeout(self):
        timeout = self._config.get("TIMEOUT", 0)
        if timeout > 0:
            return self._elapsed_steps >= timeout
        else:
            return False

    def to(self, device):
        return self


@my_registry.register_skill
class Wait(Skill):
    def act(self, obs, **kwargs):
        self._elapsed_steps += 1
        return {"action": "EmptyAction"}

    def should_terminate(self, obs, **kwargs):
        return self.is_timeout()


@my_registry.register_skill
class Terminate(Skill):
    def act(self, obs, **kwargs):
        self._vec_rl_env.call_at(self._vec_rl_env_idx, "set_terminate", {"should_terminate": True})
        return {"action": "EmptyAction"}

    def should_terminate(self, obs, **kwargs):
        return True


@my_registry.register_skill
class CompositeSkill(Skill):
    skills: Dict[str, Skill]
    _skill_idx: int
    _skill_reset: bool

    def __init__(
        self,
        config: Config,
        vec_rl_env: VectorEnv,
        vec_rl_env_idx: int,
        saved_actor_critics: dict
    ):
        self._config = config
        self._vec_rl_env = vec_rl_env
        self._vec_rl_env_idx = vec_rl_env_idx
        self._saved_actor_critics = saved_actor_critics

        self.skill_sequence = config.get("SKILL_SEQUENCE", config.SKILLS)
        self.skills = self._init_entities(
            entity_names=config.SKILLS,
            register_func=my_registry.get_skill,
            entities_config=config,
            vec_rl_env=vec_rl_env,
            vec_rl_env_idx=vec_rl_env_idx,
        )
        self.set_skill_idx(0)

    def _init_entities(
        self, entity_names, register_func, entities_config=None, **kwargs
    ) -> OrderedDict:
        """Modified from EmbodiedTask."""
        if entities_config is None:
            entities_config = self._config

        entities = OrderedDict()
        for entity_name in entity_names:
            entity_cfg = getattr(entities_config, entity_name)
            entity_type = register_func(entity_cfg.TYPE)
            assert (
                entity_type is not None
            ), f"invalid {entity_name} type {entity_cfg.TYPE}"
            entities[entity_name] = entity_type(config=entity_cfg, saved_actor_critics=self._saved_actor_critics, **kwargs)
        return entities

    @property
    def current_skill_name(self):
        return self.skill_sequence[self._skill_idx]

    @property
    def current_skill(self):
        if self._skill_idx >= len(self.skill_sequence):
            return None
        return self.skills[self.current_skill_name]

    def set_skill_idx(self, idx):
        if idx is None:
            self._skill_idx += 1
        else:
            self._skill_idx = idx

    def reset(self, obs, **kwargs):
        self.set_skill_idx(0)
        # print("Skill <{}> begin.".format(self.current_skill_name))
        self.current_skill.reset(obs, **kwargs)


    def act(self, obs, **kwargs):
        if self.current_skill.should_terminate(obs, **kwargs):
            # print("Skill <{}> terminate.".format(self.current_skill_name))
            self.set_skill_idx(None)
            if self.current_skill is not None:
                # print("Skill <{}> begin.".format(self.current_skill_name))
                self.current_skill.reset(obs, **kwargs)

        if self.current_skill is not None:
            action = self.current_skill.act(obs, **kwargs)
            if action is None:  # nested composite skill terminate
                action = self.act(obs, **kwargs)
            return action
            
    def check_if_done(self, obs):
        return (self.current_skill is None) or (self.current_skill.should_terminate(obs) and (self._skill_idx == (len(self.skill_sequence)-1)))
    
    def to(self, device):
        for skill in self.skills.values():
            if skill.device is None:
                skill.to(device)
        return self

    def should_terminate(self, obs, **kwargs):
        return self._skill_idx >= len(self.skill_sequence)
