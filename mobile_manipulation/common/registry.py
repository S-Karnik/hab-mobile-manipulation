from typing import Optional

from habitat.core.registry import Registry


class MobileManipulationRegistry(Registry):
    @classmethod
    def register_policy(cls, to_register=None, *, name: Optional[str] = None):
        # NOTE(jigu): import on-the-fly to avoid import loop
        from mobile_manipulation.ppo.policy import ActorCritic

        return cls._register_impl(
            "policy", to_register, name, assert_type=ActorCritic
        )

    def register_bilinear_policy(cls, to_register=None, *, name: Optional[str] = None):
        # NOTE(jigu): import on-the-fly to avoid import loop
        from mobile_manipulation.ppo.policy import BilinearActorCritic

        return cls._register_impl(
            "policy", to_register, name, assert_type=BilinearActorCritic
        )

    @classmethod
    def get_policy(cls, name: str):
        return cls._get_impl("policy", name)

    @classmethod
    def register_skill(cls, to_register=None, *, name: Optional[str] = None):
        # NOTE(jigu): import on-the-fly to avoid import loop
        from mobile_manipulation.methods.skill import Skill

        return cls._register_impl(
            "skill", to_register, name, assert_type=Skill
        )

    @classmethod
    def get_skill(cls, name: str):
        return cls._get_impl("skill", name)


mm_registry = MobileManipulationRegistry()

import importlib
class VecMobileManipulationRegistry(Registry):
    gt_skill_names = ['ResetArm', 'NextTarget']
    @classmethod
    def register_policy(cls, to_register=None, *, name: Optional[str] = None):
        # NOTE(jigu): import on-the-fly to avoid import loop
        from mobile_manipulation.ppo.policy import ActorCritic

        return cls._register_impl(
            "policy", to_register, name, assert_type=ActorCritic
        )

    @classmethod
    def get_policy(cls, name: str):
        return cls._get_impl("policy", name)

    @classmethod
    def register_skill(cls, to_register=None, *, name: Optional[str] = None):
        # NOTE(jigu): import on-the-fly to avoid import loop
        from mobile_manipulation.methods.vec_skill import Skill

        return cls._register_impl(
            "skill", to_register, name, assert_type=Skill
        )

    @classmethod
    def get_skill(cls, name: str):
        if name in cls.gt_skill_names:
            my_module = importlib.import_module("mobile_manipulation.methods.vector_skills.vec_gt_skills")
        else:
            my_module = importlib.import_module("mobile_manipulation.methods.vector_skills.vec_rl_skills")
        my_class = getattr(my_module, name)
        return my_class

class VecBilinearMobileManipulationRegistry(Registry):
    gt_skill_names = ['ResetArm', 'NextTarget']
    @classmethod
    def register_policy(cls, to_register=None, *, name: Optional[str] = None):
        # NOTE(jigu): import on-the-fly to avoid import loop
        from mobile_manipulation.ppo.policy import BilinearActorCritic

        return cls._register_impl(
            "policy", to_register, name, assert_type=BilinearActorCritic
        )

    @classmethod
    def get_policy(cls, name: str):
        return cls._get_impl("policy", name)

    @classmethod
    def register_skill(cls, to_register=None, *, name: Optional[str] = None):
        # NOTE(jigu): import on-the-fly to avoid import loop
        from mobile_manipulation.methods.vec_bilinear_skill import Skill

        return cls._register_impl(
            "skill", to_register, name, assert_type=Skill
        )

    @classmethod
    def get_skill(cls, name: str):
        if name in cls.gt_skill_names:
            my_module = importlib.import_module("mobile_manipulation.methods.vector_skills.vec_gt_skills")
        else:
            my_module = importlib.import_module("mobile_manipulation.methods.vector_skills.vec_bilinear_rl_skills")
        my_class = getattr(my_module, name)
        return my_class

vmm_registry = VecMobileManipulationRegistry()
vbmm_registry = VecBilinearMobileManipulationRegistry()
