from typing import Dict, List, Union

import torch
import torch.nn as nn
from mobile_manipulation.ppo.policies.cnn_policy import CRNet, BilinearCRNet
from mobile_manipulation.ppo.policy import CriticHead, BilinearCriticHead, GaussianNet, BilinearGaussianNet, CategoricalNet, ActorCritic, BilinearActorCritic, MultiHeadActorCritic
from mobile_manipulation.common.registry import mm_registry
from gym import spaces
GOAL_DIM = 3

class TaskGoalCriticModel(nn.Module):
    def __init__(self, observation_space, action_space) -> None:
        super().__init__()
        self._crnn = CRNet(
            observation_space=observation_space,
            action_space=action_space,
            use_prev_actions=False,
            rgb_uuids = [],
            depth_uuids = ["robot_head_depth"], 
            state_uuids = ["nav_goal_at_base", "next_skill", "goal"],
            hidden_size = 512,
            state_hidden_sizes = [],
            rnn_hidden_size = 512,
        )
        self._task_level_critic = CriticHead(
            self._crnn.output_size,
            hidden_sizes=[]
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        batch should also include next skills in observations or skill index or an embedding of skills suffix
        """
        crnn_outputs = self._crnn(batch)
        outputs = self._task_level_critic(crnn_outputs["features"])
        return outputs

class TaskGoalActorCriticModel(TaskGoalCriticModel):
    def __init__(self, observation_space, action_space) -> None:
        super().__init__(observation_space, action_space)
        self._goal_select_actor = GaussianNet(
            num_inputs=self._crnn.output_size, 
            num_outputs=GOAL_DIM,
            hidden_sizes=[],
            action_activation="tanh",
            std_transform = "log",
            min_std = -5,
            max_std = 2,
            conditioned_std = False,
            std_init_bias = -5,
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        batch should also include after skills in observations
        """
        crnn_outputs = self._crnn(batch)
        value = self._task_level_critic(crnn_outputs["features"])
        goal = self._goal_select_actor(crnn_outputs["features"])
        return value, goal
    

@mm_registry.register_policy
class GoalPredictorPolicy(ActorCritic):
    @classmethod
    def setup_policy(
        cls, observation_space, action_space
    ):
        # TODO: switch back to use_prev_actions = True
        net = CRNet(
            observation_space=observation_space,
            action_space=action_space,
            use_prev_actions=True,
            rgb_uuids = [],
            depth_uuids = ["robot_head_depth", "robot_arm_depth"], 
            state_uuids = [
                "arm_joint_pos", 
                "is_grasped", 
                "gripper_pos_at_base", 
                "resting_pos_at_base",
                "place_goal_at_gripper", 
                "pick_goal_at_gripper",
                "pick_goal_at_base", 
                "place_goal_at_base", 
                "nav_goal_at_base", 
                "next_skill",
            ],
            hidden_size = 256,
            state_hidden_sizes = [],
            rnn_hidden_size = 256,
        )

        actor = GaussianNet(
            num_inputs=net.output_size, 
            num_outputs=GOAL_DIM,
            hidden_sizes=[],
            action_activation="tanh",
            std_transform = "log",
            min_std = -5,
            max_std = 2,
            conditioned_std = False,
            std_init_bias = -1.5,
            ddpg_init = None,
        )

        critic = CriticHead(
            net.output_size,
            hidden_sizes=[]
        )

        return cls(net=net, actor=actor, critic=critic)
    
def load_goal_critic_model_only(model: TaskGoalActorCriticModel, file_path: str):
    observation_space = model._crnn.obs_space
    action_space = model._crnn.action_space
    critic_model = TaskGoalCriticModel(observation_space, action_space)
    critic_model.load_state_dict(torch.load(file_path))

    # Initialize the weights of crnn and task_level_critic with the pre-trained model weights
    model._crnn.load_state_dict(critic_model._crnn.state_dict())
    model._task_level_critic.load_state_dict(critic_model._task_level_critic.state_dict())
    return model

@mm_registry.register_bilinear_policy
class BilinearGoalPredictorPolicy(BilinearActorCritic):
    @classmethod
    def setup_policy(
        cls, observation_space, action_space
    ):
        # TODO: switch back to use_prev_actions = True
        net = BilinearCRNet(
            observation_space=observation_space,
            action_space=action_space,
            use_prev_actions=True,
            rgb_uuids = [],
            depth_uuids = ["robot_head_depth", "robot_arm_depth"], 
            goal_dep_state_uuids = [
                "resting_pos_at_base",
                "place_goal_at_gripper", 
                "pick_goal_at_gripper",
                "pick_goal_at_base", 
                "place_goal_at_base",
                "nav_goal_at_base", 
            ],
            goal_ind_state_uuids = [
                "arm_joint_pos", 
                "is_grasped", 
                "gripper_pos_at_base", 
                "next_skill",
            ],
            hidden_size = 256,
            state_hidden_sizes = [],
            rnn_hidden_size = 128,
        )

        actor = BilinearGaussianNet(
            num_inputs=net.output_size, 
            num_outputs=GOAL_DIM,
            hidden_sizes=[],
            action_activation="tanh",
            std_transform = "log",
            min_std = -5,
            max_std = 1,
            conditioned_std = False,
            std_init_bias = -2,
            ddpg_init = None,
            pre_bilinear_size = 64
        )

        critic = BilinearCriticHead(
            net.output_size,
            hidden_sizes=[],
            pre_bilinear_size=64
        )
        
        return cls(net=net, actor=actor, critic=critic)

def load_goal_actor_critic_model(model: TaskGoalActorCriticModel, file_path: str):
    observation_space = model._crnn.obs_space
    action_space = model._crnn.action_space
    actor_critic_model = TaskGoalActorCriticModel(observation_space, action_space)
    actor_critic_model.load_state_dict(torch.load(file_path))
    model.load_state_dict(actor_critic_model.state_dict())
    return model

class ResidualRLActionMethod(MultiHeadActorCritic):
    @classmethod
    def setup_policy(
        cls, task_config, skill_config, observation_space, action_space
    ):
        
        net = CRNet(
            observation_space=observation_space,
            action_space=None,
            use_prev_actions=False,
            rgb_uuids = [],
            depth_uuids = ["robot_head_depth", "robot_arm_depth"], 
            state_uuids = [
                "arm_joint_pos", 
                "is_grasped", 
                "gripper_pos_at_base", 
                "resting_pos_at_base",
                "place_goal_at_gripper", 
                "pick_goal_at_gripper",
                "pick_goal_at_base", 
                "place_goal_at_base", 
                "nav_goal_at_base", 
                "next_skill",
            ],
            hidden_size = 256,
            state_hidden_sizes = [],
            rnn_hidden_size = 256,
        )
        skill_config.defrost()
        cls.gaussian_action_space = action_space[task_config.SOLUTION.PlaceRLSkill.ACTION]
        cls.categorical_action_space = action_space[task_config.SOLUTION.NavRLSkill.ACTION]
        modified_gaussian_actor = skill_config.RL.POLICY.GAUSSIAN_ACTOR
        modified_gaussian_actor.std_init_bias = -1.0
        # modified_gaussian_actor.max_std = 0
        skill_config.freeze()
        gaussian_actor = cls.build_gaussian_actor(
                net.output_size, cls.gaussian_action_space, **modified_gaussian_actor
            )
        categorical_actor = cls.build_categorical_actor(
                net.output_size, cls.categorical_action_space, **skill_config.RL.POLICY.CATEGORICAL_ACTOR
            )
        
        critic = CriticHead(
            net.output_size,
            hidden_sizes=[]
        )

        return cls(net=net, gaussian_actor=gaussian_actor, categorical_actor=categorical_actor, critic=critic)