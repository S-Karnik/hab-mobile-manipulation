from typing import Dict, List, Union

import torch
import torch.nn as nn
from mobile_manipulation.ppo.policies.cnn_policy import CRNet
from mobile_manipulation.ppo.policy import ClassifierNet


class FailureBeliefModel(nn.Module):
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
        self._failure_predictor = ClassifierNet(
            num_inputs=self._crnn.output_size,
            hidden_sizes=[]
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        batch should also include after skills in observations
        """
        crnn_outputs = self._crnn(batch)
        outputs = self._failure_predictor(crnn_outputs["features"])
        return outputs
