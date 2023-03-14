from .belief_model_init import FailureBeliefModel
from mobile_manipulation.methods.skills.rl_skills import batch_obs

import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class TrainBeliefClassifier:

    def __init__(self, observation_space, action_space, device, prefix, num_epochs_per_update=10, lr=1e-3, reload=False) -> None:
        self._belief_model = FailureBeliefModel(observation_space, action_space).to(device)
        self._device = device
        self._optimizer = optim.SGD(self._belief_model.parameters(), lr=lr)
        self._num_epochs = num_epochs_per_update
        current_file_path = os.path.abspath(__file__)
        parent_dir = os.path.abspath(os.path.join(current_file_path, os.pardir))
        self._writer = SummaryWriter(os.path.join(parent_dir, 'logs'))
        self._num_updates = 0
        self._saved_model_dir = os.path.join(parent_dir, f'models_{prefix}')
        if not os.path.exists(self._saved_model_dir):
            os.mkdir(self._saved_model_dir)
        self._latest_path = os.path.join(self._saved_model_dir, f"model.pt")
        if reload:
            state_dict = torch.load(self._latest_path)
            self._belief_model.load_state_dict(state_dict["model_state_dict"])
            self._optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self._num_updates = state_dict["num_updates"]
            np.random.set_state(state_dict['np_random_state'])
            torch.random.set_rng_state(state_dict['torch_random_state'])
            random.setstate(state_dict['py_random_state'])

    def _compute_ground_truth_beliefs(self, next_skill_fails, N):
        ground_truth = np.full((N, 1), next_skill_fails) * 1.0
        return torch.from_numpy(ground_truth)

    def train_failure_belief_classifier(self, trajectories, next_skill_fails):
        self._belief_model.train()
        criterion = nn.BCELoss()
        avg_losses = 0
        for _ in range(self._num_epochs):
            for i, sub_traj in enumerate(trajectories):
                N = len(sub_traj)
                batch = batch_obs(sub_traj, device=self._device)
                step_batch = dict(observations=batch, recurrent_hidden_states=torch.zeros(
                    N,
                    self._belief_model._crnn.num_recurrent_layers,
                    self._belief_model._crnn.rnn_hidden_size,
                    device=self._device
                ), masks=torch.zeros(
                    1,
                    N,
                    device=self._device,
                    dtype=torch.bool,
                ))
                predictions = self._belief_model(step_batch).view(N, 1)                
                ground_truth = self._compute_ground_truth_beliefs(next_skill_fails[i], N).to(self._device)
                loss = criterion(predictions.float(), ground_truth.float())
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                avg_losses += loss.item()
        # log the performance metrics
        self._writer.add_scalar('training_loss', avg_losses/(self._num_epochs * len(trajectories)), self._num_updates)
        self._writer.add_scalar('learning_rate', self._optimizer.param_groups[0]['lr'], self._num_updates)
        self._num_updates += 1
        save_dict = {
            'model_state_dict': self._belief_model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'num_updates': self._num_updates,
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.random.get_rng_state(),
            'py_random_state': random.getstate()
        }
        torch.save(save_dict, os.path.join(self._saved_model_dir, f"model.pt"))

        if self._num_updates % 5 == 0:
            torch.save(save_dict, os.path.join(self._saved_model_dir, f"checkpoint_{self._num_updates}.pt"))
            
        
