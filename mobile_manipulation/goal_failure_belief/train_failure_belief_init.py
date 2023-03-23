from .belief_model_init import FailureBeliefModel
from mobile_manipulation.methods.skills.rl_skills import batch_obs

import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from habitat_baselines.common.tensor_dict import TensorDict
import pickle
import datetime

MAX_STEPS = 600

class TrainBeliefClassifier:

    def __init__(self, observation_space, action_space, device, prefix, lr=3e-4, reload=False, max_grad_norm=0.2, size_batch_sub_trajs=32, runtype="") -> None:
        self._belief_model = FailureBeliefModel(observation_space, action_space).to(device)
        self._observation_space = observation_space
        self._device = device
        self._optimizer = optim.Adam(self._belief_model.parameters(), lr=lr, eps=1e-5)
        current_file_path = os.path.abspath(__file__)
        parent_dir = os.path.abspath(os.path.join(current_file_path, os.pardir))
        self._writer = SummaryWriter(os.path.join(parent_dir, f'{runtype}logs'))
        self._epoch_num = 0
        self._losses = 0
        self._num_sub_traj = 0
        self._saved_model_dir = os.path.join(parent_dir, f'{runtype}models_{prefix}')
        self._max_grad_norm = max_grad_norm
        self._size_batch_sub_trajs = size_batch_sub_trajs
        self._runtype = runtype
        if not os.path.exists(self._saved_model_dir):
            os.mkdir(self._saved_model_dir)
        self._latest_path = os.path.join(self._saved_model_dir, f"model.pt")
        if runtype != "":
            now = datetime.datetime.now()
            dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
            self._loss_log_fname = os.path.join(parent_dir, f"{runtype}train_model_{dt_string}.csv")
            f = open(self._loss_log_fname, "w")
            f.write("Train Loss, Val Loss\n")
            f.close()
        if reload:
            state_dict = torch.load(os.path.join(self._saved_model_dir, f"model.pt"))
            self._belief_model.load_state_dict(state_dict["model_state_dict"])
            self._optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self._epoch_num = state_dict["epoch_num"]
            np.random.set_state(state_dict['np_random_state'])
            torch.random.set_rng_state(state_dict['torch_random_state'])
            random.setstate(state_dict['py_random_state'])
        
    def _compute_ground_truth_beliefs(self, next_skill_fails, batch_size, num_steps=MAX_STEPS):
        ground_truth = np.full((batch_size, num_steps), next_skill_fails) * 1.0
        return torch.from_numpy(ground_truth).to(self._device)
    
    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self._belief_model.parameters(), self._max_grad_norm
        )

    def train_failure_belief_classifier(self, trajectories, next_skill_fails):
        self._belief_model.train()
        criterion = nn.BCELoss()
        indices = np.random.permutation(len(trajectories))
        for i in indices:
            sub_traj = trajectories[i]
            self._optimizer.zero_grad()
            T = len(sub_traj)
            batch = batch_obs(sub_traj, device=self._device)
            step_batch = dict(observations=batch, recurrent_hidden_states=torch.zeros(
                1,
                self._belief_model._crnn.num_recurrent_layers,
                self._belief_model._crnn.rnn_hidden_size,
                device=self._device
            ), masks=torch.ones(
                T,
                1,
                device=self._device,
                dtype=torch.bool,
            ))
            predictions = self._belief_model(step_batch).view(T, 1)                
            ground_truth = self._compute_ground_truth_beliefs(next_skill_fails[i], T, 1)
            loss = criterion(predictions.float(), ground_truth.float())
            loss.backward()
            self.before_step()
            self._optimizer.step()
            self._losses += loss.item()
            self._num_sub_traj += 1

    def eval_model(self, trajectories, next_skill_fails):
        total_loss = 0
        num_sub_traj = 0
        criterion = nn.BCELoss()
        self._belief_model.eval()

        with torch.no_grad():
            for i, sub_traj in enumerate(trajectories):
                N = len(sub_traj)
                batch = batch_obs(sub_traj, device=self._device)
                step_batch = dict(observations=batch, recurrent_hidden_states=torch.zeros(
                    1,
                    self._belief_model._crnn.num_recurrent_layers,
                    self._belief_model._crnn.rnn_hidden_size,
                    device=self._device
                ), masks=torch.zeros(
                    N,
                    1,
                    device=self._device,
                    dtype=torch.bool,
                ))
                predictions = self._belief_model(step_batch).view(N, 1)                
                ground_truth = self._compute_ground_truth_beliefs(next_skill_fails[i], N, 1)
                loss = criterion(predictions.float(), ground_truth.float())
                total_loss += loss.item()
                num_sub_traj += 1
        return total_loss / num_sub_traj

    def save_train_info(self, trajectories=None, next_skill_fails=None, write_sep_file=False):
        # log the performance metrics
        train_loss = self._losses/(self._num_sub_traj)
        self._writer.add_scalar('training_loss', train_loss, self._epoch_num)
        val_loss = "N/A"
        if trajectories is not None:
            val_loss = self.eval_model(trajectories, next_skill_fails)
            self._writer.add_scalar('val_loss', val_loss, self._epoch_num)
        if write_sep_file:
            f = open(self._loss_log_fname, "a")
            f.write(f"{train_loss}, {val_loss}\n")
            f.close()
        save_dict = {
            'model_state_dict': self._belief_model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'epoch_num': self._epoch_num,
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.random.get_rng_state(),
            'py_random_state': random.getstate()
        }
        torch.save(save_dict, os.path.join(self._saved_model_dir, f"model.pt"))

        if self._epoch_num % 10 == 0:
            torch.save(save_dict, os.path.join(self._saved_model_dir, f"checkpoint_{self._epoch_num}.pt"))

        self._epoch_num += 1
        self._losses = 0
        self._num_sub_traj = 0

