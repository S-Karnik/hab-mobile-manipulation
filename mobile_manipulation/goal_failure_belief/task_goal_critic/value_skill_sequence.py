from .task_success_value_network import TaskGoalCriticModel
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
from torch.optim.lr_scheduler import LambdaLR

MAX_STEPS = 600

class TrainSkillValueNetwork:

    def __init__(self, observation_space, action_space, device, prefix, lr=2.5e-4, reload=False, max_grad_norm=0.5, size_batch_sub_trajs=32, runtype="", max_epochs=2000) -> None:
        self._belief_model = TaskGoalCriticModel(observation_space, action_space).to(device)
        self._observation_space = observation_space
        self._device = device
        self._optimizer = optim.Adam(self._belief_model.parameters(), lr=lr, weight_decay=2e-5)
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
        self._max_epochs = max_epochs
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
            # self._optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self._epoch_num = state_dict["epoch_num"]
            np.random.set_state(state_dict['np_random_state'])
            torch.random.set_rng_state(state_dict['torch_random_state'])
            random.setstate(state_dict['py_random_state'])
        self._lr_lambda = lambda x: max(
            self.get_epoch_progress(), 0.0
        )
        self._lr_scheduler = LambdaLR(
            optimizer=self._optimizer, lr_lambda=self._lr_lambda
        )

    def get_epoch_progress(self):
        return self._epoch_num/self._max_epochs
    
    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self._belief_model.parameters(), self._max_grad_norm
        )

    def _compute_ground_truth_values(self, sub_traj_lengths, discounted_returns, batch_size=1):
        """
        Assume discounted_returns has length batch_size
        """
        ground_truth = np.repeat(np.repeat(discounted_returns, sub_traj_lengths).reshape(-1, 1), repeats=batch_size, axis=1)
        return torch.from_numpy(ground_truth).to(dtype=torch.float32, device=self._device)

    def train_model(self, trajectories, gt_discounted_returns):
        self._belief_model.train()
        criterion = nn.MSELoss()
        num_trajs = len(trajectories)
        # import time
        ordering = np.random.permutation(num_trajs)
        for i in ordering:
            # t_1 = time.time()
            sub_trajs = trajectories[i]
            self._optimizer.zero_grad()
            T = len(sub_trajs)
            batch = batch_obs(sub_trajs, device=self._device)
            # t_2 = time.time()
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
            predictions = self._belief_model(step_batch)
            # t_3 = time.time()
            # import pdb; pdb.set_trace()
            loss = criterion(predictions, torch.from_numpy(gt_discounted_returns[i]).view(-1, 1).to(self._device))
            loss.backward()
            self.before_step()
            self._optimizer.step()
            # t_4 = time.time()
            # print(t_2 - t_1, t_3 - t_2, t_4 - t_3)
            self._losses += loss.item()
            self._num_sub_traj += 1

    def run_model_batch_sub_traj(self, sub_trajs):
        N = len(sub_trajs)
        T = len(sub_trajs[0])
        batched_sub_trajs = sum(sub_trajs, [])
        batch = batch_obs(batched_sub_trajs, device=self._device)
        step_batch = dict(observations=batch, recurrent_hidden_states=torch.zeros(
            T,
            self._belief_model._crnn.num_recurrent_layers,
            self._belief_model._crnn.rnn_hidden_size,
            device=self._device
        ), masks=torch.ones(
            N,
            T,
            device=self._device,
            dtype=torch.bool,
        ))
        predictions = self._belief_model(step_batch).view(N, T)
        return predictions

    def run_model_sub_traj(self, sub_traj):
        N = len(sub_traj)
        batch = batch_obs(sub_traj, device=self._device)
        step_batch = dict(observations=batch, recurrent_hidden_states=torch.zeros(
            1,
            self._belief_model._crnn.num_recurrent_layers,
            self._belief_model._crnn.rnn_hidden_size,
            device=self._device
        ), masks=torch.ones(
            N,
            1,
            device=self._device,
            dtype=torch.bool,
        ))
        predictions = self._belief_model(step_batch).view(N, 1)
        return predictions

    def eval_model(self, trajectories, gt_discounted_returns):
        total_loss = 0
        num_sub_traj = 0
        self._belief_model.eval()
        criterion = nn.MSELoss()
        num_trajs = len(trajectories)

        with torch.no_grad():
            for i in range(num_trajs):
                # t_1 = time.time()
                sub_trajs = trajectories[i]
                self._optimizer.zero_grad()
                T = len(sub_trajs)
                batch = batch_obs(sub_trajs, device=self._device)
                # t_2 = time.time()
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
                predictions = self._belief_model(step_batch)
                # t_3 = time.time()
                # import pdb; pdb.set_trace()
                loss = criterion(predictions, torch.from_numpy(gt_discounted_returns[i]).view(-1, 1).to(self._device))
                total_loss += loss.item()
                num_sub_traj += 1
        return total_loss / num_sub_traj

    def save_train_info(self, trajectories=None, discounted_returns=None, write_sep_file=False):
        # log the performance metrics
        train_loss = self._losses/(self._num_sub_traj)
        self._writer.add_scalar('training_loss', train_loss, self._epoch_num)
        val_loss = "N/A"
        if trajectories is not None:
            val_loss = self.eval_model(trajectories, discounted_returns)
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
        self._lr_scheduler.step()

