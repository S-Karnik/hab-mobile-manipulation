import numpy as np
import torch

class EnergyGoalSampler:

    def __init__(self, model, goal_radius=1.0, num_goal_samples=100) -> None:
        self._model = model
        self._goal_radius = goal_radius
        self._num_goal_samples = num_goal_samples

    def update_model(self, model) -> None:
        self._model = model

    def random_samples_in_sphere(self, goal):
        assert len(goal) == 3
        # Generate a random radius that is less than the radius of the sphere
        r = self._goal_radius * np.cbrt(np.random.uniform(0, 1, self._num_goal_samples))
        # Generate a random vector that lies on the surface of the unit sphere
        samples = np.random.multivariate_normal(goal, np.eye(3), self._num_goal_samples).T
        norm = np.sqrt(samples[0]**2 + samples[1]**2 + samples[2]**2)
        samples[0] /= norm
        samples[1] /= norm
        samples[2] /= norm
        # Scale the vector by the random radius to generate a random point within the sphere
        samples[0] *= r
        samples[1] *= r
        samples[2] *= r
        return samples.T
    
    def energy_based_sample(self, weights):
        weights = -torch.log((1 / (weights + 1e-8)) - 1)
        exp_probs = torch.exp(weights)
        probs = exp_probs / torch.sum(exp_probs)
        return probs

    def sample_goal(self, sub_traj, goal):
        samples = self.random_samples_in_sphere(goal)
        assert samples.shape[1] == 3
        goal_mod_sub_trajs = []
        for sample in samples:
            ob_sample = []
            for ob in sub_traj:
                new_ob = ob.copy()
                new_ob["goal"] = sample
                ob_sample.append(new_ob)
            goal_mod_sub_trajs.append(ob_sample)
        goal_probs = self._model.run_model_batch_sub_traj(goal_mod_sub_trajs) # N x T
        success_goal_probs = 1 - goal_probs[:, -1]
        probs = self.energy_based_sample(success_goal_probs).cpu().detach().numpy()
        goal_index = np.random.choice(samples.shape[0], p=probs)
        goal = samples[goal_index]
        return goal
