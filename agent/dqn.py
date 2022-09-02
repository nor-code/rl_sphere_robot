import numpy as np
import torch
import torch.nn as nn


class DeepQLearningAgent(nn.Module):
    def __init__(self, state_dim, batch_size, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.platform_action_left = np.array(np.meshgrid(np.linspace(-0.24, -0.17, 4), np.zeros(1))).T.reshape(-1, 2)
        self.zero_action = np.array([[0, 0]])
        self.platform_action_right = np.array(np.meshgrid(np.linspace(0.17, 0.24, 4), np.zeros(1))).T.reshape(-1, 2)
        self.wheel_action = np.array(np.meshgrid(np.zeros(1), np.linspace(-0.25, 0.25, 8))).T.reshape(-1, 2)

        self.action_count = 1 + len(self.platform_action_left) + len(self.platform_action_right) + len(self.wheel_action)

        self.all_pairs_actions = np.concatenate((self.platform_action_left, self.platform_action_right, self.zero_action, self.wheel_action), axis=0)
        self.index_to_pair = dict(zip(range(self.action_count), self.all_pairs_actions))

        self.network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_count)
        )

    def forward(self, state):
        return self.network(state)

    def get_qvalues(self, state):
        model_device = next(self.parameters()).device
        tensor_state = torch.tensor(state, device=model_device, dtype=torch.float32)
        return self.network(tensor_state).data.cpu().numpy()

    def sample_actions(self, q_values):
        rand = np.random.rand()

        if rand <= self.epsilon:
            return np.random.choice(range(self.action_count), size=self.batch_size)
        else:
            return q_values.argmax(axis=-1)
