import numpy as np
import torch
import torch.nn as nn


class DeepQLearningAgent(nn.Module):
    def __init__(self, state_dim, batch_size, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.platform_action = np.array(np.meshgrid(np.linspace(-0.26, 0.26, 5), np.zeros(1))).T.reshape(-1, 2)
        self.wheel_action = np.array(np.meshgrid(np.zeros(1), np.linspace(-0.25, 0.25, 5))).T.reshape(-1, 2)
        self.action_count = len(self.platform_action) + len(self.wheel_action)

        self.all_pairs_actions = np.concatenate((self.platform_action, self.wheel_action), axis=0)
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
