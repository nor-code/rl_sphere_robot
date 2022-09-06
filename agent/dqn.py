import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQLearningAgent(nn.Module):
    def __init__(self, state_dim, batch_size, epsilon, gamma, device, algo):
        super().__init__()
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.algo = algo

        # TODO добавить отрицательные действия
        self.wheel_action_arr = np.array([0.15, 0.18, 0.21, 0.23, 0.24, 0.26, 0.28, 0.3])

        self.platform_wheel_action_pair = np.array(
            np.meshgrid(
                np.array([-0.15, -0.18, -0.22, 0.15, 0.18, 0.22]),
                np.array([0.20, 0.23, 0.26, 0.28, 0.3])
            )
        ).T.reshape(-1, 2)

        self.wheel_action_pair = np.array(np.meshgrid(np.zeros(1), self.wheel_action_arr)).T.reshape(-1, 2)

        self.action_count = len(self.platform_wheel_action_pair) + len(self.wheel_action_pair)
        self.all_pairs_actions = np.concatenate((self.platform_wheel_action_pair, self.wheel_action_pair), axis=0)

        self.index_to_pair = dict(zip(range(self.action_count), self.all_pairs_actions))

        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_count)
        )
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)

        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_count)
        )

    def forward(self, state):
        return self.q_network(state)

    def get_qvalues(self, state):
        model_device = next(self.q_network.parameters()).device
        tensor_state = torch.tensor(state, device=model_device, dtype=torch.float32)
        return self.q_network(tensor_state).data.cpu().numpy()

    def sample_actions(self, q_values):
        rand = np.random.rand()

        if rand <= self.epsilon:
            return np.random.choice(range(self.action_count), size=self.batch_size)
        else:
            return q_values.argmax(axis=-1)

    def train_network(self, replay_buffer):
        s, action_index, r, next_s, is_done = replay_buffer.sample(self.batch_size)

        current_state = torch.tensor(s, device=self.device, dtype=torch.float32)
        next_state = torch.tensor(next_s, device=self.device, dtype=torch.float32)
        rewards = torch.tensor(r, device=self.device, dtype=torch.float32)
        done = torch.tensor(is_done.astype('float32'), device=self.device, dtype=torch.float32)
        action_index = torch.tensor(action_index.reshape(-1, 1), device=self.device, dtype=torch.long)

        q = self.q_network(current_state).gather(1, action_index.long()).squeeze(1)

        # Target for Q regression
        if self.algo == 'dqn':  # DQN
            q_target = self.target_network(next_state)
        elif self.algo == 'ddqn':  # Double DQN
            q2 = self.q_network(next_state)
            q_target = self.target_network(next_state)
            q_target = q_target.gather(1, q2.max(1)[1].unsqueeze(1))

        q_backup = rewards + self.gamma * (1 - done) * q_target.max(1)[0]
        q_backup.to(self.device)

        # Update prediction network parameter
        loss = F.mse_loss(q, q_backup.detach())
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        return loss

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())