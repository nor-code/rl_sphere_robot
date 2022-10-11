import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_control.rl.control import PhysicsError
from dm_env import StepType


class DeepQLearningAgent(nn.Module):
    def __init__(self,
                 state_dim, batch_size, epsilon,
                 gamma, device, algo,
                 replay_buffer, writer, refresh_target):
        super().__init__()
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.algo = algo
        self.replay_buffer = replay_buffer
        self.writer = writer
        self.loss_history = list()
        self.refresh_target = refresh_target

        # TODO добавить отрицательные действия
        self.wheel_action_arr = np.array([0.15, 0.18, 0.21, 0.23, 0.24, 0.26, 0.28, 0.3])

        self.platform_wheel_action_pair = np.array(
            np.meshgrid(
                np.array([-0.22, -0.205, -0.18, -0.15, 0.15, 0.18, 0.205, 0.22]),  # add 0.2205
                np.array([0.12, 0.20, 0.23, 0.26, 0.28, 0.3, 0.33])  # 0.15
            )
        ).T.reshape(-1, 2)
        # [0.2205, 0.15]
        self.wheel_action_pair = np.array(np.meshgrid(np.zeros(1), self.wheel_action_arr)).T.reshape(-1, 2)

        self.action_count = len(self.platform_wheel_action_pair) + len(self.wheel_action_pair)
        self.all_pairs_actions = np.concatenate((self.platform_wheel_action_pair, self.wheel_action_pair), axis=0)

        self.index_to_pair = dict(zip(range(self.action_count), self.all_pairs_actions))

        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.Tanhshrink(),

            nn.Linear(1024, 4096),
            nn.LeakyReLU(),

            nn.Linear(4096, 8192),
            nn.LeakyReLU(),

            nn.Linear(8192, 8192),
            nn.LeakyReLU(),

            nn.Linear(8192, self.action_count)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3, weight_decay=1e-4)

        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.Tanhshrink(),

            nn.Linear(1024, 4096),
            nn.LeakyReLU(),

            nn.Linear(4096, 8192),
            nn.LeakyReLU(),

            nn.Linear(8192, 8192),
            nn.LeakyReLU(),

            nn.Linear(8192, self.action_count)
        ).to(self.device)

    def forward(self, state):
        return self.q_network(state)

    def get_qvalues(self, state):
        self.q_network.eval()
        model_device = next(self.q_network.parameters()).device
        tensor_state = torch.tensor(state, device=model_device, dtype=torch.float32)
        return self.q_network(tensor_state).data.cpu().numpy()

    def sample_actions(self, q_values):
        # if self.epsilon > 0:
        #     return np.random.choice(range(self.action_count), size=1)
        # else:
        #     return q_values.argmax(axis=-1)
        rand = np.random.rand()

        if rand <= self.epsilon:
            return np.random.choice(range(self.action_count), size=1)
        else:
            return q_values.argmax(axis=-1)

    def train_model(self):
        self.q_network.train()

        s, action_index, r, next_s, is_done = self.replay_buffer.sample(self.batch_size)

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

        del current_state, next_state, rewards, done, action_index
        self.loss_history.append(loss.item())

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_learn_freq(self):
        if self.replay_buffer.buffer_len() >= self.replay_buffer.get_maxsize():
            return 32
        return 32

    def play_episode(self, initial_state, enviroment, episode_timeout, n_steps, global_iteration, episode):
        s = initial_state
        total_reward = 0
        self.q_network.eval()

        for i in range(n_steps):
            global_iteration += 1

            qvalues = self.get_qvalues([s])
            action_idx = self.sample_actions(qvalues)[0]
            action = self.index_to_pair[action_idx]

            try:
                timestep = enviroment.step(action)
            except PhysicsError:
                print("поломка в физике, метод play_and_record")
                _enviroment = enviroment.reset()
                s = _enviroment.observation
                break

            state = timestep.observation
            is_done = timestep.step_type == StepType.LAST or enviroment.physics.data.time > episode_timeout

            if timestep.reward is None:
                break

            total_reward += timestep.reward

            self.replay_buffer.add(s, action_idx, timestep.reward, state, is_done)  # agent add to cash

            if global_iteration > self.batch_size and global_iteration % self.get_learn_freq() == 0:
                self.train_model()

            if is_done:
                _enviroment = enviroment.reset()
                s = _enviroment.observation
                break

            s = state

        if len(self.loss_history) > 0:
            self.writer.add_scalar("average loss", round(np.mean(self.loss_history), 5), episode)
            self.writer.add_scalar("loss", self.loss_history[-1], episode)

        if global_iteration % self.refresh_target == 0 and global_iteration > self.batch_size:
            print("copy parameters from agent to target")
            self.update_target_network()

        return total_reward, global_iteration

    def run_eval_mode(self, enviroment, episode_timeout, s, t_max):
        reward = 0.0
        self.q_network.eval()

        for _ in range(t_max):
            action = self.get_action(s)

            try:
                timestep = enviroment.step(action)
                is_done = timestep.step_type == StepType.LAST or enviroment.physics.data.time > episode_timeout

                if timestep.reward is None:
                    break

                reward += timestep.reward
                if is_done:
                    break
            except PhysicsError:
                print("поломка в физике, метод run_eval_mode")
                break

        return reward

    def get_action(self, state):
        qvalues = self.get_qvalues([state])
        action = self.index_to_pair[qvalues.argmax(axis=-1)[0]]
        return action