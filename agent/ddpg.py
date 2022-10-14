import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_control.rl.control import PhysicsError
from dm_env import StepType
from torch import Tensor


class DeepDeterministicPolicyGradient(object):
    """An implementation of the Deep Deterministic Policy Gradient (DDPG) agent."""

    def __init__(self,
                 obs_dim,
                 device,
                 act_dim,
                 replay_buffer,
                 writer,
                 gamma=0.99,
                 act_noise=0.1,
                 batch_size=1,
                 gradient_clip_policy=0.5,  # 0.5
                 gradient_clip_qf=1.0,  # 1.0
                 policy_losses=list(),
                 qf_losses=list()):

        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.replay_buffer = replay_buffer
        self.epsilon = 1
        self.gamma = gamma
        self.writer = writer
        self.batch_size = batch_size

        self.gradient_clip_policy = gradient_clip_policy
        self.gradient_clip_qf = gradient_clip_qf
        self.policy_losses = policy_losses
        self.qf_losses = qf_losses

        # Main network Actor & Critic
        self.policy = Actor(self.obs_dim, self.device)
        self.qf = Critic(self.obs_dim + self.act_dim, self.device)

        # Target network Actor & Critic
        self.policy_target = Actor(self.obs_dim, self.device)
        self.qf_target = Critic(self.obs_dim + self.act_dim, self.device)

        # Initialize target parameters to match main parameters
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.qf.load_state_dict(self.qf_target.state_dict())

        # Create optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3, weight_decay=1e-4)
        self.qf_optimizer = torch.optim.Adam(self.qf.parameters(), lr=2e-3, weight_decay=1e-4)

        # for noize action per one episode
        self.phase_platform = np.random.uniform(-np.pi, np.pi, size=1)
        self.phase_wheel = np.random.uniform(-np.pi, np.pi, size=1)
        self.sigma, self.amp, self.omega = np.random.randn(3)
        self.prev_action_platform = 0
        self.prev_action_wheel = 0
        self.alpha = 0 # для сглаживающего фильтра

    def sample_actions(self, state, t, i):
        if self.epsilon > 0:
            return [np.random.uniform(-0.97, 0.97, size=1)[0], np.random.uniform(0.1, 0.5, size=1)[0]]  # only random
        else:
            self.policy.to_eval_mode()
            action = self.get_action([state])

            platform, wheel = action[0][0], action[0][1]

            mu_p = 0.001 * self.amp * np.sin(self.omega * t + self.phase_platform)
            mu_w = 0.0008 * self.amp * np.sin(self.omega * t + self.phase_wheel)
            sigma = np.sqrt(self.sigma)

            platform += (sigma * np.random.randn(1) + mu_p)
            wheel += (sigma * np.random.randn(1) + mu_w)

            platform = np.clip(platform, -0.97, 0.97)
            wheel = np.clip(wheel, 0.1, 0.5)

            self.writer.add_scalar("noise_action plat", platform, i)
            self.writer.add_scalar("noise_action wheel", wheel, i)

            platform = self.alpha * platform + (1 - self.alpha) * self.prev_action_platform
            wheel = self.alpha * wheel + (1 - self.alpha) * self.prev_action_wheel

            return [platform[0], wheel[0]]  # with noise

    def train_model(self):
        self.policy.to_train_mode()

        s, actions, r, next_s, is_done = self.replay_buffer.sample(self.batch_size)

        obs1 = torch.tensor(s, device=self.device, dtype=torch.float32)
        obs2 = torch.tensor(next_s, device=self.device, dtype=torch.float32)
        acts = torch.tensor(actions, device=self.device, dtype=torch.float32)
        rews = torch.tensor(r, device=self.device, dtype=torch.float32)
        done = torch.tensor(is_done.astype('float32'), device=self.device, dtype=torch.float32)

        # Prediction Q(s,𝜇(s)), Q(s,a), Q‾(s',𝜇‾(s'))
        q_pi = self.qf(obs1, self.policy(obs1))
        q = self.qf(obs1, acts).squeeze(1)
        q_pi_target = self.qf_target(obs2, self.policy_target(obs2)).squeeze(1)

        # Target for Q regression
        q_backup = rews + self.gamma * (1 - done) * q_pi_target
        q_backup.to(self.device)

        # DDPG losses
        policy_loss = -q_pi.mean()
        qf_loss = F.mse_loss(q, q_backup.detach())

        # Update policy network parameter
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip_policy)
        self.policy_optimizer.step()

        # Update Q-function network parameter
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        # nn.utils.clip_grad_norm_(self.qf.parameters(), self.gradient_clip_qf)
        self.qf_optimizer.step()

        # Polyak averaging for target parameter
        soft_target_update(self.policy, self.policy_target)
        soft_target_update(self.qf, self.qf_target)

        # Save losses
        self.policy_losses.append(policy_loss.item())
        self.qf_losses.append(qf_loss.item())

    def get_learn_freq(self):
        if self.replay_buffer.buffer_len() >= self.replay_buffer.get_maxsize():
            return 32
        return 32

    def play_episode(self, initial_state, enviroment, episode_timeout, n_steps, global_iteration, episode):
        s = initial_state
        total_reward = 0

        self.phase_platform = np.random.uniform(-np.pi, np.pi, size=1)[0]
        self.phase_wheel = np.random.uniform(-np.pi, np.pi, size=1)[0]
        self.sigma, self.amp, self.omega = np.random.randn(3)
        self.sigma = abs(self.sigma)
        self.alpha = np.random.uniform(0.3, 0.9, size=1)[0]

        init_action = self.get_action([s])
        self.prev_action_platform = init_action[0][0]
        self.prev_action_wheel = init_action[0][1]

        for i in range(n_steps):
            global_iteration += 1
            action = self.sample_actions(s, enviroment.physics.data.time, i)

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

            self.replay_buffer.add(s, action, timestep.reward, state, is_done)  # agent add to cash

            if global_iteration > self.batch_size and global_iteration % self.get_learn_freq() == 0:
                self.train_model()

            if is_done:
                _enviroment = enviroment.reset()
                break

            s = state

        if len(self.policy_losses) > 0 and len(self.qf_losses) > 0:
            self.writer.add_scalar("average loss actor", round(np.mean(self.policy_losses), 5), episode)
            self.writer.add_scalar("average loss critic", round(np.mean(self.qf_losses), 5), episode)
            self.writer.add_scalar("loss actor", self.policy_losses[-1], episode)
            self.writer.add_scalar("loss critic", self.qf_losses[-1], episode)

        return total_reward, global_iteration

    def run_eval_mode(self, enviroment, episode_timeout, s, t_max):
        reward = 0.0
        for _ in range(t_max):
            action = self.policy(torch.tensor([s], dtype=torch.float32, device=self.device)).detach().cpu()
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
        self.policy.to_eval_mode()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self.policy(state).detach().cpu().numpy()


def identity(x):
    """Return input without any change."""
    return x


def soft_target_update(main, target, tau=0.008):  # tau = 0.005
    for main_param, target_param in zip(main.parameters(), target.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0-tau) * target_param.data)


class Actor(nn.Module):
    def __init__(self, input_dim, device):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.device = device

        self.base = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU()
        ).to(self.device)

        self.platform_out = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024, 1),
            PlatformTanh()
        ).to(self.device)

        self.wheel_out = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024, 1),
            WheelSigmoid()
        ).to(self.device)

    def forward(self, state):
        out_base = self.base(state)

        platform_out = self.platform_out(out_base)
        wheel_out = self.wheel_out(out_base)

        return torch.cat([platform_out, wheel_out], dim=-1)

    def to_eval_mode(self):
        self.base.eval()
        self.platform_out.eval()
        self.wheel_out.eval()

    def to_train_mode(self):
        self.base.train()
        self.platform_out.train()
        self.wheel_out.train()


class Critic(nn.Module):
    def __init__(self, input_dim, device):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.device = device

        self.base = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.LeakyReLU(),

            nn.Linear(2048, 2048),
            nn.LeakyReLU(),

            nn.Linear(2048, 2048),
            nn.LeakyReLU(),

            nn.Linear(2048, 1)
        ).to(self.device)

    def forward(self, state, action):
        q = torch.cat([state, action], dim=-1)
        return self.base(q)


class PlatformTanh(nn.Tanh):
    def forward(self, input: Tensor) -> Tensor:
        return 0.97 * torch.tanh(input)


class WheelSigmoid(nn.Sigmoid):
    def forward(self, input: Tensor) -> Tensor:
        return 0.4 * torch.sigmoid(input) + 0.1
