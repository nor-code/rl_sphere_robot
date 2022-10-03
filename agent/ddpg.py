import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from dm_control.rl.control import PhysicsError
from dm_env import StepType


class DeepDeterministicPolicyGradient(object):
    """An implementation of the Deep Deterministic Policy Gradient (DDPG) agent."""

    def __init__(self,
                 obs_dim,
                 device,
                 act_dim,
                 act_limit,
                 replay_buffer,
                 writer,
                 gamma=0.99,
                 act_noise=0.11,
                 hidden_sizes_actor=(1024, 2048, 1024),
                 hidden_sizes_critic=(1024, 2048, 1024),
                 batch_size=1,
                 gradient_clip_policy=0.5,  # 0.5
                 gradient_clip_qf=1.0,  # 1.0
                 policy_losses=list(),
                 qf_losses=list()):

        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.replay_buffer = replay_buffer
        self.epsilon = 1
        self.gamma = gamma
        self.act_noise = act_noise
        self.hidden_sizes_critic = hidden_sizes_critic
        self.hidden_sizes_actor = hidden_sizes_actor
        self.writer = writer
        self.batch_size = batch_size

        self.gradient_clip_policy = gradient_clip_policy
        self.gradient_clip_qf = gradient_clip_qf
        self.policy_losses = policy_losses
        self.qf_losses = qf_losses

        # Main network Actor
        self.policy = MLP(self.obs_dim, self.act_dim, self.act_limit,
                          hidden_sizes=self.hidden_sizes_actor,
                          use_actor=True).to(self.device)
        # Critic
        self.qf = FlattenMLP(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes_critic).to(self.device)

        # Target network Actor
        self.policy_target = MLP(self.obs_dim, self.act_dim, self.act_limit,
                                 hidden_sizes=self.hidden_sizes_actor,
                                 use_actor=True).to(self.device)
        self.qf_target = FlattenMLP(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes_critic).to(self.device)

        # Initialize target parameters to match main parameters
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.qf.load_state_dict(self.qf_target.state_dict())

        # Create optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3, weight_decay=1e-4)
        self.qf_optimizer = torch.optim.Adam(self.qf.parameters(), lr=1e-3, weight_decay=1e-4)

    def sample_actions(self, state):
        if self.epsilon > 0:
            return self.select_action(state)  # with noise
        else:
            return self.get_action(state)  # without noise

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = self.policy(state).detach().cpu().numpy()
        action += self.act_noise * np.random.randn(self.act_dim)

        for key in self.act_limit:
            interval = self.act_limit[key]
            action[key] = np.clip(action[key], interval[0], interval[1])

        return action

    def train_model(self):
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
        for i in range(n_steps):
            global_iteration += 1
            action = self.sample_actions(s)

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
            action = self.policy(torch.tensor(s, dtype=torch.float32, device=self.device)).detach().cpu()
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
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self.policy(state).detach().cpu().numpy()


def identity(x):
    """Return input without any change."""
    return x


def soft_target_update(main, target, tau=0.005):  # tau = 0.005
    for main_param, target_param in zip(main.parameters(), target.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0-tau) * target_param.data)


"""
DQN, DDQN, A2C critic, VPG critic, TRPO critic, PPO critic, DDPG actor, TD3 actor
"""


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 output_limit=1.0,
                 hidden_sizes=(64, 64),
                 activation=F.leaky_relu,
                 output_activation=identity,
                 use_output_layer=True,
                 use_actor=False,
                 ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_limit = output_limit
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer
        self.use_actor = use_actor

        # Set hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc)

        # Set output layers
        if self.use_output_layer:
            self.output_layer = nn.Linear(in_size, self.output_size)
        else:
            self.output_layer = identity

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        if self.use_actor:
            x = self.output_layer(x)
            for key in self.output_limit:
                interval = self.output_limit[key]
                x[key] = ((interval[1] - interval[0])/2) * torch.tanh(x[key]) + (interval[1] + interval[0])/2
        else:
            x = self.output_activation(self.output_layer(x))
        return x


class FlattenMLP(MLP):
    def forward(self, x, a):
        q = torch.cat([x, a], dim=-1)
        return super(FlattenMLP, self).forward(q)
