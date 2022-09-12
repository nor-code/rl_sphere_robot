import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class Agent(object):
    """An implementation of the Deep Deterministic Policy Gradient (DDPG) agent."""

    def __init__(self,
                 env,
                 args,
                 device,
                 obs_dim,
                 act_dim,
                 act_limit,
                 steps=0,
                 expl_before=2000,
                 train_after=1000,
                 gamma=0.99,
                 act_noise=0.3,
                 hidden_sizes=(1024, 2048, 1024),
                 batch_size=1,
                 gradient_clip_policy=0.5,
                 gradient_clip_qf=1.0,
                 eval_mode=False,
                 policy_losses=list(),
                 qf_losses=list(),
                 logger=dict(),
                 ):

        self.env = env
        self.args = args
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.steps = steps
        self.expl_before = expl_before
        self.train_after = train_after
        self.gamma = gamma
        self.act_noise = act_noise
        self.hidden_sizes = hidden_sizes

        self.batch_size = batch_size

        self.gradient_clip_policy = gradient_clip_policy
        self.gradient_clip_qf = gradient_clip_qf
        self.eval_mode = eval_mode
        self.policy_losses = policy_losses
        self.qf_losses = qf_losses
        self.logger = logger

        # Main network
        self.policy = MLP(self.obs_dim, self.act_dim, self.act_limit,
                          hidden_sizes=self.hidden_sizes,
                          output_activation=torch.tanh,
                          use_actor=True).to(self.device)
        self.qf = FlattenMLP(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(self.device)
        # Target network Actor
        self.policy_target = MLP(self.obs_dim, self.act_dim, self.act_limit,
                                 hidden_sizes=self.hidden_sizes,
                                 output_activation=torch.tanh,
                                 use_actor=True).to(self.device)
        self.qf_target = FlattenMLP(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(self.device)

        # Initialize target parameters to match main parameters
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.qf.load_state_dict(self.qf_target.state_dict())

        # Create optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.qf_optimizer = torch.optim.Adam(self.qf.parameters(), lr=1e-3)


    def select_action(self, obs):
        action = self.policy(obs).detach().cpu().numpy()
        action += self.act_noise * np.random.randn(self.act_dim)
        return np.clip(action, -self.act_limit, self.act_limit)

    def train_model(self, replay_buffer):
        s, action_index, r, next_s, is_done = replay_buffer.sample(self.batch_size)

        obs1 = torch.tensor(s, device=self.device, dtype=torch.float32)
        obs2 = torch.tensor(next_s, device=self.device, dtype=torch.float32)
        acts = torch.tensor(action_index.reshape(-1, 1), device=self.device, dtype=torch.long)
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
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip_policy)
        self.policy_optimizer.step()

        # Update Q-function network parameter
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        nn.utils.clip_grad_norm_(self.qf.parameters(), self.gradient_clip_qf)
        self.qf_optimizer.step()

        # Polyak averaging for target parameter
        soft_target_update(self.policy, self.policy_target)
        soft_target_update(self.qf, self.qf_target)

        # Save losses
        self.policy_losses.append(policy_loss.item())
        self.qf_losses.append(qf_loss.item())

    def run(self, max_step):
        step_number = 0
        total_reward = 0.

        obs = self.env.reset()
        done = False

        # Keep interacting until agent reaches a terminal state.
        while not (done or step_number == max_step):
            if self.args.render:
                self.env.render()

            if self.eval_mode:
                action = self.policy(torch.Tensor(obs).to(self.device))
                action = action.detach().cpu().numpy()
                next_obs, reward, done, _ = self.env.step(action)
            else:
                self.steps += 1

                # Until expl_before have elapsed, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy.
                if self.steps > self.expl_before:
                    action = self.select_action(torch.Tensor(obs).to(self.device))
                else:
                    action = self.env.action_space.sample()

                # Collect experience (s, a, r, s') using some policy
                next_obs, reward, done, _ = self.env.step(action)

                # Add experience to replay buffer
                self.replay_buffer.add(obs, action, reward, next_obs, done)

                # Start training when the number of experience is greater than train_after
                if self.steps > self.train_after:
                    self.train_model()

            total_reward += reward
            step_number += 1
            obs = next_obs

        # Save logs
        self.logger['LossPi'] = round(np.mean(self.policy_losses), 5)
        self.logger['LossQ'] = round(np.mean(self.qf_losses), 5)
        return step_number, total_reward


def identity(x):
    """Return input without any change."""
    return x


"""
DQN, DDQN, A2C critic, VPG critic, TRPO critic, PPO critic, DDPG actor, TD3 actor
"""


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 output_limit=1.0,
                 hidden_sizes=(64, 64),
                 activation=F.relu,
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
        x = self.output_activation(self.output_layer(x))
        # If the network is used as actor network, make sure output is in correct range
        x = x * self.output_limit if self.use_actor else x
        return x


class FlattenMLP(MLP):
    def forward(self, x, a):
        q = torch.cat([x,a], dim=-1)
        return super(FlattenMLP, self).forward(q)