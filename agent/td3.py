import numpy as np
import torch
import torch.nn.functional as F
from dm_control.rl.control import PhysicsError
from dm_env import StepType

from agent.ddpg import Actor, Critic, soft_target_update


class TwinDelayedAgent(object):
    """Twin Delayed DDPG (TD3) agent"""

    def __init__(self,
                 obs_dim,
                 device,
                 act_dim,
                 replay_buffer,
                 writer,
                 gamma=0.99,
                 act_noise=0.1,
                 batch_size=1,
                 target_noise=0.2,
                 noise_clip=0.5,
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
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.gradient_clip_policy = gradient_clip_policy
        self.gradient_clip_qf = gradient_clip_qf
        self.policy_losses = policy_losses
        self.qf_losses = qf_losses

        # Main network
        self.policy = Actor(self.obs_dim, self.device).to(self.device)
        self.qf1 = Critic(self.obs_dim + self.act_dim, self.device)
        self.qf2 = Critic(self.obs_dim + self.act_dim, self.device)
        # Target network
        self.policy_target = Actor(self.obs_dim, self.device).to(self.device)
        self.qf1_target = Critic(self.obs_dim + self.act_dim, self.device)
        self.qf2_target = Critic(self.obs_dim + self.act_dim, self.device)

        # Initialize target parameters to match main parameters
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.qf1.load_state_dict(self.qf1_target.state_dict())
        self.qf2.load_state_dict(self.qf2_target.state_dict())

        # Concat the Q-network parameters to use one optim
        self.qf_parameters = list(self.qf1.parameters()) + list(self.qf2.parameters())
        # Create optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4, weight_decay=1e-4)
        self.qf_optimizer = torch.optim.Adam(self.qf_parameters, lr=3e-4, weight_decay=1e-4)

        # for noize action per one episode
        self.phase_platform = np.random.uniform(-np.pi, np.pi, size=1)
        self.phase_wheel = np.random.uniform(-np.pi, np.pi, size=1)
        self.sigma, self.amp, self.omega = np.random.randn(3)
        self.prev_action_platform = 0
        self.prev_action_wheel = 0
        self.alpha = 0 # для сглаживающего фильтра

    def sample_actions(self, state, t, i):
        rand = np.random.rand()

        if rand <= self.epsilon:
            self.policy.to_eval_mode()
            action = self.get_action([state])

            platform, wheel = action[0][0], action[0][1]

            mu_p = 0.001 * self.amp * np.sin(self.omega * t + self.phase_platform)
            mu_w = 0.0001 * self.amp * np.sin(self.omega * t + self.phase_wheel)

            platform += (self.sigma * np.random.randn(1) + mu_p)
            wheel += (self.sigma * np.random.randn(1) + mu_w)

            platform = np.clip(platform, -0.9985, 0.9985)
            wheel = np.clip(wheel, 0.26, 0.28)

            platform = self.alpha * platform + (1 - self.alpha) * self.prev_action_platform
            wheel = 0.15 * wheel + (1 - 0.15) * self.prev_action_wheel

            platform = np.clip(platform, -0.9985, 0.9985)
            wheel = np.clip(wheel, 0.26, 0.28)

            self.writer.add_scalar("noise_action plat", platform, i)
            self.writer.add_scalar("noise_action wheel", wheel, i)

            self.prev_action_platform = platform[0]
            self.prev_action_wheel = wheel[0]

            return [platform[0], wheel[0]]  # with noise
        else:
            action = self.get_action([state])
            self.prev_action_platform = action[0][0]
            self.prev_action_wheel = action[0][1]
            return [action[0][0], action[0][1]]

    def train_model(self):
        s, actions, r, next_s, is_done = self.replay_buffer.sample(self.batch_size)

        obs1 = torch.tensor(s, device=self.device, dtype=torch.float32)
        obs2 = torch.tensor(next_s, device=self.device, dtype=torch.float32)
        acts = torch.tensor(actions, device=self.device, dtype=torch.float32)
        rews = torch.tensor(r, device=self.device, dtype=torch.float32)
        done = torch.tensor(is_done.astype('float32'), device=self.device, dtype=torch.float32)

        # Prediction Q1(s,𝜇(s)), Q1(s,a), Q2(s,a)
        q1_pi = self.qf1(obs1, self.policy(obs1))
        q1 = self.qf1(obs1, acts).squeeze(1)
        q2 = self.qf2(obs1, acts).squeeze(1)

        # Target policy smoothing, by adding clipped noise to target actions
        pi_target = self.policy_target(obs2)
        epsilon = torch.normal(mean=0, std=self.target_noise, size=pi_target.size()).to(self.device)
        epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip).to(self.device)
        pi_target = pi_target + epsilon
        pi_target[:, 0] = torch.clamp(pi_target[:, 0], -0.9985, 0.9985)
        pi_target[:, 1] = torch.clamp(pi_target[:, 1], 0.24, 0.28)
        pi_target.to(self.device)

        # Min Double-Q: min(Q1‾(s',𝜇(s')), Q2‾(s',𝜇(s')))
        min_q_pi_target = torch.min(self.qf1_target(obs2, pi_target),
                                    self.qf2_target(obs2, pi_target)).squeeze(1).to(self.device)

        # Target for Q regression
        q_backup = rews + self.gamma * (1 - done) * min_q_pi_target
        q_backup.to(self.device)

        # TD3 losses
        policy_loss = -q1_pi.mean()
        qf1_loss = F.mse_loss(q1, q_backup.detach())
        qf2_loss = F.mse_loss(q2, q_backup.detach())
        qf_loss = qf1_loss + qf2_loss

        # Delayed policy update
        # if self.steps % self.policy_delay == 0:
        # Update policy network parameter
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Polyak averaging for target parameter
        soft_target_update(self.policy, self.policy_target)
        soft_target_update(self.qf1, self.qf1_target)
        soft_target_update(self.qf2, self.qf2_target)

        # Update two Q-network parameter
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        # Save losses
        self.policy_losses.append(policy_loss.item())
        self.qf_losses.append(qf_loss.item())

    def get_learn_freq(self):
        if self.replay_buffer.buffer_len() >= self.replay_buffer.get_maxsize():
            return 10
        return 10

    def play_episode(self, initial_state, enviroment, episode_timeout, n_steps, global_iteration, episode):
        s = initial_state
        total_reward = 0

        self.phase_platform = np.random.uniform(-np.pi, np.pi, size=1)[0]
        self.phase_wheel = np.random.uniform(-np.pi, np.pi, size=1)[0]
        self.sigma, self.amp, self.omega = np.random.randn(3)
        self.sigma = np.sqrt(abs(self.sigma))
        self.alpha = np.random.uniform(0.1, 0.9, size=1)[0]

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