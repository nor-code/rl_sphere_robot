import random
from collections import deque, namedtuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import fftconvolve, gaussian
from tqdm import trange

from replay_buffer import ReplayBuffer

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        assert len(state_shape) == 1
        state_dim = state_shape[0]

        # state_shape =  (4,)
        # n_action =  2

        hidden_size = 150
        self._nn = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.ReLU()
        )

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch states, shape = [batch_size, *state_dim=4]
        """
        # Use your network to compute qvalues for given state

        ##############################################
        qvalues = self._nn(state_t)
        ##############################################

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert (
                len(qvalues.shape) == 2 and
                qvalues.shape[0] == state_t.shape[0] and
                qvalues.shape[1] == self.n_actions
        )

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1 - epsilon, epsilon]
        )
        return np.where(should_explore, random_actions, best_actions)


ENV_NAME = 'CartPole-v1'


def make_env(seed=None):
    # some envs are wrapped with a time limit wrapper by default
    env = gym.make(ENV_NAME).unwrapped
    if seed is not None:
        env.seed(seed)
    return env


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n_steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays

    hint: use agent.sample.actions
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    # <YOUR CODE>
    for _ in range(n_steps):
        qvalues = agent.get_qvalues([s])

        action = agent.sample_actions(qvalues)[0]
        # action = action.argmax(axis=-1)[0]
        state, reward, done, _, _ = env.step(action)
        sum_rewards += reward

        exp_replay.add(s, action, reward, state, done)

        if done:
            state = env.reset()

        s = state

    return sum_rewards, s


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network, gamma=0.99, check_shapes=False, device=device):
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float32)  # shape: [batch_size, *state_shape]
    actions = torch.tensor(actions, device=device, dtype=torch.int64)  # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float32,
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)  # shape: [batch_size, n_actions]

    # compute q-values for all actions in next states
    # with torch.no_grad():
    predicted_next_qvalues = target_network(next_states)  # shape: [batch_size, n_actions]

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]  # shape: [batch_size]

    # compute V*(next_states) using predicted next q-values
    ##############################################
    next_state_values = predicted_next_qvalues.max(axis=-1)[0]
    ##############################################

    assert next_state_values.dim() == 1 and next_state_values.shape[0] == states.shape[0], \
        "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    ###############################################
    target_qvalues_for_actions = rewards + gamma * next_state_values * is_not_done
    ##############################################

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim() == 2, \
            "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim() == 1, \
            "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim() == 1, \
            "there's something wrong with target q-values, they must be a vector"

    return loss


def linear_decay(init_val, final_val, cur_step, total_steps):
    if cur_step >= total_steps:
        return final_val
    return (init_val * (total_steps - cur_step) +
            final_val * cur_step) / total_steps


def smoothen(values):
    kernel = gaussian(100, std=100)
    # kernel = np.concatenate([np.arange(100), np.arange(99, -1, -1)])
    kernel = kernel / np.sum(kernel)
    return fftconvolve(values, kernel, 'valid')


# your favourite random seed
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env = make_env(seed)
state_dim = env.observation_space.shape
n_actions = env.action_space.n
state = env.reset()

agent = DQNAgent(state_dim, n_actions, epsilon=1).to(device)
target_network = DQNAgent(state_dim, n_actions, epsilon=1).to(device)
target_network.load_state_dict(agent.state_dict())

timesteps_per_epoch = 1
batch_size = 32
total_steps = 8 * 10 ** 4
decay_steps = 1 * 10 ** 4

optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

init_epsilon = 1
final_epsilon = 0.1

loss_freq = 20
refresh_target_network_freq = 1000
eval_freq = 1000

max_grad_norm = 5000

mean_rw_history = []
td_loss_history = []
grad_norm_history = []
initial_state_v_history = []
step = 0

exp_replay = ReplayBuffer(10000)

np.linspace(-0.5, 0.3)

state = env.reset()
with trange(step, total_steps + 1) as progress_bar:
    for step in progress_bar:

        agent.epsilon = linear_decay(init_epsilon, final_epsilon, step, decay_steps)

        # play
        _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

        # train
        # <YOUR CODE: sample batch_size of data from experience replay>
        s, a, r, next_s, is_done = exp_replay.sample(batch_size)
        # loss = <YOUR CODE: compute TD loss>
        loss = compute_td_loss(s, a, r, next_s, is_done, agent, target_network)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if step % loss_freq == 0:
            td_loss_history.append(loss.data.cpu().item())
            grad_norm_history.append(grad_norm.data.cpu().item())

        if step % refresh_target_network_freq == 0:
            # Load agent weights into target_network
            # <YOUR CODE>
            target_network.load_state_dict(agent.state_dict())
            state = env.reset()

        if step % eval_freq == 0:
            mean_rw_history.append(evaluate(
                make_env(seed=step), agent, n_games=3, greedy=True, t_max=1000)
            )
            initial_state_q_values = agent.get_qvalues(
                [make_env(seed=step).reset()]
            )
            initial_state_v_history.append(np.max(initial_state_q_values))

# clear_output(True)
print("buffer size = %i, epsilon = %.5f" %
      (len(exp_replay), agent.epsilon))

final_score = evaluate(
  make_env(),
  agent, n_games=30, greedy=True, t_max=1000
)
print('final score:', final_score)
if final_score > 300:
  print('Well done')
else:
  print('not good enough for DQN')

plt.figure(figsize=[16, 9])

plt.subplot(2, 2, 1)
plt.title("Mean reward per episode")
plt.plot(mean_rw_history)
plt.grid()

assert not np.isnan(td_loss_history[-1])
plt.subplot(2, 2, 2)
plt.title("TD loss history (smoothened)")
plt.plot(smoothen(td_loss_history))
plt.grid()

plt.subplot(2, 2, 3)
plt.title("Initial state V")
plt.plot(initial_state_v_history)
plt.grid()

plt.subplot(2, 2, 4)
plt.title("Grad norm history (smoothened)")
plt.plot(smoothen(grad_norm_history))
plt.grid()

plt.show()