import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dm_control.rl.control import PhysicsError
from dm_env import StepType
from scipy.signal import fftconvolve, gaussian
from tqdm import trange

from agent.dqn import DeepQLearningAgent
from replay_buffer import ReplayBuffer
from robot.enviroment import make_env, trajectory


def play_and_record(initial_state, agent, _enviroment, cash, n_steps=1):
    s = initial_state
    sum_rewards = 0

    for iteration in range(n_steps):
        qvalues = agent.get_qvalues([s])

        action_idx = agent.sample_actions(qvalues)[0]
        action = agent.index_to_pair[action_idx]

        try:
            _time_step = _enviroment.step(action)
        except PhysicsError:
            print("поломка в физике, метод play_and_record")
            _enviroment.reset()
            cash.add(s, action_idx, -1, s, True)
            break

        state = _time_step.observation
        is_done = _time_step.step_type == StepType.LAST or _enviroment.physics.data.time > 50

        sum_rewards += _time_step.reward

        cash.add(s, action_idx, _time_step.reward, state, is_done)

        if is_done:
            _enviroment = _enviroment.reset()
            break

        s = _time_step.observation

    return sum_rewards, s


def evaluate(_enviroment, _agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    time_step = _enviroment.reset()
    for _ in range(n_games):
        s = time_step.observation
        reward = 0.0
        for _ in range(t_max):
            qvalues = _agent.get_qvalues([s])
            action = _agent.index_to_pair[qvalues.argmax(axis=-1)[0]] if greedy else _agent.sample_actions(qvalues)[0]
            try:
                time_step = _enviroment.step(action)

                if time_step.reward is None:
                    break
                reward += time_step.reward
                if time_step.step_type == StepType.LAST:
                    break
            except PhysicsError:
                print("поломка в физике, метод evaluate")
                break
        rewards.append(reward)
    return np.mean(rewards)


def compute_td_loss(states, action_indexes, rewards, next_states, is_done,
                    agent, target_network, gamma=0.99, check_shapes=False, device=None):
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float32)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float32,
    )
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)  # shape: [batch_size, n_actions]

    # compute q-values for all actions in next states
    # with torch.no_grad():
    predicted_next_qvalues = target_network(next_states)  # shape: [batch_size, n_actions]

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(action_indexes)), action_indexes]  # shape: [batch_size]

    # compute V*(next_states) using predicted next q-values
    ##############################################
    next_state_values = predicted_next_qvalues.max(axis=-1)[0]
    ##############################################

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    ###############################################
    target_qvalues_for_actions = rewards + gamma * next_state_values * is_not_done
    ##############################################

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

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


env = make_env()

# action_spec = env.action_spec()
cash = ReplayBuffer(2_000_000)
state_dim = 2  # 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

timesteps_per_epoch = 1000
batch_size = 512
total_steps = 1 * 10 ** 3  # 10 ** 4
decay_steps = 1 * 10 ** 3  # 10 ** 4

agent = DeepQLearningAgent(state_dim, batch_size=batch_size, epsilon=1).to(device)
target_network = DeepQLearningAgent(state_dim, batch_size=batch_size, epsilon=1).to(device)
target_network.load_state_dict(agent.state_dict())

optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

loss_freq = 50
refresh_target_network_freq = 500
eval_freq = 500

max_grad_norm = 5000

mean_rw_history = []
td_loss_history = []
grad_norm_history = []
initial_state_v_history = []
step = 0

init_epsilon = 1
final_epsilon = 0.05

state = env.reset().observation

with trange(step, total_steps + 1) as progress_bar:
    for step in progress_bar:

        agent.epsilon = linear_decay(init_epsilon, final_epsilon, step, decay_steps)

        # play
        _, state = play_and_record(state, agent, env, cash, timesteps_per_epoch)

        s, a, r, next_s, is_done = cash.sample(batch_size)

        loss = compute_td_loss(s, a, r, next_s, is_done, agent, target_network, device=device)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if step % loss_freq == 0:
            td_loss_history.append(loss.data.cpu().item())
            grad_norm_history.append(grad_norm.data.cpu().item())

        if step % refresh_target_network_freq == 0:
            print("copy parameters from agent to target")
            target_network.load_state_dict(agent.state_dict())
            state = env.reset().observation
            print("init state = ", state)

        if step % eval_freq == 0:
            print("buffer size = %i, epsilon = %.5f" %
                  (len(cash), agent.epsilon))

            mean_rw_history.append(evaluate(
                make_env(), agent, n_games=2, greedy=True, t_max=1000)
            )
            initial_state_q_values = agent.get_qvalues(
                make_env().reset().observation
            )
            initial_state_v_history.append(np.max(initial_state_q_values))

print("buffer size = %i, epsilon = %.5f" %
      (len(cash), agent.epsilon))

final_score = evaluate(
    make_env(),
    agent, n_games=30, greedy=True, t_max=1000
)
print('final score:', final_score)

fig, rl = plt.subplots(2, 2)

rl[0][0].set_title("Mean reward per episode")
rl[0][0].plot(mean_rw_history)
rl[0][0].grid()

rl[1][0].set_title("TD loss history (smoothened)")
rl[1][0].plot(smoothen(td_loss_history))
rl[1][0].grid()

rl[0][1].set_title("Initial state V")
rl[0][1].plot(initial_state_v_history)
rl[0][1].grid()

rl[1][1].set_title("Grad norm history (smoothened)")
rl[1][1].plot(smoothen(grad_norm_history))
rl[1][1].grid()

# plt.show()

frames = []
time_step = env.reset()
pos = np.array([[0, 0, 0]])
times = []

env = make_env()
time_step = env.reset()
prev_time = env.physics.data.time
while env.physics.data.time < 40:
    qvalues = agent.get_qvalues([time_step.observation])
    action = agent.index_to_pair[qvalues.argmax(axis=-1)[0]]  # if greedy else agent.sample_actions(qvalues)[0]
    try:
        time_step = env.step(action=action)
    except PhysicsError:
        print("physicx error  time = ", prev_time)
        break

    reward = time_step.reward
    if reward is None:
        print("reward is None ! time = ", prev_time)
        break

    prev_time = env.physics.data.time
    print("reward = ", reward, "\n")

    # frame = env.physics.render(camera_id=0, width=300, height=300)

    observation = np.concatenate([time_step.observation[0:2], [0]], axis=0)

    pos = np.append(pos, [observation], axis=0)
    times.append(env.physics.data.time)
    # if env.physics.data.time > 1:
    #     frames.append(frame)

fig1, ax = plt.subplots(3, 1)
ax[0].plot(times, pos[:, 0][1:])
ax[1].plot(times, pos[:, 1][1:])
ax[2].plot(times, pos[:, 2][1:])

traj = plt.figure().add_subplot()
traj.plot(pos[:, 1][1:], pos[:, 1][1:], label="trajectory")
traj.plot(trajectory()[0], trajectory()[1], label="desired_trajectory")
traj.set_xlabel('x')
traj.set_ylabel('y')

plt.show()
