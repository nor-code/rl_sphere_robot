import argparse

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
from dm_control.rl.control import PhysicsError
from dm_env import StepType
from scipy.signal import fftconvolve, gaussian
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from tqdm import trange

from agent.dqn import DeepQLearningAgent
from replay_buffer import ReplayBuffer
from robot.enviroment import make_env, trajectory
# from IPython.display import clear_output
from utils.utils import build_trajectory


def play_and_record(initial_state, agent, _enviroment, cash, episode_timeout, n_steps=1000):
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
            cash.add(s, action_idx, -1, s, True)
            _enviroment = _enviroment.reset()
            s = _enviroment.observation
            break

        state = _time_step.observation
        is_done = _time_step.step_type == StepType.LAST or _enviroment.physics.data.time > episode_timeout

        if _time_step.reward is None:
            break

        sum_rewards += _time_step.reward
        cash.add(s, action_idx, _time_step.reward, state, is_done)

        if is_done:
            _enviroment = _enviroment.reset()
            s = _enviroment.observation
            break

        s = state

    return sum_rewards, s


def evaluate(_enviroment, _agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        time_step = _enviroment.reset()
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='DQN Spherical Robot')
parser.add_argument('--simu_number', type=int, default=0, help='number of simulation')
parser.add_argument('--type_task', type=int, default=1, help='type of task. now available 1 and 2')
# parser.add_argument('--agent', type=str, default='dqn', help='type agent, dqn or ddqn available')
# parser.add_argument('--trajectory', type=str, default='circle', help='desired trajectory')

args = parser.parse_args()

number = args.simu_number

writer = SummaryWriter()
timeout = 40
env, state_dim = make_env(episode_timeout=timeout, type_task=args.type_task)
cash = ReplayBuffer(20_000_000)

timesteps_per_epoch = 1000
batch_size = 3 * 2048
total_steps = 40 * 10 ** 4  # 10 ** 4
decay_steps = 40 * 10 ** 4  # 10 ** 4
agent = DeepQLearningAgent(state_dim, batch_size=batch_size, epsilon=1).to(device)
target_network = DeepQLearningAgent(state_dim, batch_size=batch_size, epsilon=1).to(device)
target_network.load_state_dict(agent.state_dict())

optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

loss_freq = 300                   # 300
refresh_target_network_freq = 500  # 400
eval_freq = 300                    # 400

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
        _, state = play_and_record(state, agent, env, cash, timeout, timesteps_per_epoch)

        s, a, r, next_s, is_done = cash.sample(batch_size)

        loss = compute_td_loss(s, a, r, next_s, is_done, agent, target_network, device=device)

        loss.backward()
        # grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if step % loss_freq == 0:
            writer.add_scalar("TD_loss #" + str(number), loss.data.cpu().item(), step)
            # grad_norm_history.append(grad_norm.data.cpu().item())

        if step % refresh_target_network_freq == 0:
            print("copy parameters from agent to target")
            target_network.load_state_dict(agent.state_dict())
            state = env.reset().observation

        if step % eval_freq == 0:
            plot_buf, fig = build_trajectory(
                agent=agent, enviroment=env, timeout=timeout, trajectory_func=trajectory, type_task=args.type_task
            )

            pillow_img = PIL.Image.open(plot_buf)
            tensor_img = ToTensor()(pillow_img)
            writer.add_image("trajectory #" + str(number), tensor_img, step / eval_freq)
            plt.close(fig)

            mean_reward = evaluate(make_env(episode_timeout=timeout, type_task=args.type_task)[0], agent, n_games=2, greedy=True, t_max=1000)
            writer.add_scalar("Mean_reward_history #" + str(number), mean_reward, step)

            initial_state_q_values = agent.get_qvalues(
                make_env(episode_timeout=timeout, type_task=args.type_task)[0].reset().observation
            )
            writer.add_scalar("init_state_Q_value #" + str(number), np.max(initial_state_q_values), step)

writer.flush()

# print("buffer size = %i, epsilon = %.5f" % (len(cash), agent.epsilon))
#
# final_score = evaluate(
#     make_env(episode_timeout=timeout),
#     agent, n_games=30, greedy=True, t_max=1000
# )
# print('final score:', final_score)


PATH = './models/'
FILE = PATH + "name" + str(number) + ".pt"

torch.save(agent.state_dict(), FILE)

# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
