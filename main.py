import argparse

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
from dm_control.rl.control import PhysicsError
from dm_env import StepType
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from tqdm import trange

from agent.dqn import DeepQLearningAgent
from replay_buffer import ReplayBuffer
from robot.enviroment import make_env
# from IPython.display import clear_output
from utils.utils import build_trajectory

iteration = 0

x_max = 1.15
y_max = 2.15
v_x_max = 0.250
v_y_max = 0.190


def get_learn_freq(_cash):
    if _cash.buffer_len() >= _cash.get_maxsize():
        return 256
    return 512


def play_and_record(initial_state, _agent, _enviroment, _cash, episode_timeout, n_steps=1000):
    global iteration
    s = initial_state
    s[0] = s[0] / x_max
    s[1] = s[1] / y_max
    s[2] = s[2] / v_x_max
    s[3] = s[3] / v_y_max

    sum_rewards = 0
    loss = None

    for i in range(n_steps):
        iteration += 1

        qvalues = _agent.get_qvalues([s])
        action_idx = _agent.sample_actions(qvalues)[0]
        action = _agent.index_to_pair[action_idx]

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

        state[0] = state[0] / x_max
        state[1] = state[1] / y_max
        state[2] = state[2] / v_x_max
        state[3] = state[3] / v_y_max

        cash.add(s, action_idx, _time_step.reward, state, is_done)

        if iteration > _agent.batch_size and iteration % get_learn_freq(_cash) == 0:
            loss = _agent.train_network(_cash)

        if is_done:
            _enviroment = _enviroment.reset()
            s = _enviroment.observation
            break

        s = state

    return sum_rewards, s, loss


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


def linear_decay(init_val, final_val, cur_step, total_steps):
    if cur_step >= total_steps:
        return final_val
    return (init_val * (total_steps - cur_step) +
            final_val * cur_step) / total_steps


def get_envs(size):
    env_list_ = []
    dim = 0
    for i in [0, size // 2]:
        env_i, dim = make_env(
            episode_timeout=timeout, type_task=args.type_task, trajectory=args.trajectory, begin_index_=i
        )
        env_list_.append(env_i)
    return env_list_, dim


def get_size():
    if args.trajectory == 'circle':
        return 30
    elif args.trajectory == 'curve':
        return 30


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='DQN Spherical Robot')
parser.add_argument('--simu_number', type=int, default=1, help='number of simulation')
parser.add_argument('--type_task', type=int, default=3, help='type of task. now available 1 and 2')
parser.add_argument('--algo', type=str, default='ddqn', help='type agent, dqn or ddqn available')
parser.add_argument('--trajectory', type=str, default='circle', help='trajectory for agent')
args = parser.parse_args()

number = args.simu_number

writer = SummaryWriter()
timeout = 50
env_list, state_dim = get_envs(get_size())
cash = ReplayBuffer(1_500_000)

timesteps_per_epoch = 2000
batch_size = 4 * 2048
total_steps = 25 * 10 ** 4  # 40 * 10 ** 4  # 10 ** 4
decay_steps = 25 * 10 ** 4  # 40 * 10 ** 4 name1 # 10 ** 4

agent = DeepQLearningAgent(state_dim,
                           batch_size=batch_size,
                           epsilon=1,
                           gamma=0.99,
                           device=device,
                           algo=args.algo)

# loss_freq = 250  # 300 # 300
refresh_target_network_freq = 800  # 350 # 400
eval_freq = 200  # 300  # 400statestate = env.reset().observation = env.reset().observation
change_env_freq = 2

mean_rw_history = []
td_loss_history = []
grad_norm_history = []
initial_state_v_history = []
step = 0

init_epsilon = 1
final_epsilon = 0.15

env = np.random.choice(env_list, size=1)[0]
state = env.reset().observation

with trange(step, total_steps + 1) as progress_bar:
    for step in progress_bar:

        agent.epsilon = linear_decay(init_epsilon, final_epsilon, step, decay_steps)

        # play 1 episode and push to replay buffer
        _, state, loss = play_and_record(state, agent, env, cash, timeout, timesteps_per_epoch)

        env = env_list[step % 2]
        state = env.reset().observation

        if loss is not None:
            writer.add_scalar("TD_loss #" + str(number), loss.data.cpu().item(), step)
            # grad_norm_history.append(grad_norm.data.cpu().item())

        if step % refresh_target_network_freq == 0 and iteration > batch_size:
            print("copy parameters from agent to target")
            agent.update_target_network()
            state = env.reset().observation

        if step % eval_freq == 0:
            plot_buf, fig = build_trajectory(
                agent=agent, enviroment=env, timeout=timeout, trajectory_type=args.trajectory, type_task=args.type_task
            )

            pillow_img = PIL.Image.open(plot_buf)
            tensor_img = ToTensor()(pillow_img)
            writer.add_image("trajectory #" + str(number), tensor_img, step / eval_freq)
            plt.close(fig)

            env.reset()
            mean_reward = evaluate(
                env, agent, n_games=3, greedy=True, t_max=1000
            )
            writer.add_scalar("Mean_reward_history 3 episode #" + str(number), mean_reward, step)

            initial_state_q_values = agent.get_qvalues(env.reset().observation)
            writer.add_scalar("size of replay buffer # " + str(number), cash.buffer_len(), step)

writer.flush()

PATH = './models/'
FILE = PATH + "name" + str(number) + ".pt"

torch.save(agent.q_network.state_dict(), FILE)

# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
