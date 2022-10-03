import argparse

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from tqdm import trange

from agent.ddpg import DeepDeterministicPolicyGradient
from agent.dqn import DeepQLearningAgent
from replay_buffer import ReplayBuffer
from robot.enviroment import make_env
# from IPython.display import clear_output
from utils.utils import build_trajectory
from torchsummary import summary


iteration = 0


def play_and_record(initial_state, _agent, _enviroment, episode_timeout, episode, n_steps=1500):
    global iteration
    total_reward, new_iteration = _agent.play_episode(
        initial_state,
        _enviroment,
        episode_timeout,
        n_steps,
        iteration,
        episode
    )
    iteration = new_iteration
    return total_reward


def mean_reward_per_episode(_agent, _enviroment, episode_timeout, n_games=1, t_max=1500):
    rewards = []
    for _ in range(n_games):
        time_step = _enviroment.reset()
        s = time_step.observation
        rewards.append(_agent.run_eval_mode(_enviroment, episode_timeout, s, t_max))
    return np.mean(rewards)


def linear_decay(init_val, final_val, cur_step, total_steps):
    if cur_step >= total_steps:
        return final_val
    return (init_val * (total_steps - cur_step) +
            final_val * cur_step) / total_steps


def get_envs(size):
    env_list_ = []
    dim = 0
    for i in [0, size // 10, 2 * size // 10, (3 * size) // 10, (4 * size) // 10, size // 2, (6 * size) // 10,
              (7 * size) // 10, (8 * size) // 10, (9 * size) // 10]:
        env_i, dim = make_env(
            episode_timeout=timeout, type_task=args.type_task, trajectory=args.trajectory, begin_index_=i
        )
        env_list_.append(env_i)
    return env_list_, dim


def get_size():
    if args.trajectory == 'circle':
        return 50
    elif args.trajectory == 'curve':
        return 25


def save_model(backup_iteration, _number, name_agent):
    print("backup model param")
    PATH = './models/'

    if name_agent == 'ddqn' or name_agent == 'dqn':
        FILE = PATH + name_agent + str(_number) + "_" + str(backup_iteration) + ".pt"
        torch.save(agent.q_network.state_dict(), FILE)
    else:
        FILE_POLICY = PATH + name_agent + "_policy_" + str(_number) + "_" + str(backup_iteration) + ".pt"
        torch.save(agent.policy.state_dict(), FILE_POLICY)

        FILE_Q = PATH + name_agent + "_Q_" + str(_number) + "_" + str(backup_iteration) + ".pt"
        torch.save(agent.qf.state_dict(), FILE_Q)


np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='DQN/DDPG Spherical Robot')
parser.add_argument('--simu_number', type=int, default=1, help='number of simulation')
parser.add_argument('--type_task', type=int, default=4, help='type of task. now available 1, 2, 3, 4')
parser.add_argument('--trajectory', type=str, default='circle', help='trajectory for agent')
parser.add_argument('--buffer_size', type=int, default=10 ** 6, help='size of buffer')
parser.add_argument('--batch_size', type=int, default=2 ** 10, help='batch size')
parser.add_argument('--refresh_target', type=int, default=600, help='refresh target network')
parser.add_argument('--total_steps', type=int, default=10**4, help='total_steps')
parser.add_argument('--decay_steps', type=int, default=2000, help='decay_steps')
parser.add_argument('--agent_type', type=str, default='ddpg', help='type of agent. available now: dqn, ddqn, ddpg')
args = parser.parse_args()

timeout = 50
max_steps_per_episode = 1500

env_list, state_dim = get_envs(get_size())
replay_buffer = ReplayBuffer(args.buffer_size)

number = args.simu_number
batch_size = args.batch_size
total_steps = args.total_steps
decay_steps = args.decay_steps
refresh_target = args.refresh_target

writer = SummaryWriter(comment="  agent = " + args.agent_type + ", simulation_number = " + str(number)
                               + ", batch_size = " + str(batch_size) + ", refresh_target = " + str(refresh_target)
                               + " ,total_steps = " + str(total_steps) + ", decay steps = " + str(decay_steps)
                               + ", buffer_size = " + str(args.buffer_size))

agent_type = args.agent_type
if agent_type == 'dqn' or agent_type == 'ddqn':
    agent = DeepQLearningAgent(state_dim,
                               batch_size=batch_size,
                               epsilon=1,
                               gamma=0.99,
                               device=device,
                               algo=agent_type,
                               writer=writer,
                               refresh_target=refresh_target,
                               replay_buffer=replay_buffer)
elif agent_type == 'ddpg':
    agent = DeepDeterministicPolicyGradient(state_dim,
                                            device=device,
                                            act_dim=2,
                                            replay_buffer=replay_buffer,
                                            act_limit={0: [-0.21, 0.21], 1: [0.15, 0.35]},  # 0 - platform, 1 - wheel
                                            hidden_sizes_actor=(2048, 8192, 2048),
                                            hidden_sizes_critic=(2048, 8192, 2048),
                                            batch_size=batch_size,
                                            gamma=0.99,
                                            writer=writer)
else:
    raise RuntimeError('unknown type agent')

# loss_freq = 250  # 300 # 300
eval_freq = 100  # 300  # 400 statestate = env.reset().observation = env.reset().observation
change_env_freq = 1

step = 0

init_epsilon = 1
final_epsilon = 0

env = np.random.choice(env_list, size=1)[0]
state = env.reset().observation

rewards = list()
with trange(step, total_steps + 1) as progress_bar:
    for step in progress_bar:

        agent.epsilon = linear_decay(init_epsilon, final_epsilon, step, decay_steps)

        # play 1 episode and push to replay buffer
        total_reward = play_and_record(state, agent, env, timeout, step, max_steps_per_episode)
        rewards.append(total_reward)
        writer.add_scalar("episode reward ", total_reward, step)
        writer.add_scalar("average reward per episode", round(np.mean(rewards), 4), step)

        env = env_list[step % len(env_list)]
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
            mean_reward = mean_reward_per_episode(
                agent, env, timeout, n_games=3, t_max=1500
            )
            writer.add_scalar("Mean reward per 3 episode #" + str(number), mean_reward, step)
            writer.add_scalar("size of replay buffer # " + str(number), replay_buffer.buffer_len(), step)
            writer.add_scalar("epsilon change # " + str(number), agent.epsilon, step)

            env.reset()

        if step % 3000 == 0:
            i = int(step / 3000)
            save_model(i, number, agent_type)

writer.flush()

save_model(int(step/3000), number, agent_type)

# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
