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
from robot.enviroment import make_env, get_state_dim
# from IPython.display import clear_output
from utils.utils import build_trajectory

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


def get_env(size,timeout,args):
    # begin_index = [size // 10, 2 * size // 10, (3 * size) // 10, (4 * size) // 10,
    #                size // 2, (6 * size) // 10, (7 * size) // 10, (8 * size) // 10, (9 * size) // 10]

    if size == 1:
        begin_index = 0
    else:
        begin_index = np.random.choice(range(0, size - 1), size=1)
        begin_index = np.random.choice(begin_index, size=1)[0]

    env_i, x_y = make_env(
        episode_timeout=timeout, type_task=args.type_task, trajectory=args.trajectory, begin_index_= begin_index
    )
    return env_i, x_y


def get_size(args):
    if args.trajectory == 'circle':
        return 50
    elif args.trajectory == 'curve':
        return 25
    elif args.trajectory == 'random':
        return 60
    elif args.trajectory == 'one_point':
        return 1


def save_model(backup_iteration, _number, name_agent, agent):
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



def build_agent(args, d = 'cuda:0'):
    np.random.seed(42)
    device = torch.device(d if torch.cuda.is_available() else 'cpu')
    replay_buffer = ReplayBuffer(args.buffer_size)

    batch_size = args.batch_size
    refresh_target = args.refresh_target
    writer = SummaryWriter(comment="  agent = " + args.agent_type + ", simulation_number = " + str(args.simu_number)
                                   + ", batch_size = " + str(batch_size) + ", refresh_target = " + str(refresh_target)
                                   + " ,total_steps = " + str(args.total_steps) + ", decay steps = " + str(
        args.decay_steps)
                                   + ", buffer_size = " + str(args.buffer_size))

    # Get agent
    agent_type = args.agent_type
    state_dim = get_state_dim(type_task=args.type_task)
    if agent_type == 'dqn' or agent_type == 'ddqn':
        agent = DeepQLearningAgent(state_dim,
                                   batch_size=args.batch_size,
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
                                                batch_size=batch_size,
                                                gamma=0.99,
                                                writer=writer)
    else:
        raise RuntimeError('unknown type agent')

    return agent, writer, replay_buffer




def train(args,agent,writer,replay_buffer):

    timeout = args.timeout
    max_steps_per_episode = args.max_steps_per_episode


    number = args.simu_number

    total_steps = args.total_steps
    decay_steps = args.decay_steps


    # dataset_file = open('dataset/' + 'dataset_' + str(args.agent_type) + "_" + str(args.simu_number) + '.csv', 'a')

    # loss_freq = 250  # 300 # 300
    eval_freq = args.eval_freq  # 300  # 400 statestate = env.reset().observation = env.reset().observation
    change_env_freq = 1

    step = 0
    count_point_on_trajectory = get_size(args)

    init_epsilon = 1
    final_epsilon = 0

    rewards = list()
    with trange(step, total_steps + 1) as progress_bar:
        for step in progress_bar:
            env, x_y = get_env(count_point_on_trajectory,timeout,args)
            state = env.reset().observation

            agent.epsilon = linear_decay(init_epsilon, final_epsilon, step, decay_steps)

            # play 1 episode and push to replay buffer
            total_reward = play_and_record(state, agent, env, timeout, step, max_steps_per_episode)
            rewards.append(total_reward)
            writer.add_scalar("episode reward ", total_reward, step)
            writer.add_scalar("average reward per episode", round(np.mean(rewards), 4), step)

            if step % eval_freq == 0:
                env.reset()

                plot_buf, fig = build_trajectory(
                    agent=agent, enviroment=env, timeout=timeout, x_y=x_y, type_task=args.type_task
                )

                pillow_img = PIL.Image.open(plot_buf)
                #pillow_img.save("tmp.jpg")
                tensor_img = ToTensor()(pillow_img)
                writer.add_image("trajectory #" + str(number), tensor_img, step / eval_freq)
                plt.close(fig)

                env.reset()
                mean_reward = mean_reward_per_episode(
                    agent, env, timeout, n_games=3, t_max=1600
                )
                writer.add_scalar("Mean reward per 3 episode #" + str(number), mean_reward, step)
                writer.add_scalar("size of replay buffer # " + str(number), replay_buffer.buffer_len(), step)
                writer.add_scalar("epsilon change # " + str(number), agent.epsilon, step)

                env.reset()

            if step > 0 and step % 3000 == 0:
                i = int(step / 3000)
                save_model(i, number, args.agent_type,agent)

    writer.flush()

    save_model(int(step / 3000), number, args.agent_type, agent)