import argparse


from runner import build_agent,train


parser = argparse.ArgumentParser(description='DQN/DDPG Spherical Robot')
parser.add_argument('--simu_number', type=int, default=1, help='number of simulation')
parser.add_argument('--type_task', type=int, default=8, help='type of task. now available 4, 5')
parser.add_argument('--trajectory', type=str, default='one_point', help='trajectory for agent, circle, curve, random')
parser.add_argument('--buffer_size', type=int, default=10 ** 6, help='size of buffer')
parser.add_argument('--batch_size', type=int, default=2 ** 10, help='batch size')
parser.add_argument('--refresh_target', type=int, default=400, help='refresh target network')
parser.add_argument('--total_steps', type=int, default=10**4, help='total_steps')
parser.add_argument('--decay_steps', type=int, default=100, help='decay_steps')
parser.add_argument('--agent_type', type=str, default='ddpg', help='type of agent. available now: dqn, ddqn, ddpg')
args = parser.parse_args()


agent, writer, replay_buffer = build_agent(args)
train(args, agent, writer, replay_buffer)

# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
