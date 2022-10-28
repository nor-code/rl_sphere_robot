import argparse


from runner import build_agent,train


parser = argparse.ArgumentParser(description='DQN/DDPG Spherical Robot')
parser.add_argument('--simu_number', type=int, default=1, help='number of simulation')
parser.add_argument('--type_task', type=int, default=8, help='type of task. now available 4, 5')
parser.add_argument('--trajectory', type=str, default='one_point', help='trajectory for agent, circle, curve, random')
parser.add_argument('--buffer_size', type=int, default=10 ** 6, help='size of buffer')
parser.add_argument('--batch_size', type=int, default=2 ** 10, help='batch size')
parser.add_argument('--refresh_target', type=int, default=100, help='refresh target network') #400
parser.add_argument('--total_steps', type=int, default=10**3, help='total_steps') # 10**4
parser.add_argument('--decay_steps', type=int, default=100, help='decay_steps')
parser.add_argument('--agent_type', type=str, default='ddpg', help='type of agent. available now: dqn, ddqn, ddpg')
parser.add_argument('--timeout', type=int, default=70, help='Max episode len in simulator time')
parser.add_argument('--max_steps_per_episode', type=int, default=1600, help='Max steps per episode')
parser.add_argument('--eval_freq', type=int, default=5, help='Evaluation run frequency')
parser.add_argument('--neurons', type=int, default=64, help='Neurons in first layer')
parser.add_argument('--device', type=str, default="cuda:1", help='GPU num')

args = parser.parse_args()

agent, writer, replay_buffer = build_agent(args,d = args.device)
try:
    train(args, agent, writer, replay_buffer)
finally:
    writer.flush()

# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
