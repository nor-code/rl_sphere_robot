import matplotlib.pyplot as plt
import numpy as np
import torch
from dm_control.viewer import application

from agent.dqn import DeepQLearningAgent
from robot.enviroment import make_env, curve, circle

pos = np.array([[0, 0]])
i = 0
V = []
U = []

# def action_policy(timestamp):
#     qvalues = agent.get_qvalues([timestamp.observation])
#     action = agent.index_to_pair[qvalues.argmax(axis=-1)[0]]
#     return action
#
agent = DeepQLearningAgent(state_dim=26,
                           batch_size=1,
                           epsilon=0,
                           gamma=0.99,
                           device='cpu',
                           algo='ddqn',
                           replay_buffer=None,
                           refresh_target=None,
                           writer=None,
                           file=None)
#
agent.q_network.load_state_dict(torch.load('../models/ddqn1_3.pt', map_location=torch.device('cpu')))
agent.q_network.eval()

final_time = 0
actions = np.array([])
total_reward = 0.0

def action_policy(time_step):
    global pos, V, U, i, agent, actions, final_time, total_reward

    i += 1

    val = np.random.random()
    observation = time_step.observation

    pos = np.append(pos, [observation[0:2]], axis=0)
    V.append(observation[2])
    U.append(observation[3])

    q = agent.get_qvalues([observation])
    index = q.argmax(axis=-1)[0]
    print(index)

    if time_step.reward is not None:
        total_reward += time_step.reward

    action = agent.index_to_pair[index]
    actions = np.concatenate((actions, action), axis=0)
    final_time = observation[4]
    return action

    # if i < 200:
    #     return [0.2205, 0.20]
    #
    # else:nohup python3 main.py --simu_number=1 --type_task=4 --agent_type=ddqn --trajectory=circle --buffer_size=6000000 --batch_size=2048 --total_steps=12000 --decay_steps=6000 > out1.log &
    #     return [-0.22, 0.31]


env, x_y = make_env(episode_timeout=90, type_task=4, trajectory='circle', begin_index_=0)
app = application.Application()
app.launch(env, policy=action_policy)

actions = actions.T.reshape(-1, 2)

traj = plt.figure().add_subplot()
traj.plot(pos[:, 0][1:], pos[:, 1][1:], label="trajectory")
traj.plot(x_y[0], x_y[1], label="desired_trajectory")
traj.quiver(pos[:, 0][1:], pos[:, 1][1:], V, U, color=['r', 'b', 'g'], angles='xy', width=0.002)
traj.legend(loc='upper right')
traj.set_xlabel('x,    total reward = ' + str(round(total_reward, 3)))
traj.set_ylabel('y')

control, ax = plt.subplots(2, 1)
ax[0].plot(np.linspace(0, 60, len(actions)), actions[:, 0], label="platform_signal")
ax[0].set_xlabel('time, s')
ax[0].set_ylabel('platform_signal')
ax[0].legend(loc='upper right')

ax[1].plot(np.linspace(0, 60, len(actions)), actions[:, 1], label="wheel_signal", color='red')
ax[1].set_xlabel('time, s')
ax[1].set_ylabel('wheel_signal')
ax[1].legend(loc='upper right')

print("V_max = ", max(V))
print("U_max = ", max(U))
print("count iteration: i = ", i)
plt.show()
