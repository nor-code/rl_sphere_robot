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
                           algo='ddqn')
#
agent.q_network.load_state_dict(torch.load('../models/name3.pt', map_location=torch.device('cpu')))
agent.eval()

final_time = 0
actions = np.array([])
def action_policy(time_step):
    global pos, V, U, i, agent, actions, final_time

    i += 1

    val = np.random.random()
    observation = time_step.observation

    pos = np.append(pos, [observation[0:2]], axis=0)
    V.append(observation[2])
    U.append(observation[3])

    q = agent.get_qvalues([observation])
    index = q.argmax(axis=-1)[0]
    print(index)
    action = agent.index_to_pair[index]
    actions = np.concatenate((actions, action), axis=0)
    final_time = observation[4]
    return [0.23, 0.33]

    # if i < 200:
    #     return [0.2205, 0.20]
    #
    # else:
    #     return [-0.22, 0.31]


env = make_env(episode_timeout=50, type_task=3, trajectory='circle', begin_index_=18)[0]
app = application.Application()
app.launch(env, policy=action_policy)

actions = actions.T.reshape(-1, 2)

traj = plt.figure().add_subplot()
traj.plot(pos[:, 0][1:], pos[:, 1][1:], label="trajectory")
traj.plot(circle()[0], circle()[1], label="desired_trajectory")
traj.quiver(pos[:, 0][1:], pos[:, 1][1:], V, U, color=['r', 'b', 'g'], angles='xy', width=0.002)
traj.legend(loc='upper right')
traj.set_xlabel('x')
traj.set_ylabel('y')

control, ax = plt.subplots(2, 1)
ax[0].plot(np.linspace(0, final_time, len(actions)), actions[:, 0], label="platform_signal")
ax[0].set_xlabel('time, s')
ax[0].set_ylabel('platform_signal')
ax[0].legend(loc='upper right')

ax[1].plot(np.linspace(0, final_time, len(actions)), actions[:, 1], label="wheel_signal", color='red')
ax[1].set_xlabel('time, s')
ax[1].set_ylabel('wheel_signal')
ax[1].legend(loc='upper right')

print("V_max = ", max(V))
print("U_max = ", max(U))
print("count iteration: i = ", i)
plt.show()
