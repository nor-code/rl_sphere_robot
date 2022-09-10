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
# agent = DeepQLearningAgent(state_dim=4,
#                            batch_size=1,
#                            epsilon=0,
#                            gamma=0.99,
#                            device='cpu',
#                            algo='ddqn')
#
# agent.q_network.load_state_dict(torch.load('../models/name_dqn.pt', map_location=torch.device('cpu')))
# agent.eval()

def action_policy(time_step):
    global pos, V, U, i, agent

    i += 1

    val = np.random.random()
    observation = time_step.observation

    pos = np.append(pos, [observation[0:2]], axis=0)
    V.append(observation[2])
    U.append(observation[3])

    if i < 200:
        return [0, 0.33]
        # q = agent.get_qvalues(observation)
        # index = agent.sample_actions(q)
        # print(index)
        # action = agent.index_to_pair[index]
        # return action
    else:
        return [-0.22, 0.31]
    # return np.array([0.25, 0])
    # if 0 < val < 0.25:
    #     return np.array([0, 0.2])
    # elif 0.4 <= val < 0.8:
    #     return np.array([0, -0.2])
    # else:
    #     return np.array([-0.3, 0])


env = make_env(type_task=-1, trajectory='circle', begin_index_=0)[0]
app = application.Application()
app.launch(env, policy=action_policy)

traj = plt.figure().add_subplot()
traj.plot(pos[:, 0][1:], pos[:, 1][1:], label="trajectory")
traj.plot(circle()[0], circle()[1], label="desired_trajectory")
traj.quiver(pos[:, 0][1:], pos[:, 1][1:], V, U, color=['r', 'b', 'g'], angles='xy', width=0.002)
traj.set_xlabel('x')
traj.set_ylabel('y')
plt.show()
