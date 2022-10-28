import matplotlib.pyplot as plt
import numpy as np
import torch
from dm_control.viewer import application

from agent.ddpg import DeepDeterministicPolicyGradient
from robot.enviroment import make_env, curve, circle, get_state_dim

task = 9
dim_state = get_state_dim(task)
pos = np.array([[0, 0]])
i = 0
V = []
U = []

agent = DeepDeterministicPolicyGradient(dim_state,
                                        device='cpu',
                                        act_dim=2,
                                        replay_buffer=None,
                                        batch_size=1,
                                        gamma=0.99,
                                        writer=None)
#
# agent.q_network.load_state_dict(torch.load('../models/ddqn2_3.pt', map_location=torch.device('cpu')))
# agent.q_network.eval()
# 22_october/task8/
agent.policy.load_state_dict(torch.load('../models/27_october/task_9/ddpg_policy_1_3.pt', map_location=torch.device('cpu')))
agent.policy.eval()
agent.qf.load_state_dict(torch.load('../models/27_october/task_9/ddpg_Q_1_3.pt', map_location=torch.device('cpu')))
agent.qf.eval()

final_time = 0
actions = np.array([])
total_reward = 0.0

def action_policy(time_step):
    global pos, V, U, i, agent, actions, final_time, total_reward

    i += 1

    val = np.random.random()
    observation = time_step.observation

    x, y, _ = env.physics.named.data.geom_xpos['wheel_']
    pos = np.append(pos, [[x, y]], axis=0)

    v_x, v_y, _ = env.physics.named.data.sensordata['wheel_vel']
    V.append(v_x)
    U.append(v_y)

    if time_step.reward is not None:
        total_reward += time_step.reward
    action = agent.get_action([observation])
    actions = np.concatenate((actions, action[0]), axis=0)
    final_time = env.physics.data.time
    return action


env, x_y = make_env(episode_timeout=110, type_task=task, trajectory='random', begin_index_=5, count_substeps=3)
app = application.Application()
app.launch(env, policy=action_policy)

actions = actions.T.reshape(-1, 2)

traj = plt.figure().add_subplot()
traj.plot(pos[:, 0][1:], pos[:, 1][1:], label="trajectory")
traj.plot(x_y[0], x_y[1], label="desired_trajectory")
# traj.quiver(pos[:, 0][1:], pos[:, 1][1:], V, U, color=['r', 'b', 'g'], angles='xy', width=0.002)
traj.legend(loc='upper right')
traj.set_xlabel('x,    total reward = ' + str(round(total_reward, 3)))
traj.set_ylabel('y')

control, ax = plt.subplots(2, 1)
ax[0].plot(np.linspace(0, 95, len(actions)), actions[:, 0], label="platform_signal")
ax[0].set_xlabel('time, s')
ax[0].set_ylabel('platform_signal')
ax[0].legend(loc='upper right')

ax[1].plot(np.linspace(0, 95, len(actions)), actions[:, 1], label="wheel_signal", color='red')
ax[1].set_xlabel('time, s')
ax[1].set_ylabel('wheel_signal')
ax[1].legend(loc='upper right')

print("V_max = ", max(V))
print("U_max = ", max(U))
print("count iteration: i = ", i)
plt.show()

# U_max = 0.2408
# U_max = 0.2958

# def action_policy(timestamp):
#     qvalues = agent.get_qvalues([timestamp.observation])
#     action = agent.index_to_pair[qvalues.argmax(axis=-1)[0]]
#     return action
#
# agent = DeepQLearningAgent(state_dim=20,
#                            batch_size=1,
#                            epsilon=0,
#                            gamma=0.99,
#                            device='cpu',
#                            algo='ddqn',
#                            replay_buffer=None,
#                            refresh_target=None,
#                            writer=None)
