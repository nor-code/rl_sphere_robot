import matplotlib.pyplot as plt
import numpy as np
from dm_control.viewer import application

from robot.enviroment import make_env, trajectory

pos = np.array([[0, 0]])
V = []
U = []


def action_policy(time_step):
    global pos, V, U

    val = np.random.random()
    observation = time_step.observation

    pos = np.append(pos, [observation[0:2]], axis=0)
    V.append(observation[2])
    U.append(observation[3])

    # return np.array([0.25, 0])
    if 0 < val < 0.4:
        return np.array([0, 0.2])
    elif 0.4 <= val < 0.8:
        return np.array([0, -0.2])
    else:
        return np.array([-0.3, 0])


env = make_env(xml_file="../robot_4.xml")
app = application.Application()
app.launch(env, policy=action_policy)

traj = plt.figure().add_subplot()
traj.plot(pos[:, 0][1:], pos[:, 1][1:], label="trajectory")
traj.plot(trajectory()[0], trajectory()[1], label="desired_trajectory")
traj.quiver(pos[:, 0][1:], pos[:, 1][1:], V, U, color=['r', 'b', 'g'], angles='xy', width=0.002)
traj.set_xlabel('x')
traj.set_ylabel('y')
plt.show()
