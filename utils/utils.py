import io

import matplotlib.pyplot as plt
import numpy as np
from dm_control.rl.control import PhysicsError
from dm_env import StepType


def build_trajectory(agent=None, enviroment=None, timeout=50, x_y=None, type_task=None):
    _x_y = x_y

    times = []
    V = []
    U = []
    pos = np.array([[0, 0]])
    env = enviroment
    total_reward = 0
    time_step = env.reset()
    prev_time = env.physics.data.time
    observation = time_step.observation

    t = np.linspace(0, 2 * np.pi, 20)
    local_x_O = [0.24 * np.sin(t_) for t_ in t]
    local_y_O = [0.24 * np.cos(t_) for t_ in t]
    circles = np.array([local_x_O, local_y_O]).T

    while env.physics.data.time < timeout:
        action = agent.get_action([observation])
        try:
            time_step = env.step(action)
        except PhysicsError:
            print("physicx error  time = ", prev_time)
            break

        if time_step.reward is None or time_step.step_type == StepType.LAST:
            # print("reward is None ! time = ", prev_time)
            break

        total_reward += time_step.reward
        prev_time = env.physics.data.time

        x, y, _ = env.physics.named.data.geom_xpos['wheel_']

        array = np.array([[x_ + x for x_ in local_x_O], [y_ + y for y_ in local_y_O]]).T
        circles = np.append(circles, array, axis=0)

        if type_task == 1:
            V.append(observation[2])
            U.append(observation[3])
        pos = np.append(pos, [[x, y]], axis=0)

        times.append(env.physics.data.time)

    figure = plt.figure(figsize=(10, 10))
    trajectory = figure.add_subplot()
    trajectory.plot(pos[:, 0][1:], pos[:, 1][1:], label="trajectory")
    trajectory.plot(_x_y[0], _x_y[1], label="desired_trajectory")

    circles = circles[21:]
    trajectory.plot(circles.T[0], circles.T[1], alpha=0.4)
    trajectory.scatter(_x_y[0], _x_y[1], color='red', lw=0.01)

    if type_task == 1:
        trajectory.quiver(pos[:, 0][1:], pos[:, 1][1:], V, U, color=['r', 'b', 'g'], angles='xy', width=0.002)

    trajectory.set_xlabel('x, total_reward = ' + str(total_reward))
    trajectory.set_ylabel('y')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpeg')
    buffer.seek(0)
    return buffer, figure
