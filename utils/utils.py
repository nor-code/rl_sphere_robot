import io

import matplotlib.pyplot as plt
import numpy as np
from dm_control.rl.control import PhysicsError


def build_trajectory(agent=None, enviroment=None, timeout=30, trajectory_func=None, type_task=None):
    times = []
    actions = []
    V = []
    U = []
    pos = np.array([[0, 0]])
    env = enviroment
    total_reward = 0
    time_step = env.reset()
    prev_time = env.physics.data.time
    # frames = []
    while env.physics.data.time < timeout:
        qvalues = agent.get_qvalues([time_step.observation])
        action = agent.index_to_pair[qvalues.argmax(axis=-1)[0]]
        actions = np.append(actions, action)

        try:
            time_step = env.step(action=action)
        except PhysicsError:
            print("physicx error  time = ", prev_time)
            break

        if time_step.reward is None:
            # print("reward is None ! time = ", prev_time)
            break

        total_reward += time_step.reward
        prev_time = env.physics.data.time

        observation = time_step.observation

        if type_task == 1:
            V.append(observation[2])
            U.append(observation[3])
        pos = np.append(pos, [observation[0:2]], axis=0)

        times.append(env.physics.data.time)

        # frame = env.physics.render(camera_id=0, width=300, height=300)
        # if env.physics.data.time > 1:
        #     frames.append(frame)

    # actions = actions.reshape(-1, 2)
    # fig1, ax = plt.subplots(5, 1)
    # ax[0].plot(times, pos[:, 0][1:])
    # ax[1].plot(times, pos[:, 1][1:])
    # ax[3].plot(times, actions[:, 0][:-1])
    # ax[4].plot(times, actions[:, 1][:-1])

    figure = plt.figure(figsize=(10, 10))
    trajectory = figure.add_subplot()
    trajectory.plot(pos[:, 0][1:], pos[:, 1][1:], label="trajectory")
    trajectory.plot(trajectory_func()[0], trajectory_func()[1], label="desired_trajectory")
    if type_task == 1:
        trajectory.quiver(pos[:, 0][1:], pos[:, 1][1:], V, U, color=['r', 'b', 'g'], angles='xy', width=0.002)
    trajectory.set_xlabel('x, total_reward = ' + str(total_reward))
    trajectory.set_ylabel('y')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpeg')
    buffer.seek(0)
    return buffer, figure
