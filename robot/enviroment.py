import collections

import numpy as np
from dm_control.rl import control

from robot.model import RobotPhysics
from robot.task_1 import TrakingTrajectoryTask1
from robot.task_2 import TrakingTrajectoryTask2
from robot.task_3 import TrakingTrajectoryTask3


def trajectory():
    t = np.linspace(0, np.pi, 12)
    x_ = [np.sin(t_) for t_ in t]
    y_ = [- np.cos(t_) + 1 for t_ in t]
    # x_ = [0, 0.5, 1, 1, 1.1]
    # y_ = [0, 0.5, 1, 1.5, 2]
    return x_, y_


def point():
    x, y = trajectory()
    return collections.OrderedDict().fromkeys(zip(x, y))


def make_env(xml_file='robot_4.xml', episode_timeout=30, type_task=2):
    task = None
    state_dim = 0
    physics = RobotPhysics.from_xml_path(xml_file)

    if type_task == 1:
        state_dim = 4  # x y cos_a sin_a
        task = TrakingTrajectoryTask1(points_function=point, timeout=episode_timeout)
    elif type_task == 2:
        state_dim = 2  # x y
        task = TrakingTrajectoryTask2(points_function=point, timeout=episode_timeout)
    elif type_task == 3:
        state_dim = 4
        task = TrakingTrajectoryTask3(points_function=point, timeout=episode_timeout)

    return control.Environment(physics, task, time_limit=40, n_sub_steps=30), state_dim # n_sub_steps = 50
