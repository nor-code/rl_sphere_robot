import collections

import numpy as np
from dm_control.rl import control

from robot.model import RobotPhysics
from robot.task import TrakingTrajectoryTask


def trajectory():
    # t = np.linspace(0, 2 * np.pi, 120)
    # x_ = [2 * np.sin(t_) for t_ in t]
    # y_ = [2 * np.cos(t_) - 2 for t_ in t]
    x_ = [0, 0.2, 0.5]
    y_ = [0, 0.2, 0.2]
    return x_, y_


def point():
    x, y = trajectory()
    return collections.OrderedDict().fromkeys(zip(x, y))


def make_env():
    physics = RobotPhysics.from_xml_path('robot_4.xml')
    task = TrakingTrajectoryTask(points_function=point)
    return control.Environment(
        physics, task, time_limit=50, n_sub_steps=50
    )
