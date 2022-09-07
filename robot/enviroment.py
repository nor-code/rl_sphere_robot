import collections

import numpy as np
from dm_control.rl import control

from robot.mock_task import MockTask
from robot.model import RobotPhysics
from robot.task_1 import TrakingTrajectoryTask1
from robot.task_2 import TrakingTrajectoryTask2
from robot.task_3 import TrakingTrajectoryTask3


def curve():
    t = np.linspace(0, 2 * np.pi, 16)
    x_ = [np.sin(t_) for t_ in t]
    y_ = [- np.cos(t_/2) + 1 for t_ in t]

    deg = np.rad2deg(np.arctan(y_[1] / x_[1]))
    print("deg curve = ", deg)

    return x_, y_


def circle():
    t = np.linspace(0, 2 * np.pi, 30)
    x_ = [np.sin(t_) for t_ in t]
    y_ = [- np.cos(t_) + 1 for t_ in t]

    deg = np.rad2deg(np.arctan(y_[1] / x_[1]))
    print("deg circle = ", deg)

    return x_, y_


def point(type):
    if type == 'circle':
        x, y = circle()
        return collections.OrderedDict().fromkeys(zip(x, y))
    elif type == 'curve':
        x, y = curve()
        return collections.OrderedDict().fromkeys(zip(x, y))


def make_env(xml_file='robot_4.xml', episode_timeout=30, type_task=2, trajectory=None):
    physics = RobotPhysics.from_xml_path(xml_file)

    if type_task == 1:
        state_dim = 4  # x y cos_a sin_a
        task = TrakingTrajectoryTask1(points_function=point,  type_curve=trajectory, timeout=episode_timeout)
    elif type_task == 2:
        state_dim = 2  # x y
        task = TrakingTrajectoryTask2(points_function=point,  type_curve=trajectory, timeout=episode_timeout)
    elif type_task == 3:
        state_dim = 4  # x y v_x v_y
        task = TrakingTrajectoryTask3(points_function=point, type_curve=trajectory, timeout=episode_timeout)
    else:
        state_dim = 4  # x y v_x v_y
        task = MockTask(points_function=point, type_curve=trajectory, timeout=episode_timeout)

    return control.Environment(physics, task, time_limit=50, n_sub_steps=45), state_dim
