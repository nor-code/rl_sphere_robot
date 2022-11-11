import collections

import numpy as np
from dm_control.rl import control

from robot.mock_task import MockTask
from robot.model import RobotPhysics
from robot.task_3 import TrakingTrajectoryTask3
from robot.task_4 import TrakingTrajectoryTask4
from robot.task_5 import TrakingTrajectoryTask5
from robot.task_6 import TrakingTrajectoryTask6
from robot.task_7 import TrakingTrajectoryTask7
from robot.task_one_point import TrakingTrajectoryTaskOnePoint
import os


def get_string_xml(roll_angle):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path + os.sep + "robot.xml"
    with open(filename, 'r') as file:
        xml_in_string = file.read()
    # TODO replace roll_angle to changing last zero in `euler="0 0 0"`
    return xml_in_string.replace("{roll_angle}", str(roll_angle))


# область в которой будут генерироваться начало замкнутой траектории
scope = {
    "x": [-1.5, 1.5],
    "y": [-0.5, 2.5],
}


def curve():
    t = np.linspace(0, 3 * np.pi, 30)
    x_ = [np.sin(t_ * 0.8) for t_ in t]
    y_ = [- np.cos(t_ / 1.5) + 1 for t_ in t]
    return x_, y_


def circle():
    t = np.linspace(0, 2 * np.pi, 50)
    x_ = [np.sin(t_) for t_ in t]
    y_ = [- np.cos(t_) + 1 for t_ in t]
    return x_, y_


def one_point():
    x = np.random.uniform()
    y = np.random.uniform()

    x = 1.0
    y = 1.0
    return [x], [y]  # np.expand_dims(x_y, axis = 0)


def random_trajectory():
    global scope
    # общее количество точек на кривой
    total_points = 65  # task 3 - 25

    x_init = np.random.uniform(scope['x'][0], scope['x'][1])
    y_init = np.random.uniform(scope['y'][0], scope['y'][1])

    radius = np.random.randn(1, total_points) * np.logspace(-1.63, -3.5, total_points)
    phi = np.random.randn(1, total_points) * np.logspace(-0.01, -1.2, total_points)
    omega = 2 * np.random.randn(1, total_points) * np.logspace(-0.01, -0.85, total_points) * np.pi

    t = np.linspace(0, 2 * np.pi, total_points)
    r = np.ones(total_points)
    for i in range(total_points):
        r += radius[0][i] * np.sin(omega[0][i] * t + phi[0][i])

    x = r * np.sin(t)  # + x_init
    y = - r * np.cos(t) + 1  # + y_init
    x[-1] = x[0]
    y[-1] = y[0]
    return x.tolist(), y.tolist()


def determine_trajectory(type):
    if type == 'one_point':
        return one_point
    if type == 'circle':
        return circle
    elif type == 'curve':
        return curve
    elif type == 'random':
        return random_trajectory


def get_state_dim(type_task):
    if type_task == 3:
        return 16
    elif type_task == 4:
        return 26
    elif type_task == 5:
        return 9
    elif type_task == 6:
        return 16
    elif type_task == 7:
        return 28
    elif type_task == 8:  # one point
        return 6  # x , y , v_x , x_y, target_x,target_y
    return -1


def make_env(episode_timeout=30, type_task=2, trajectory=None, begin_index_=0):
    trajectory_fun = determine_trajectory(trajectory)

    x_y = trajectory_fun()

    points = np.array(x_y).T

    roll_angle = 0

    physics = RobotPhysics.from_xml_string(get_string_xml(roll_angle))

    if type_task == 3:
        task = TrakingTrajectoryTask3(trajectory_x_y=points, begin_index=begin_index_, timeout=episode_timeout)
    if type_task == 4:
        task = TrakingTrajectoryTask4(trajectory_x_y=points, begin_index=begin_index_, timeout=episode_timeout)
    elif type_task == 5:
        task = TrakingTrajectoryTask5(trajectory_x_y=points, begin_index=begin_index_, timeout=episode_timeout)
    elif type_task == 6:
        task = TrakingTrajectoryTask6(trajectory_x_y=points, begin_index=begin_index_, timeout=episode_timeout)
    elif type_task == 7:
        task = TrakingTrajectoryTask7(trajectory_x_y=points, begin_index=begin_index_, timeout=episode_timeout)
    elif type_task == 8:
        task = TrakingTrajectoryTaskOnePoint(trajectory_x_y=points, begin_index=begin_index_, timeout=episode_timeout)

    return control.Environment(physics, task, time_limit=episode_timeout, n_sub_steps=12), x_y  # n_sub_steps = 17
