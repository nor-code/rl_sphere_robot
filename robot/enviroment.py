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


def get_string_xml(roll_angle):
    return f"""
    <?xml version="1.0"?>
    <mujoco model="sphere_robot">

        <compiler inertiafromgeom="true" angle="degree"/>
        <option integrator="RK4"/>
        <option timestep="0.005" iterations="50" solver="Newton" tolerance="1e-10"/>
        <size njmax="1500" nconmax="5000" nstack="5000000"/>


        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
            <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
            <texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="1278"
                rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>
            <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
            <material name="matgeom" texture="texgeom" texuniform="true" rgba="0.8 0.6 .4 1"/>
        </asset>

        <visual>
            <map force="0.1" zfar="30"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <quality shadowsize="2048"/>
            <global offwidth="800" offheight="800"/>
        </visual>

        <worldbody>
            <geom name="floor" pos="0 0 0" size="0 0 .25" type="plane" material="matplane" condim="3"/>
            <camera name="fixed" pos="0 -4 1" zaxis="0 -1 0"/>

            <light pos="0 0 6"/>
            <light directional="false" diffuse=".2 .2 .2" specular="0 0 0" pos="0 0 5" dir="0 0 -1" castshadow="false"/>

            <body name="shell" pos="0 0 0" euler="0 0 {roll_angle}">
                <joint name="shell_floor" type="free"/>
                <geom name="sphere_shell" type="sphere" pos="0 0 0" size=".2 .19" rgba=".0 .0 .0 .2" mass="0.2" friction="1 5 1" group="1"/>

                <body name="wheel">
                    <joint name="wheel_with_shell" type="ball" frictionloss="0.0068"/>
                    <geom name="wheel_" type="cylinder" fromto="-0.008 0 -0.15  0.008 0 -0.15" size="0.049 0.005" mass="0.7"/>
                    <!-- friction[0] sliding friction, friction[1] torsional friction, friction[2] rolling friction -->

                    <body>
                       <geom name="wheel_axis1" type="cylinder" fromto="-0.03 0 -0.15  -0.008 0 -0.15" size="0.005" mass="0.005"/>
                       <geom name="wheel_axis2" type="cylinder" fromto="0.008 0 -0.15  0.03 0 -0.15" size="0.005" mass="0.005"/>

                       <geom name="fork1" type="capsule" fromto="-0.03 0 -0.07  -0.03 0 -0.16" size="0.01" mass="0.01" group="1"/>
                       <geom name="fork2" type="capsule" fromto="0.03 0 -0.07  0.03 0 -0.16" size="0.01" mass="0.01" group="1"/>
                       <geom name="link_f1_f2" type="capsule" fromto="-0.03 0 -0.07   0.03 0 -0.07" size="0.01" mass="0.001" group="1"/>
                       <geom name="fork0" type="capsule" fromto="0 0 -0.03  0 0 -0.07" size="0.01" mass="0.1" group="0.01"/>
                    </body>

                    <body>
                        <joint name="fork_with_platform" type="hinge" axis="0 0 1" frictionloss="0.1"/>
                        <geom name="platform" type="cylinder" pos="0 0 -0.03" size=".15 .005" rgba=".0 .0 .3 .5" mass="2" group="1"/>
                        <geom name="line1" type="cylinder" fromto="0       -0.15 -0.03    0     -0.167 -0.03" size="0.005"/>
                        <geom name="line2" type="cylinder" fromto="0.1299  0.075 -0.03    0.1472 0.085 -0.03" size="0.005"/>
                        <geom name="line3" type="cylinder" fromto="-0.1299 0.075 -0.03   -0.1472 0.085 -0.03" size="0.005"/>
                        <site name="mpu9250" pos="0 0 -0.03" size=".03 .03 .03" type="ellipsoid" rgba="0.3 0.2 0.1 0.3"/>
                    </body>
                </body>
            </body>
        </worldbody>

    <!--     <contact>-->
    <!--       <pair name="friction_shell" geom1="floor" geom2="sphere_shell" condim="3" friction="0 1"/>-->
    <!--     </contact>-->

        <actuator>
            <motor name="platform_motor" gear="0.107" joint="fork_with_platform" ctrllimited="true" ctrlrange="-0.975 0.975"/>
            <motor name="wheel_motor" gear="90" joint="wheel_with_shell" ctrllimited="true" ctrlrange="0.26 0.6"/>
        </actuator>

        <sensor>
            <subtreecom name="shell_center" body="shell"/>

            <jointpos name="fork_wheel_angle" joint="fork_with_platform"/> <!-- угол поворота колеса -->

            <subtreelinvel name="sphere_vel" body="shell"/>
            
            <subtreelinvel name="wheel_vel" body="wheel"/>
            
            <frameangvel name="sphere_angular_vel" objtype="geom" objname="sphere_shell"/>
            
            <accelerometer name="imu_accel" site="mpu9250"/>
            <gyro name="imu_gyro" site="mpu9250"/>
        </sensor>
    </mujoco>
    """

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


def random_trajectory():
    global scope
    # общее количество точек на кривой
    total_points = 45 # task 3 - 25

    x_init = np.random.uniform(scope['x'][0], scope['x'][1])
    y_init = np.random.uniform(scope['y'][0], scope['y'][1])

    radius = np.random.randn(1, total_points) * np.logspace(-1.4, -3.5, total_points)
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
        return 23
    return -1


def make_env(episode_timeout=30, type_task=2, trajectory=None, begin_index_=0):
    trajectory_fun = determine_trajectory(trajectory)

    x_y = trajectory_fun()
    points = np.array(x_y).T
    dy = points[begin_index_ + 1][1] - points[begin_index_][1]
    dx = points[begin_index_ + 1][0] - points[begin_index_][0]

    sign_y = np.sign(np.dot([dx, dy], [0, 1]))
    sign_x = np.sign(np.dot([dx, dy], [1, 0]))

    deg = np.rad2deg(np.arctan(dy / dx))

    roll_angle = deg - 90  # y / x

    if sign_y < 0 and sign_x < 0:
        roll_angle -= 180
    elif sign_x < 0:
        roll_angle += 180

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

    return control.Environment(physics, task, time_limit=episode_timeout, n_sub_steps=25), x_y  # n_sub_steps = 17
