import numpy as np
from dm_control.suite import base

class MockTask(base.Task):

    def __init__(self, trajectory_function, begin_index, timeout, random=None):

        """тайм-аут одного эпизода"""
        self.timeout = timeout
        """ количество точек, которые мы достигли в рамках текущего эпищода """
        self.achievedPoints = 0

        """ целевая траектория и начальная точка на ней """
        self.p_fun = trajectory_function
        self.points = np.array(self.p_fun()).T
        self.begin_index = begin_index
        self.current_index = begin_index

        """текущая и предыдущая целевая точка ( координаты ) """
        self.current_point = self.points[begin_index]

        self.prev_point = None

        """ общее количество точек на пути """
        self.totalPoint = len(self.points)

        """ количество неправильных состояний за эпизод """
        self.count_invalid_states = 0

        """предыдущее расстояние и текущее до целевой точки на кривой"""
        self.prev_dist = self.current_dist = 0

        """ точка невозврата """
        self.point_no_return = [0, 0]

        self.state = [0, 0, 0, 0]

        super().__init__(random=random)

    def initialize_episode(self, physics):
        self.points = np.array(self.p_fun()).T

        index = self.begin_index
        self.current_index = index
        self.current_point = self.points[index]
        self.prev_point = None

        physics.named.data.qpos[0:3] = [self.current_point[0], self.current_point[1], 0.2]
        physics.named.data.qvel[:] = 0

        self.achievedPoints = 0
        self.count_invalid_states = 0

        self.prev_dist = self.current_dist = 0

        self.state = [0, 0, 0, 0]

        super().initialize_episode(physics)

    def get_observation(self, physics):
        # координаты центра колеса
        x, y, z = physics.named.data.geom_xpos['wheel_']
        # вектор скорости в абс системе координат
        v_x, v_y, v_z = physics.named.data.sensordata['wheel_vel']

        w_x, w_y, w_z = physics.named.data.sensordata['sphere_angular_vel']
        print("w_x = ", w_x, " w_y = ", w_y, " w_z = ", w_z)

        self.state = [x, y, v_x, v_y, physics.data.time]
        return self.state  # np.concatenate((xy, acc_gyro), axis=0)

    def get_reward(self, physics):
        return 1