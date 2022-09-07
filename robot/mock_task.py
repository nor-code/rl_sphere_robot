import numpy as np
from dm_control.suite import base

class MockTask(base.Task):

    def __init__(self, points_function, type_curve, timeout, random=None):
        self.type_curve = type_curve
        """тайм-аут одного эпизода"""
        self.timeout = timeout
        """ количество точек, которые мы достигли в рамках текущего эпищода """
        self.achievedPoints = 0

        """ функции для целевой траектории"""
        self.p_fun = points_function
        self.points = points_function(type_curve)

        """текущая и предыдущая целевая точка ( координаты ) """
        self.current_point = self.points.popitem(last=False)

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
        physics.named.data.qpos[0:3] = [0, 0, 0.2]
        physics.named.data.qvel[:] = 0
        self.points = self.p_fun(self.type_curve)

        self.current_point = self.points.popitem(last=False)
        self.prev_point = None

        self.achievedPoints = 0

        self.count_invalid_states = 0

        self.prev_dist = self.current_dist = 0

        self.state = [0, 0, 0, 0]

        super().initialize_episode(physics)

    def get_observation(self, physics):
        # координаты центра колеса
        x, y, z = physics.named.data.geom_xpos['wheel_']
        # вектор скорости в абс системе координат
        v_x, v_y, v_z = physics.named.data.sensordata['sphere_vel']
        self.state = [x, y, v_x, v_y]
        return self.state  # np.concatenate((xy, acc_gyro), axis=0)

    def get_reward(self, physics):
        return 1