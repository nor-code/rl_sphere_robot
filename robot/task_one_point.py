import numpy as np
from numpy.linalg import norm
from dm_control.suite import base
import torch

class TrakingTrajectoryTaskOnePoint(base.Task):

    def __init__(self, trajectory_x_y, begin_index, timeout, R=0.242, random=None):
        """ тайм-аут одного эпизода """
        self.timeout = timeout
        """ количество точек, которые мы достигли в рамках текущего эпищода """
        self.achievedPoints = 0
        """ радиус окрестности для 4х ближайших точек """
        #self.radius = R

        """ целевая траектория и начальная точка на ней """
        self.point_x_y = trajectory_x_y
        self.points = np.copy(self.point_x_y, order='K')
        assert begin_index == 0
        self.begin_index = begin_index

        self.dist = 0.0

        # TODO make random in future
        self.robot_position = [0, 0] # initial robot position

        """ общее количество точек на пути """
        self.totalPoint = len(self.points)

        assert self.totalPoint == 1 # one point with two coords

        self.state = []

        super().__init__(random=random)


    def initialize_episode(self, physics):
        self.points = np.copy(self.point_x_y, order='K')
        x, y = self.robot_position

        physics.named.data.qpos[0:3] = [x, y, 0.2] # Probably it's initial robot position x,y,z
        # or may be it a first point ?

        physics.named.data.qvel[:] = 0 # Probably it's initial robot velocity

        super().initialize_episode(physics)

        self.state = self.get_observation(physics)

    def get_observation(self, physics):
        # координаты центра колеса
        x, y, z = physics.named.data.geom_xpos['wheel_']
        # вектор скорости в абс системе координат
        v_x, v_y, v_z = physics.named.data.sensordata['wheel_vel']

        self.state =[
            x,y, # robot position
            v_x, v_y, # robot velocity
            self.points[0][0], self.points[0][1] # next point
        ]
        return self.state


    def get_termination(self, physics):
        if len(self.points) == 0 or physics.data.time > self.timeout or len(self.points) == self.achievedPoints :
            print("end episode at t = ", np.round(physics.data.time, 2), "\n")
            return 0.0

    def extract_robot_xy(self):
        return self.state[0],self.state[1]

    def extract_target_point_xy(self):
        return self.state[4],self.state[5]

    def get_reward(self, physics):

        robox_xy = torch.tensor(self.extract_robot_xy()).unsqueeze(0)

        target_point_xy = torch.tensor(self.extract_target_point_xy()).unsqueeze(0)

        l2_distance_to_target_point =  torch.cdist(robox_xy, target_point_xy, p=2)

        l2_distance_to_target_point = l2_distance_to_target_point[0] # remove batch dim

        reward = - torch.pow(l2_distance_to_target_point,2)
        return reward.item()


