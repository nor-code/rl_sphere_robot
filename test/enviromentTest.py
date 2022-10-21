import sys
import os
import inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
#sys.path.insert(0,parentdir+os.sep + "src" + os.sep + "generated")

import unittest
from robot.enviroment import make_env
import numpy as np


class TestGrpc(unittest.TestCase):

    def testBasic(self):
        env, x_y = make_env(episode_timeout=30, type_task=8, trajectory='one_point', begin_index_=0)
        #print(env)
        x = env.reset() # type(x) = dm_env._environment.TimeStep
        print(x)
        print(len(x.observation))
        print(x.observation)
        print(x.reward)
        print(env.task.points)

        # Taking random actions
        for i in range(10):
            action = [np.random.uniform(-0.975, 0.975, size=1)[0], np.random.uniform(0.26, 0.6, size=1)[0]]  # only random
            timestep = env.step(action)
            print(timestep)


if __name__ == '__main__':
    unittest.main()