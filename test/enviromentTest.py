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
import time
import tqdm

class TestGrpc(unittest.TestCase):

    @unittest.skip
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

    @unittest.skip
    def testSpeed(self):

        env, x_y = make_env(episode_timeout=5000, type_task=8, trajectory='one_point', begin_index_=0)
        s =time.time()

        for i in tqdm.tqdm(range(1000)):
            action = [np.random.uniform(-0.975, 0.975, size=1)[0], np.random.uniform(0.26, 0.6, size=1)[0]]  # only random
            timestep = env.step(action)
        l = time.time() - s

        print("real time", l, "env time ",env.physics.time(), "ratio", env.physics.time()/l )

    def testAction(self):
        env, x_y = make_env(episode_timeout=5000, type_task=8, trajectory='one_point', begin_index_=0)
        action = [0, 0]  # no Action
        env.step(action) # Initial position setup
        x0, y0, _ = env.physics.named.data.geom_xpos['wheel_']

        for i in tqdm.tqdm(range(100)):
            env.step(action)
        x1, y1 , _ = env.physics.named.data.geom_xpos['wheel_']
        self.assertAlmostEqual(x0 ,x1, 2, "X")
        self.assertAlmostEqual(y0, y1, 2, "Y")
        self.assertEquals(sum(env.physics.named.data.qvel), 0, "Velocity must be zero")








if __name__ == '__main__':
    unittest.main()