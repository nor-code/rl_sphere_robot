import threading
import json

import numpy as np
import torch
from dm_control.rl.control import PhysicsError
from dm_env import StepType

from agent.ddpg import DeepDeterministicPolicyGradient
from robot.enviroment import get_state_dim, make_env

c = threading.Condition()

top_result_map = {}


class SimulationResult:

    def __init__(self, time, required, result):
        self.time = time
        self.required = required
        self.result = result

    def to_json(self):
        return {"time": self.time,
                "x_req": self.required[0],
                "y_req": self.required[1],
                "res_x": self.result[:, 0].tolist(),
                "res_y": self.result[:, 1].tolist()}


class ThreadSimulation(threading.Thread):
    def __init__(self, i, j):
        threading.Thread.__init__(self)
        self.i = i
        self.j = j

    def run(self):
        global top_result_map
        type_task = 11
        timeout = 110
        state_dim = get_state_dim(type_task)

        agent = DeepDeterministicPolicyGradient(state_dim,
                                                device='cpu',
                                                act_dim=2,
                                                replay_buffer=None,
                                                batch_size=1,
                                                gamma=0.99,
                                                writer=None)

        agent.policy.load_state_dict(
            torch.load('../models/30_october/task_11/ddpg_policy_1_2.pt', map_location=torch.device('cpu'))
        )
        agent.policy.eval()

        agent.qf.load_state_dict(
            torch.load('../models/30_october/task_11/ddpg_Q_1_2.pt', map_location=torch.device('cpu'))
        )
        agent.qf.eval()
        env, x_y = make_env(episode_timeout=timeout, type_task=type_task,
                            trajectory='random', begin_index_=5, count_substeps=3)
        pos = np.array([[0, 0]])
        observation = env.reset().observation

        while env.physics.data.time < timeout:
            action = agent.get_action([observation])
            try:
                time_step = env.step(action)
            except PhysicsError:
                print("physicx error  time = ", prev_time)
                break

            if time_step.reward is None or time_step.step_type == StepType.LAST:
                break

            prev_time = env.physics.data.time

            x, y, _ = env.physics.named.data.geom_xpos['wheel_']

            pos = np.append(pos, [[x, y]], axis=0)

            observation = time_step.observation

        c.acquire()
        top_result_map["_" + str(self.i) + str(self.j)] = SimulationResult(env.physics.data.time,
                                                                           required=x_y,
                                                                           result=pos)
        c.release()


ci, cj = 5, 8
threads = []

for i in range(ci):
    for j in range(cj):
        th_ij = ThreadSimulation(i, j)
        th_ij.start()
        threads.append(th_ij)

for th in threads:
    th.join()

all_res_list = list(top_result_map.values())
all_res_list.sort(key=lambda r: r.time, reverse=True)

top_10 = {}
for i in range(10):
    top_10[i] = all_res_list[i].to_json()

with open('./top_10.json', 'w') as outfile:
    outfile.write(json.dumps(top_10))