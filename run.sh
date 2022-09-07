mkdir models

tensorboard --logdir=runs &

nohup python3 main.py --simu_number=1 --type_task=3 --algo=ddqn --trajectory=circle > out1.log &

nohup python3 main.py --simu_number=2 --type_task=3 --algo=ddqn --trajectory=curve > out2.log &

#scp root@95.143.188.75:/root/robot_rl/rl_sphere_robot/models/name.pt /home/nikita/simulation_sph_robot