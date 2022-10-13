mkdir models

tensorboard --logdir=runs &

nohup python3 main.py --simu_number=1 --type_task=4 --agent_type=ddqn --trajectory=circle --buffer_size=6000000 --batch_size=4096 --refresh_target=400 --total_steps=20000 --decay_steps=5000 > out1.log &

nohup python3 main.py --simu_number=2 --type_task=4 --agent_type=ddpg --trajectory=circle --buffer_size=6000000 --batch_size=4096 --total_steps=20000 --decay_steps=5000 > out2.log &

#scp root@95.143.188.75:/root/robot_rl/rl_sphere_robot/models/name.pt /home/nikita/simulation_sph_robot
#scp root@95.143.188.113:/root/rl_sphere_robot/models/ddpg_Q_1_15.pt /home/nikita/simulation_sph_robot/models
#scp root@95.143.188.113:/root/rl_sphere_robot/models/ddpg_policy_1_15.pt /home/nikita/simulation_sph_robot/models