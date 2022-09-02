tensorboard --logdir=runs # port = 6006
nohup python3 main.py --simu_number=1 --type_task=1 > out1.log &
nohup python3 main.py --simu_number=2 --type_task=2 > out2.log &