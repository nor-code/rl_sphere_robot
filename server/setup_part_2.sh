sudo apt install nvidia-cuda-toolkit -y
export CUDA_PATH=/usr
source ~/.bashrc
nvidia-smi


sudo apt-get install libxslt1-dev zlib1g-dev libglew1.5 libglew-dev -y
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 libglfw3-dev -y

mkdir .mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.2.2/mujoco-2.2.2-linux-x86_64.tar.gz
tar -xzf mujoco-2.2.2-linux-x86_64.tar.gz
rm mujoco-2.2.2-linux-x86_64.tar.gz

export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PATH="$LD_LIBRARY_PATH:$PATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

pip install dm_control