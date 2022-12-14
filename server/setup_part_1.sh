# обновляем пакеты и ядро
sudo apt update -y
sudo apt upgrade -y

sudo apt install git net-tools mc curl -y

# установка conda и PyTorch скомпилированный под cuda
touch anaconda.sh
curl https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh --output anaconda.sh
sha256sum anaconda.sh
bash anaconda.sh
cd $HOME
source ~/.bashrc

conda create --name=env
conda activate env

# установка драйвера
sudo apt install nvidia-driver-470 -y
reboot

conda activate env
sudo apt install nvidia-cuda-toolkit -y # sudo nvidia-smi -pm 1
export CUDA_PATH=/usr/lib/cuda # whereis cuda
source ~/.bashrc

sudo apt install python3-pip
pip install --upgrade pip setuptools wheel
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch-nightly -c nvidia
pip install matplotlib numpy tqdm tensorboard torchsummary geos shapely