# обновляем пакеты и ядро
sudo apt update -y
sudo apt upgrade -y

# установка git
sudo apt install git -y

# установка драйвера
sudo apt install nvidia-driver-455 -y

# установка conda
sudo apt install curl -y
touch anaconda.sh
curl https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh --output anaconda.sh
sha256sum anaconda.sh
bash anaconda.sh
cd $HOME
conda install pytorch torchvision cudatoolkit=11 -c pytorch-nightly
pip install matplotlib
pip install numpy
pip install tqdm
pip install Pillow
pip install dm_control

# акстивируем анаконду
bash ~/.bashrc

reboot