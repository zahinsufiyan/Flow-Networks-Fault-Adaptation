## CFlowNets: Continuous Control with Generative Flow Networks for Fault Adaptation

CFlowNets Paper:
[![arXiv](https://img.shields.io/badge/arXiv-2303.02430-b31b1b.svg)](https://arxiv.org/abs/2303.02430)


## Prerequisites

#### Install dependencies

See `requirments.txt` file for more information about how to install the dependencies.

#### Install mujoco 210
1. [Download](https://mujoco.org/) and [install](https://github.com/openai/mujoco-py#install-mujoco) MuJoCo.
or,

```bash
cd ~/download/
wget -c https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
cd ~/.mujoco
tar -zxvf ~/download/mujoco210-linux-x86_64.tar.gz mujoco210


echo "export LD_LIBRARY_PATH=\$HOME/.mujoco/mujoco210/bin:\$LD_LIBRARY_PATH" >> ~/.profile
echo "export MUJOCO_PY_MUJOCO_PATH=\"\$HOME/.mujoco/mujoco210\"" >> ~/.profile
echo "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:/usr/lib/nvidia\"" >> ~/.profile
pip install -U 'mujoco-py<2.2,>=2.1'
```


## Usage

#### Train continuous flow network of different environments.
```bash
# Reacher
python CFN_Reacher.py
```

