## CFlowNets: Continuous Control with Generative Flow Networks for Fault Adaptation

In this research, we investigate the efficacy of generative flow networks in robotic environments, specifically in machine fault adaptation. The experimentations were done in a simulated robotic environment Reacher-v2. This environment was manipulated and modified to introduce four distinct fault environments which are reduced range of motion, increased damping, actuator damage, and structural damage. The empirical evaluation of this research indicates that continuous generative flow networks indeed have the capability to add adaptive behaviors to robots. Furthermore, the comparative study with state-of-the-art reinforcement learning algorithms also provides some key insights into the performance of CFlowNets.

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
python CFlowNets.py
```

#### Create a Malfunctioning Reacher

To create a malfunctioning Ant, the following steps must be taken:
* xml file
  * within the custom_gym_envs/envs/reacher/xml folder, copy and paste ReacherEnv_v0_Normal.xml, editing the version of the .xml filename
  * add malfunctions
* python file
  * within the custom_gym_envs/envs/reacher folder, copy and paste ReacherEnv_v0_Normal.py, editing the version of the .py filename
  * update the class name with the version number
  * update the filepath instance variable for the class with the path to the appropriate xml file
* init file
  * add new environment to custom_gym_envs/__init__.py


Do not edit:
* custom_gym_envs/envs/reacher/ReacherEnv_v0_Normal.py
* custom_gym_envs/envs/reacher/xml/ReacherEnv_v0_Normal.xml

## Fault Scenarios
We introduced four different fault scenarios to evaluate the adaptive capabilities of CFlowNets:

Reduced Range of Motion:

Simulated by adjusting the <joint> element’s range attribute within the Reacher-v2’s XML file from "-3.0 3.0" radians to "-1.0 1.0" radians.
Increased Damping:

Simulated by increasing the damping attribute value in the <joint> element from "1" to "5".
Actuator Damage:

Simulated by modifying the gear attribute of the <motor> element within the XML file from "200.0" to "100.0".
Structural Damage:

Simulated by bending the link1 of the arm in the XML configuration.

## Reinforcement Learning Implementations for Comparative analysis

Hyper-parameters can be modified with different arguments to main.py. We include an implementation of DDPG (DDPG.py) and TD3 (TD3.py), which is used for easy comparison with CFlowNets.
