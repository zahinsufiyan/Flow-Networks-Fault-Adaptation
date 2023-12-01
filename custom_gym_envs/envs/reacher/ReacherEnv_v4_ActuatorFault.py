import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
from pathlib import Path


class ReacherEnvV4(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        # modification here: start
        self.hostname = os.uname()[1]
        self.localhosts = ["melco", "Legion", "amii", "mehran", "Sheila"]
        self.computecanada = not any(host in self.hostname for host in self.localhosts)
        home = str(Path.home())
        if self.computecanada:
            filepath = home + "/scratch/openai/custom_gym_envs/envs/ant/xml/AntEnv_v0_Normal.xml"
        else:
            filepath = home + "/TD3/custom_gym_envs/envs/reacher/xml/ReacherEnv_v4_ActuatorFault.xml"
        # modification here: end

        mujoco_env.MujocoEnv.__init__(self, filepath, 2)
        utils.EzPickle.__init__(self)

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )
