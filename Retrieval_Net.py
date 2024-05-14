import gym
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import csv
import GPUtil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('gpu_usage_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Algorithm', 'Frame_idx', 'GPU Load (%)', 'GPU Memory (MB)'])

class RewardModifier(gym.Wrapper):
    def _modify_reward(self, reward, done):
        adjusted_reward = 0
        if done and reward >= -0.09:
            adjusted_reward = reward + 0.2
        return adjusted_reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = 0 if not done else info['reward_dist']
        return observation, reward, done, info

class RetrievalNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RetrievalNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, state_dim)

    def forward(self, state, action):
        combined = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class RetrievalTrainer:
    def __init__(
        self, state_dim, action_dim, max_action, 
        discount=0.99, tau=0.005, policy_noise=0.2, 
        noise_clip=0.5, policy_freq=2
    ):
        self.retrieval_net = RetrievalNetwork(state_dim, action_dim).to(device)
        self.retrieval_net_target = copy.deepcopy(self.retrieval_net)
        self.optimizer = torch.optim.Adam(self.retrieval_net.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.iterations = 0

    def train(self, replay_buffer, batch_size=256):
        self.iterations += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        predicted_state = self.retrieval_net(next_state, action)

        loss = F.mse_loss(predicted_state, state)
        print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.retrieval_net.state_dict(), filename + "_retrieval")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

max_steps_per_episode = 50
env = RewardModifier(gym.make('Reacher-v2'))
test_env = RewardModifier(gym.make('Reacher-v2'))

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
max_action = float(env.action_space.high[0])

trainer = RetrievalTrainer(state_dim, action_dim, max_action)

replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

total_frames = 1000000
initial_timesteps = 1000

current_frame = 0
all_rewards = []
test_rewards = []
batch_size = 256
test_interval = 0
exploration_noise = 0.1
episode_reward = 0
episode_steps = 0
episode_count = 0

state, done = env.reset(), False

while current_frame < total_frames:
    episode_steps += 1
    action = env.action_space.sample()

    next_state, reward, done, _ = env.step(action)
    done_flag = float(done) if episode_steps < max_steps_per_episode else 0
    replay_buffer.add(state, action, next_state, reward, done_flag)

    state = next_state
    episode_reward += reward

    if current_frame >= initial_timesteps:
        trainer.train(replay_buffer, batch_size)

    if current_frame % 10000 == 0 and current_frame > 0:
        gpu = GPUtil.getGPUs()[0]
        with open('gpu_usage_log.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["RetrievalTrainer", current_frame, gpu.load * 100, gpu.memoryUsed])

    if current_frame >= initial_timesteps and current_frame % 10000 == 0:
        torch.save(trainer.retrieval_net.state_dict(), 'retrieval_reacher_model.pkl')

    if done:
        state, done = env.reset(), False
        episode_reward = 0
        episode_steps = 0
        episode_count += 1

    current_frame += 1
