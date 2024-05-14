import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

class ModifiedRewardWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = 0 if not done else info['reward_dist']
        return observation, reward, done, info

class BufferReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class ActionNormalizer(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        return action

class StateRetrieval(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(StateRetrieval, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class FlowNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(FlowNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softplus(self.fc3(x))
        return x

class ContinuousFlowNetwork:
    def __init__(self, state_dim, action_dim, hidden_dim, max_action, uniform_action_size, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.network = FlowNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.network_optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        self.state_retrieval = StateRetrieval(state_dim, action_dim).to(device)
        self.retrieval_optimizer = torch.optim.Adam(self.state_retrieval.parameters(), lr=3e-5)
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.uniform_action_size = uniform_action_size
        self.uniform_action = torch.Tensor(np.random.uniform(low=-max_action, high=max_action, size=(uniform_action_size, 2))).to(device)
        self.iterations = 0

    def select_action(self, state, use_max):
        sampled_actions = torch.Tensor(np.random.uniform(low=-self.max_action, high=self.max_action, size=(10000, 2))).to(device)
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).repeat(10000, 1).to(device)
        with torch.no_grad():
            edge_flow = self.network(state_tensor, sampled_actions).reshape(-1)
            action = sampled_actions[edge_flow.argmax()] if use_max else sampled_actions[Categorical(edge_flow.float()).sample(torch.Size([1]))[0]]
        return action.cpu().numpy().flatten()

    def update_uniform_action(self):
        self.uniform_action = torch.Tensor(np.random.uniform(low=-self.max_action, high=self.max_action, size=(self.uniform_action_size, 2))).to(device)
        return self.uniform_action

    def train(self, replay_buffer, frame_idx, batch_size=256, max_episode_steps=50, sample_flow_num=100):
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)
        state, next_state, action, reward, not_done = [torch.FloatTensor(arr).to(device) for arr in [state, next_state, action, reward, not_done]]

        with torch.no_grad():
            uniform_action = torch.Tensor(np.random.uniform(low=-self.max_action, high=self.max_action, size=(batch_size, max_episode_steps, sample_flow_num, 2))).to(device)
            current_state = next_state.repeat(1, 1, sample_flow_num).reshape(batch_size, max_episode_steps, sample_flow_num, -1)
            inflow_state = self.state_retrieval(current_state, uniform_action)
            inflow_state = torch.cat([inflow_state, state.view(batch_size, max_episode_steps, -1, state.shape[-1])], -2)
            uniform_action = torch.cat([uniform_action, action.view(batch_size, max_episode_steps, -1, action.shape[-1])], -2)
        edge_inflow = self.network(inflow_state, uniform_action).view(batch_size, max_episode_steps, -1)

        inflow = torch.log(torch.sum(torch.exp(edge_inflow), dim=-1) + 1.0)

        with torch.no_grad():
            uniform_action = torch.Tensor(np.random.uniform(low=-self.max_action, high=self.max_action, size=(batch_size, max_episode_steps, sample_flow_num, action.shape[-1]))).to(device)
            outflow_state = next_state.repeat(1, 1, sample_flow_num + 1).reshape(batch_size, max_episode_steps, sample_flow_num + 1, -1)
            last_action = torch.cat([action[:, 1:, :], torch.zeros(batch_size, 1, action.shape[-1]).to(device)], dim=1).view(batch_size, max_episode_steps, -1, action.shape[-1])
            uniform_action = torch.cat([uniform_action, last_action], dim=-2)
        edge_outflow = self.network(outflow_state, uniform_action).view(batch_size, max_episode_steps, -1)

        outflow = torch.log(torch.sum(torch.exp(edge_outflow), dim=-1) + 1.0)
        network_loss = F.mse_loss(inflow * not_done, outflow * not_done, reduction='none') + F.mse_loss(inflow * (1 - not_done), reward + 1.0, reduction='none')
        network_loss = torch.mean(torch.sum(network_loss, dim=1))
        print(network_loss)

        self.network_optimizer.zero_grad()
        network_loss.backward()
        self.network_optimizer.step()

        if frame_idx % 5 == 0:
            pre_state = self.state_retrieval(next_state, action)
            retrieval_loss = F.mse_loss(pre_state, state)
            print(retrieval_loss)
            self.retrieval_optimizer.zero_grad()
            retrieval_loss.backward()
            self.retrieval_optimizer.step()

writer = SummaryWriter(log_dir="runs/CFN_Reacher_" + current_time)

max_steps = 50
env = ModifiedRewardWrapper(gym.make('Reacher-v2'))
test_env = ModifiedRewardWrapper(gym.make('Reacher-v2'))

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
max_action = float(env.action_space.high[0])
hidden_dim = 256
uniform_action_size = 2000

policy = ContinuousFlowNetwork(state_dim, action_dim, hidden_dim, max_action, uniform_action_size)
policy.state_retrieval.load_state_dict(torch.load('retrieval_reacher_sparse.pkl'))

replay_buffer = BufferReplay(2000)

total_frames = 2000
initial_timesteps = 150
current_frame = 0
rewards_list = []
test_rewards_list = []
episode_indices = []
batch_size = 128
test_intervals = 0
expl_noise = 0.4
flow_samples = 99
repeat_episodes = 5
sample_episodes = 1000

done_tensor = torch.zeros(batch_size, max_steps).to(device)
done_tensor[:, -1] = 1.0

def adjust_reward(reward):
    low, high = -1.0, 0.0
    return (reward - low) / (high - low)

while current_frame < total_frames:
    state = env.reset()
    episode_reward = 0

    states, actions, rewards, next_states, dones = [], [], [], [], []

    for step in range(max_steps):
        with torch.no_grad():
            action = policy.select_action(state, False)
        next_state, reward, done, _ = env.step(action)
        done_flag = float(not done)

        states.append(state)
        actions.append(action)
        rewards.append(adjust_reward(reward))
        next_states.append(next_state)
        dones.append(done_flag)

        state = next_state
        episode_reward += reward

        if done:
            current_frame += 1
            replay_buffer.add(states, actions, rewards, next_states, dones)
            break

        if current_frame >= initial_timesteps and step % 7 == 0:
            policy.train(replay_buffer, current_frame, batch_size, max_steps, flow_samples)

    if current_frame >= initial_timesteps and current_frame % 10 == 0:
        print(current_frame)
        test_intervals += 1
        avg_test_reward = 0

        for _ in range(repeat_episodes):
            test_state = test_env.reset()
            test_episode_reward = 0

            for _ in range(max_steps):
                test_action = policy.select_action(test_state, True)
                test_state, test_reward, test_done, _ = test_env.step(test_action)
                test_episode_reward += test_reward
                if test_done:
                    break

            avg_test_reward += test_episode_reward

        torch.save(policy.network.state_dict(), f"runs/CFN_Reacher_{current_time}.pkl")
        writer.add_scalar("MaxTestReward", avg_test_reward / repeat_episodes, global_step=current_frame * max_steps)
