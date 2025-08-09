import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
# 定义一些符号化的变量
STATE_DIM = 4
ACTION_DIM = 2
HIDDEN_DIM = 256
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.95
TAU = 1-GAMMA
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
SOFT_up = 10
# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmod = nn.Sigmoid()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action =2*self.sigmod(self.fc3(x)/1000)
        return action

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = self.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.position = 0
        self.max_values=np.asarray([0.3,3.,0.1,3.])
        self.min_values = np.asarray([-0.3,-3.,-0.1,-3.])
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[idx] for idx in batch])
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    def normalization(self, state):
        self.max_values = np.maximum(self.max_values , state)

        self.min_values = np.minimum(self.min_values , state)

        return self.max_values,self.min_values


# 定义DDPG算法
class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, tau, buffer_size, batch_size):
        self.actor = Actor(state_dim, action_dim, hidden_dim).cuda()
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim, hidden_dim).cuda()
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        self.softupT = SOFT_up
    def select_action(self, state):
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).cuda()
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        state = state/self.buffer.max_values
        #print(state)
        state = torch.FloatTensor(state).cuda()
        action = torch.FloatTensor(np.array(action)).cuda()
        reward = torch.FloatTensor(reward).unsqueeze(1).cuda()

        next_state = next_state/self.buffer.max_values
        next_state = torch.FloatTensor(next_state).cuda()
        done = torch.FloatTensor(done).unsqueeze(1).cuda()
        if state.any()>1: print('state bug')
        if next_state.any() > 1: print('next_state bug')
        # Critic loss
        next_action = self.actor_target(next_state)
        target_q = self.critic_target(next_state, next_action)
        expected_q = reward + (1.0 - done) * self.gamma * target_q
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, expected_q.detach())

        critic_loss = torch.clamp(critic_loss,-10**8,10**8)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        policy_loss = -self.critic(state, self.actor(state)).mean()
        policy_loss = torch.clamp(policy_loss,-10**5,10**5)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        if self.softupT ==0:
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            self.softupT = SOFT_up
        else:
            self.softupT =self.softupT-1

# 创建环境
from simp_ADRC_landing import Adrc_control_landing_env
env = Adrc_control_landing_env()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# 初始化DDPG
ddpg = DDPG(state_dim, action_dim, HIDDEN_DIM, LR_ACTOR, LR_CRITIC, GAMMA, TAU, BUFFER_SIZE, BATCH_SIZE)



# 训练DDPG
num_episodes = 300

R_L = []
A_L = []
Re_L = []
Rt_L = []
max_reward = -510
np.random.seed(2023022268)
for episode in range(num_episodes):
    A_L = []
    Re_L = []
    Rt_L = []
    state = env.reset()
    episode_reward = 0
    T = min(1, 5 / (episode % 1000 + 0.1))
    print('探索性', T)
    for t in range(8000):
        #print(state)

        action = np.array(ddpg.select_action(state))
        # 更改到范围
        # action[0] = action[0]+0.5
        # action[1] = action[1]*2.
        action = np.nan_to_num(action)
        action = np.clip(action, env.low_list, env.high_list)
        # 循环探索
        noise_a = np.random.uniform(T/10, T, size=2)
        noise_b = math.copysign(1, random.random())
        action = action + noise_a * noise_b
        action = np.clip(action, env.low_list, env.high_list)
        A_L.append(action)
        next_state, reward, done, _ = env.step(action)
        if t ==8000-1:
            done = 1
        ddpg.buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        Rt_L.append(episode_reward)
        Re_L.append(reward)
        if t%10 ==0:
            ddpg.update()

        if done == 1:
            break
    R_L.append(episode_reward)

    # if episode % 10 == 0:
    #     env.ADRC_control.show()
    #     env.save(episode, A_L, Re_L, Rt_L)

    if episode_reward >max_reward:
        #env.ADRC_control.show()
        env.save(episode, A_L, Re_L, Rt_L)

        torch.save(ddpg.actor.state_dict(), f= str(episode)+'actor0.pth')
        torch.save(ddpg.critic.state_dict(), f= str(episode)+'critic0.pth')
        max_reward=episode_reward

    R_L.append(episode_reward)

    print(f"Episode {episode}, Return: {episode_reward}")

print(R_L)
da = pd.DataFrame(R_L)
print(da)
da.to_excel('sim_DDPG_11.xlsx')