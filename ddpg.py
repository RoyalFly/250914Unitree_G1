import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用PyTorch定义Actor（策略网络）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # 输出限制在[-1, 1]之间
        return action * self.max_action


# 使用PyTorch定义Critic（Q值网络）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))  # 拼接状态和动作
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


# DDPG主算法类
class DDPG:
    def __init__(self, state_dim, action_dim, max_action, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, batch_size=64):
        # 初始化演员和评论家网络
        self.max_action = max_action
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.total_steps = 0
        self.exploration_steps = 50000
        
        # 初始化目标网络（用于稳定训练）
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        
        # 将目标网络的权重设置为演员和评论家的当前权重
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 超参数
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.grad_clip = 0.5  # 更严格的梯度裁剪

        self.noise_scale = 0.5  # 初始噪声强度
        self.noise_decay = 0.99
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=100000)
    

    def update(self):
        # 1. 优先采样成功经验 + 随机经验（如果存在足够成功经验）
        successful_transitions = [t for t in self.replay_buffer if t[2] > 1.0]
        if len(successful_transitions) > self.batch_size // 2:
            # 混合采样：50%成功经验 + 50%随机经验
            batch = random.sample(successful_transitions, self.batch_size // 2) + \
                    random.sample(self.replay_buffer, self.batch_size // 2)
        else:
            # 普通随机采样
            batch = random.sample(self.replay_buffer, self.batch_size)
        
        # 2. 将batch转换为numpy数组（替换原有的sample_from_replay_buffer调用）
        states = np.stack([t[0] for t in batch])
        actions = np.stack([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.stack([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        # 3. 转换为PyTorch Tensor并移至设备
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)  # (batch,) -> (batch,1)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # 4. 更新Critic网络
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        
        # 5. 更新Actor网络
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        
        # 6. 软更新目标网络
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        
        # 7. 动态调整学习率（可选）
        if self.total_steps % 1000 == 0:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] *= 0.95  # 每1000步衰减5%
            print(f"学习率更新: {param_group['lr']:.1e}")

    def soft_update(self, target, source):
        """ 软更新目标网络，更新比例为tau """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def sample_from_replay_buffer(self):
        weights = np.array([t[2] for t in self.replay_buffer])  # 按reward加权
        weights = (weights - weights.min() + 1e-5)  # 确保非负
        probs = weights / weights.sum()
        
        indices = np.random.choice(
            len(self.replay_buffer), 
            size=self.batch_size, 
            p=probs
        )
        batch = [self.replay_buffer[i] for i in indices]
        return zip(*batch)

    def store_transition(self, state, action, reward, next_state, done):
        """ 存储过渡（状态，动作，奖励，下一个状态，是否完成） """
        transition = (state.copy(), 
                    action.copy(), 
                    float(reward), 
                    next_state.copy(), 
                    bool(done))
        self.replay_buffer.append(transition)

    def select_action(self, state):
        with torch.no_grad():
            action = self.actor(state)
            if self.total_steps < self.exploration_steps:  # 前1万步保持较强探索
                noise = self.noise_scale * torch.randn_like(action)
                return (action + noise).clamp(-self.max_action, self.max_action)
            return action  # 后期减少探索

