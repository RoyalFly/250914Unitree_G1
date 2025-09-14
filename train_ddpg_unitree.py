from humanoid_tracking_env_ddpg import UnitreeTrackingEnv
import torch
import numpy as np
import random
from collections import deque
from ddpg import DDPG
import os

# 设置设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 创建环境（8个并行环境）
    num_envs = 8
    env = UnitreeTrackingEnv(render=True, num_envs=num_envs)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"最大动作值: {max_action}")

    # 创建DDPG代理
    agent = DDPG(
        state_dim, 
        action_dim, 
        max_action,
        actor_lr=1e-5,  # 原为 1e-4（降低学习率）
        critic_lr=1e-4, # 原为 1e-3
        gamma=0.99,
        tau=0.001,      # 原为 0.005（减缓目标网络更新）
        batch_size=256   # 原为 64（增加稳定性）
    )

    # 训练参数
    num_episodes = 1000
    max_timesteps = 200
    print_interval = 50
    total_steps = 0
    reward_history = []

    for episode in range(num_episodes):
        agent.noise_scale = max(0.2, agent.noise_scale)

        if episode > 10:
            agent.tau = 0.001  # 后期减慢更新

        states = env.reset()  # 形状: (num_envs, state_dim)
        episode_rewards = np.zeros(num_envs)  # 每个环境的单独奖励
        dones = np.zeros(num_envs, dtype=bool)  # 每个环境是否结束

        for t in range(max_timesteps):
            # 获取所有环境的动作（批量处理）
            with torch.no_grad():
                actions = agent.select_action(torch.FloatTensor(states).to(device))
                actions = actions.cpu().numpy()
            agent.total_steps += num_envs
            
            # 执行所有环境的动作
            next_states, rewards, dones, _ = env.step(actions)  # 所有返回值的形状均为 (num_envs, ...)

            # 存储经验到回放缓冲区（逐个环境处理）
            for i in range(num_envs):
                agent.replay_buffer.append(
                    (states[i], actions[i], rewards[i], next_states[i], dones[i])
                )

            # 更新网络（使用所有环境的经验）
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()

            states = next_states
            episode_rewards += rewards  # 累加每个环境的奖励
            total_steps += num_envs  # 总步数 = 环境数 × 每步

            # 打印平均奖励（每隔print_interval步）
            if total_steps % print_interval == 0:
                avg_reward = np.mean(episode_rewards)
                print(f"Steps: {total_steps}, Episode: {episode + 1}, Avg Reward: {avg_reward:.2f}")

            # 如果所有环境都结束，提前终止
            if np.all(dones):
                break

        # 记录本回合的平均奖励
        avg_episode_reward = np.mean(episode_rewards)
        reward_history.append(avg_episode_reward)
        print(f"Episode {episode + 1} finished. Avg Reward: {avg_episode_reward:.2f}")

    # 训练结束，输出最后10回合的平均奖励
    avg_reward = np.mean(reward_history[-10:])
    save_dir = "saved_models"
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"目录已创建：{os.path.abspath(save_dir)}")  # 打印绝对路径确认
    except Exception as e:
        print(f"创建目录失败：{str(e)}")
        save_dir = "."  # 如果失败则保存到当前目录
    torch.save(agent.actor.state_dict(), f"{save_dir}/ddpg_actor_final.pt")
    print(f"Training completed. Avg Reward (last 10 episodes): {avg_reward:.2f}")

if __name__ == "__main__":
    main()
