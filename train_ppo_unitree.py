import numpy as np
import random
import os
from gym import Env, spaces
from humanoid_tracking_env import UnitreeTrackingEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from collections import defaultdict
import torch
import torch.nn as nn
from typing import Dict, Any
from typing import Optional

class CustomPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_value_loss(self, value_pred, value_target):
        """
        手动裁剪 value_loss，限制 value_pred 和 value_target 之间的差异范围。
        """
        # 计算delta，value_pred - value_target
        delta = value_pred - value_target
        # 对delta进行裁剪，限制其变化范围
        clipped_delta = torch.clamp(delta, min=-0.2, max=0.2)
        # 计算裁剪后的值误差
        value_loss = (clipped_delta ** 2).mean()
        return value_loss

    def _train(self, gradient_steps: int, batch_size: int) -> Dict[str, float]:
        """
        训练步骤，重写以使用自定义的 `compute_value_loss`
        """
        # 默认使用 `_train` 来训练模型, 但在内部会计算 value_loss
        # 使用自定义计算的 value_loss，而不是 Stable-Baselines3 默认的实现
        # 调用父类的 `_train` 方法来完成训练
        return super()._train(gradient_steps, batch_size)

# ✅ Gym Wrapper with support for vectorized environments
class UnitreeGymWrapper(Env):
    def __init__(self, render=False, num_envs=1):
        super(UnitreeGymWrapper, self).__init__()
        self.env = UnitreeTrackingEnv(render=render, num_envs=num_envs)
        self.upper_dof_indices = self.env.upper_body_dof_indices
        self.num_dof = len(self.upper_dof_indices)
        self.num_envs = num_envs

        obs_dim = self.num_dof * 2  # joint positions + goals
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_dof,), dtype=np.float32)

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self, seed=None, options=None, **kwargs):
        if seed is not None:
            self.seed(seed)
        self.env.reset()
        return self._get_obs(), {}

    def step(self, action):
        if len(action.shape) == 1:
            action = np.expand_dims(action, axis=0)
        
        action = np.clip(action, -1.0, 1.0)
        scaled_action = action * 0.5
        
        obs_arr, reward_arr, _, _ = self.env.step(scaled_action)
        
        if self.num_envs == 1:
            reward = float(reward_arr[0])
            return self._get_obs(), reward, False, False, {}
        else:
            dones = np.zeros(self.num_envs, dtype=bool)
            infos = [{} for _ in range(self.num_envs)]
            return self._get_obs(), reward_arr, dones, dones, infos

    def _get_obs(self):
        joint_pos = self.env.default_dof_positions[self.upper_dof_indices]
        goal = self.env.motion_goals[0]
        
        if self.num_envs == 1:
            return np.concatenate([joint_pos, goal], axis=0).astype(np.float32)
        else:
            obs = []
            for i in range(self.num_envs):
                obs.append(np.concatenate([joint_pos[i], goal], axis=0))
            return np.array(obs, dtype=np.float32)

    def render(self, mode='human'):
        self.env.render_step()

    def close(self):
        self.env.close()

def lr_schedule(progress):
    initial_lr = 3e-3
    min_lr = 5e-3
    return initial_lr - progress * (initial_lr - min_lr)

class RewardLoggingCallback(BaseCallback):
    def __init__(self, window_size=100,  log_dir='./logs', verbose=0, text_name = 'rewards.txt'):
        """
        :param window_size: 每多少步计算一次奖励的平均值
        :param verbose: 日志输出的详细程度
        """
        super().__init__(verbose)
        self.window_size = window_size  # 设置窗口大小（计算平均奖励的步数）
        self.reward_buffer = []  # 用于存储奖励的缓冲区
        self.log_dir = log_dir
        self.cnt = 0
        os.makedirs(self.log_dir, exist_ok=True)
        self.reward_file = open(os.path.join(self.log_dir, text_name), 'w')
    
    def _on_step(self) -> bool:
        # 获取当前的奖励
        reward = self.locals.get("rewards", None)
        
        if reward is not None:
            # 将当前奖励添加到奖励缓冲区
            self.reward_buffer.extend(reward)
            
            # 如果缓冲区大小超过了窗口大小，移除最旧的奖励
            if len(self.reward_buffer) > self.window_size:
                self.reward_buffer = self.reward_buffer[-self.window_size:]
            
            # 计算并输出当前窗口内的平均奖励
            avg_reward = np.mean(self.reward_buffer)
            self.reward_file.write(f"{self.n_calls},{avg_reward:.4f}\n")
            if self.n_calls / self.window_size > self.cnt:
                print(f"Iteration {self.n_calls}, Average Reward (last {self.window_size} steps): {avg_reward:.4f}")
                self.cnt = self.cnt + 1
        
        return True


class AdaptiveLRCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.last_kl = 0.0
        self.lr_upper_bound = 1e-3
        self.lr_lower_bound = 1e-5

    def _on_step(self) -> bool:
        kl = self.locals.get('kl', 0.0)
        current_lr = self.model.learning_rate

        # if kl < 0.01 and current_lr < self.lr_upper_bound:
        #     new_lr = min(current_lr * 1.5, self.lr_upper_bound)
        #     self.model.learning_rate = new_lr
        #     for param_group in self.model.policy.optimizer.param_groups:
        #         param_group['lr'] = new_lr
        #     print(f"⬆️ 上调学习率: {current_lr:.1e}→{new_lr:.1e} (KL={kl:.4f})")
        
        # elif kl > 0.05 and current_lr > self.lr_lower_bound:
        #     new_lr = max(current_lr * 0.7, self.lr_lower_bound)
        #     self.model.learning_rate = new_lr
        #     for param_group in self.model.policy.optimizer.param_groups:
        #         param_group['lr'] = new_lr
        #     print(f"⬇️ 下调学习率: {current_lr:.1e}→{new_lr:.1e} (KL={kl:.4f})")
        
        return True

# 创建环境的函数
def make_env(rank: int = 0, seed: Optional[int] = None, render: bool = False):
    def _init():
        env = UnitreeGymWrapper(render=render, num_envs=1)
        if seed is not None:
            env.seed(seed + rank)
        return env
    return _init

def reset_value_network(model: PPO) -> None:
    # 获取价值函数网络
    value_net = model.policy.value_net
    
    # 检查是否为单个 Linear 层，如果是，则初始化权重
    if isinstance(value_net, nn.Linear):
        nn.init.orthogonal_(value_net.weight, gain=0.1)
        if value_net.bias is not None:
            value_net.bias.data.zero_()
    else:
        # 如果是一个包含多个层的网络（如 nn.Sequential），则可以遍历每一层
        for layer in value_net:
            if isinstance(layer, nn.Linear):
                # 正交初始化（增益调小以适应归一化后的Reward尺度）
                nn.init.orthogonal_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    layer.bias.data.zero_()

def calibrate_policy(model, env, steps=100):
    """用随机动作运行环境，稳定网络初始输出"""
    obs = env.reset()
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, done, _ = env.step(action)
        if done.any():  # 检查是否有任意环境结束
            obs = env.reset()
    # 重置优化器状态
    model.policy.optimizer.state = defaultdict(dict)
    print("✅ 初始策略已校准")

# 主函数
def main():
    os.makedirs("checkpoints", exist_ok=True)

    
    num_envs = 8 
    seed = 42

    # # 使用SubprocVecEnv
    env = SubprocVecEnv([make_env(i, seed) for i in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 设置PPO超参数
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[256, 256]),
        ortho_init=False,
        activation_fn=torch.nn.ReLU
    )

    model = CustomPPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        ent_coef=0.0005,
        learning_rate=lr_schedule,
        n_steps=4096,  # 增加n_steps以收集更多经验
        batch_size=64,  # 减小batch_size以适应更多环境
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=0.25,
        clip_range=0.15,
        vf_coef=1.0,
        device="auto",
        tensorboard_log="./ppo_tensorboard/"
    )
    reset_value_network(model)

    # # Checkpoint回调
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,  # 更频繁地保存检查点
        save_path="./checkpoints/",
        name_prefix="ppo_unitree",
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    adaptive_lr_callback = AdaptiveLRCallback(verbose=1)
    reward_logging_callback1 = RewardLoggingCallback(window_size=1000, verbose=1, text_name='reward1.txt')

    # 开始训练
    total_timesteps = 3_000_000  # 总步数
    step1 = 100_000
    print(f"🚀 开始训练，总步数：{total_timesteps}，并行环境数：{num_envs}")
    calibrate_policy(model, env)
    model.learn(
        total_timesteps=step1,
        callback=[reward_logging_callback1, checkpoint_callback],
        tb_log_name="ppo_unitree"
    )

    env.save("vec_normalize_no_reward_norm.pkl")

    env = VecNormalize.load("vec_normalize_no_reward_norm.pkl", env)
    env.norm_reward_kwargs = dict(clip_reward=5.0, epsilon=1e-5)
    reward_logging_callback2 = RewardLoggingCallback(window_size=1000, verbose=1, text_name='reward2.txt')

    #  重置价值函数
    reset_value_network(model)

    # 过渡训练
    model.learn(total_timesteps=100000, callback=[reward_logging_callback2, checkpoint_callback], tb_log_name="ppo_unitree")

    env.norm_reward_kwargs = dict(clip_reward=10.0)  # 放宽限制
    reward_logging_callback3 = RewardLoggingCallback(window_size=1000, verbose=1, text_name='reward3.txt')
    model.learn(total_timesteps=25000000, callback=[reward_logging_callback3, checkpoint_callback], tb_log_name="ppo_unitree")
    # 保存最终模型和归一化器1
    model.save("ppo_unitree_policy_final")
    env.save("vec_normalize_final.pkl")
    print("✅ 训练完成，模型和归一化器已保存")

if __name__ == "__main__":
    main()

# def main():
#     os.makedirs("checkpoints", exist_ok=True)
#     num_envs = 8 
#     seed = 42

#     # 创建基础环境
#     env = SubprocVecEnv([make_env(i, seed) for i in range(num_envs)])
    
#     # 加载之前保存的环境参数
#     env = VecNormalize.load("ppo_unitree_vecnormalize_11737856_steps.pkl", venv=env)
#     env.training = True  # 确保继续训练模式
#     env.norm_reward = True  # 保持与之前一致的奖励归一化

#     # 加载模型
#     model = PPO.load("ppo_unitree_11737856_steps", env=env)
    
#     # 确保关键训练参数一致
#     model.n_steps = 4096
#     model.batch_size = 64
#     model.n_epochs = 20
#     model.gamma = 0.99
#     model.gae_lambda = 0.95
    
#     # 检查点回调
#     checkpoint_callback = CheckpointCallback(
#         save_freq=100_000,
#         save_path="./checkpoints/",
#         name_prefix="ppo_unitree",
#         save_replay_buffer=True,
#         save_vecnormalize=True
#     )

#     # 奖励日志回调
#     reward_logging_callback = RewardLoggingCallback(
#         window_size=1000, 
#         verbose=1, 
#         text_name='reward_continue.txt'  # 使用新文件名避免覆盖
#     )

#     # 放宽奖励裁剪限制
#     env.norm_reward_kwargs = dict(clip_reward=10.0)

#     # 继续训练
#     model.learn(
#         total_timesteps=5_000_000,
#         callback=[reward_logging_callback, checkpoint_callback],
#         tb_log_name="ppo_unitree_continue",  # 使用新日志名
#         reset_num_timesteps=False  # 关键！不重置时间步计数
#     )

#     # 保存最终模型和归一化器
#     model.save("ppo_unitree_policy_final_continued")
#     env.save("vec_normalize_final_continued.pkl")
#     print("✅ 训练完成，模型和归一化器已保存")

# if __name__ == "__main__":
#     main()