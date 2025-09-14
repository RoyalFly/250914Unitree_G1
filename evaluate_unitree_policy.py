from train_ppo_unitree import UnitreeGymWrapper  # 确保路径正确
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def evaluate_policy(model_path="ppo_unitree_policy_final.zip", num_trials=5, threshold=0.05, render=False):
    # 创建环境和加载模型
    env = DummyVecEnv([lambda: UnitreeGymWrapper(render=render)])
    model = PPO.load(model_path)

    success_count = 0
    all_errors = []
    all_rewards = []  # List to store rewards for each trial

    for trial in range(num_trials):
        obs = env.reset()
        trial_errors = []
        trial_rewards = []  # List to store rewards for this trial

        for step in range(300):  # 每个试验评估300步
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

            # 记录奖励
            trial_rewards.append(reward)

            # 直接解包出底层 env
            base_env = env.envs[0]                  # UnitreeGymWrapper
            sim_env = base_env.env                  # UnitreeTrackingEnv

            q_actual = sim_env.get_current_upperbody_pos()
            q_target = sim_env.motion_goals[0]      # 当前目标位姿

            error = np.linalg.norm(q_actual - q_target)
            trial_errors.append(error)

            if render:
                sim_env.render_step()

        all_errors.append(trial_errors)
        all_rewards.append(trial_rewards)  # Store rewards for the trial
        final_error = trial_errors[-1]
        print(f"[Trial {trial+1}] Final L2 Error: {final_error:.4f}")

        if final_error < threshold:
            success_count += 1

    success_rate = success_count / num_trials * 100
    print(f"\n✅ Success Rate (< {threshold} rad error): {success_rate:.2f}%")

    # 误差曲线绘图
    for i, err in enumerate(all_errors):
        plt.plot(err, label=f"Trial {i+1}")
    plt.title("Tracking Error Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("L2 Error")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 奖励曲线绘图
    for i, reward in enumerate(all_rewards):
        plt.plot(reward, label=f"Trial {i+1}")
    plt.title("Reward Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_policy(model_path="ppo_unitree_policy_final.zip", num_trials=5, threshold=0.05, render=True)
