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
    all_wrist_errors = []  # List to store wrist errors for each trial
    all_x_errors = []

    for trial in range(num_trials):
        obs = env.reset()
        trial_errors = []
        trial_rewards = []  # List to store rewards for this trial
        trial_wrist_errors = []  # List to store wrist errors for this trial
        trial_x_errors = []

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

            # 计算 DoF 误差
            error = np.linalg.norm(q_actual - q_target)
            trial_errors.append(error)

            # 计算 wrist 误差
            wrist_keypoints_actual = sim_env.get_hand_keypoints(0)  # 获取当前的手部关节位置
            wrist_keypoints_target = sim_env.keypoint_goals[0]  # 获取目标手部关节位置
            wrist_error = np.linalg.norm(wrist_keypoints_actual - wrist_keypoints_target)
            trial_wrist_errors.append(wrist_error)

            idx = 22  # 可能需要根据实际情况调整
            error_x = np.abs(q_actual[idx] - q_target[idx])
            trial_x_errors.append(error_x)

            if render:
                sim_env.render_step()

        all_errors.append(trial_errors)
        all_rewards.append(trial_rewards)  # Store rewards for the trial
        all_wrist_errors.append(trial_wrist_errors)  # Store wrist errors for the trial
        all_x_errors.append(trial_x_errors)

        final_error = trial_errors[-1]
        final_wrist_error = trial_wrist_errors[-1]
        final_x_error = trial_x_errors[-1]
        print(f"[Trial {trial+1}] Final L2 Error: {final_error:.4f}, Final Wrist Error: {final_wrist_error:.4f}, Final X Error: {final_x_error:.4f}")

        if final_error < threshold:
            success_count += 1

    success_rate = success_count / num_trials * 100
    print(f"\n✅ Success Rate (< {threshold} rad error): {success_rate:.2f}%")

    dof_names = sim_env.get_dof_names()
    print("All DOF names:", dof_names)
    print(dof_names[idx])

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

    # wrist 误差曲线绘图
    for i, wrist_err in enumerate(all_wrist_errors):
        plt.plot(wrist_err, label=f"Trial {i+1}")
    plt.title("Wrist Error Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Wrist Error (L2 Norm)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    for i, error_x in enumerate(all_x_errors):
        plt.plot(error_x, label=f"Trial {i+1}")
    plt.title("Error_x Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Error_x (L2 Norm)")
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
