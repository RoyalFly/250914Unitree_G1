from isaacgym import gymapi
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def save_error_to_file(dof_err, keypoint_err, epoch, log_file='error_log.txt'):
    with open(log_file, 'a') as file:
        file.write(f"{dof_err:.4f} {keypoint_err:.4f}\n")

class UnitreeTrackingEnv:
    def __init__(self, urdf_path="unitree_g1.urdf", num_envs=4, render=True):
        self.urdf_path = urdf_path
        self.num_envs = num_envs
        self.render = render

        self.gym = gymapi.acquire_gym()
        self._create_sim()
        self._create_viewer()
        self._load_robot_asset()
        self._create_envs()
        self.reset()

    def _create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("创建模拟器失败")

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # 地面朝上
        self.gym.add_ground(self.sim, plane_params)

    def _create_viewer(self):
        if not self.render:
            self.viewer = None
            return
        cam_props = gymapi.CameraProperties()
        self.viewer = self.gym.create_viewer(self.sim, cam_props)
        if self.viewer is None:
            raise RuntimeError("创建渲染窗口失败")
        self.gym.viewer_camera_look_at(self.viewer, None,
                                       gymapi.Vec3(2, 2, 1.5),
                                       gymapi.Vec3(0, 0, 0.5))

    def _load_robot_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = False

        asset_root = "."
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, self.urdf_path, asset_options)
        if self.robot_asset is None:
            raise RuntimeError(f"加载URDF失败: {self.urdf_path}")
        self.num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        dof_names = [self.gym.get_asset_dof_name(self.robot_asset, i) for i in range(self.num_dofs)]
        include_keywords = ["shoulder", "elbow", "torso", "hand", "wrist"]
        exclude_keywords = ["pelvis", "hip", "ankle", "knee", "waist"]  # 排除下肢关键词

        self.upper_body_dof_indices = [
            i for i, name in enumerate(dof_names)
            if any(kw in name.lower() for kw in include_keywords)
            and not any(bad in name.lower() for bad in exclude_keywords)
        ]
        self.default_dof_positions = np.zeros(self.num_dofs, dtype=np.float32)

        # print(f"[DEBUG] 上半身关节数量: {len(self.upper_body_dof_indices)}")
        # print(f"[DEBUG] 上半身关节索引: {self.upper_body_dof_indices}")
        # print(f"[DEBUG] 对应关节名: {[dof_names[i] for i in self.upper_body_dof_indices]}")

    def _create_envs(self):
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, 0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        for i in range(self.num_dofs):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][i] = 400.0
            dof_props["damping"][i] = 40.0
            if i not in self.upper_body_dof_indices:
                dof_props["driveMode"][i] = gymapi.DOF_MODE_NONE
                dof_props["stiffness"][i] = 0.0
                dof_props["damping"][i] = 0.0

        # 初始化默认 DOF 状态
        self.default_dof_states = np.zeros(self.num_dofs, dtype=[('pos', np.float32), ('vel', np.float32)])
        for i in range(self.num_dofs):
            self.default_dof_states[i]['pos'] = self.default_dof_positions[i]
            self.default_dof_states[i]['vel'] = 0.0

        self.envs = []
        self.actors = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)) + 1)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 2.0)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            actor = self.gym.create_actor(env, self.robot_asset, pose, f"unitree_{i}", i, 1)

            self.gym.set_actor_dof_properties(env, actor, dof_props.copy())
            self.gym.set_actor_dof_position_targets(env, actor, self.default_dof_positions)
            self.gym.set_actor_dof_states(env, actor, self.default_dof_states, gymapi.STATE_ALL)

            self.envs.append(env)
            self.actors.append(actor)

        # 设置pd
        for i in range(self.num_envs):
            env = self.envs[i]
            actor = self.actors[i]

            props = self.gym.get_actor_dof_properties(env, actor)
            dof_names = [self.gym.get_asset_dof_name(self.robot_asset, j) for j in range(self.num_dofs)]

            # 定义论文中的参数表
            pd_table = {
                "hip_yaw":      (100, 2.5),
                "hip_roll":     (100, 2.5),
                "hip_pitch":    (100, 2.5),
                "knee":         (200, 5.0),
                "ankle_pitch":  (20,  0.2),
                "ankle_roll":   (20,  0.1),
                "shoulder_pitch": (90, 2.0),
                "shoulder_roll":  (60, 1.0),
                "shoulder_yaw":   (20, 0.4),
                "elbow":        (60,  1.0),
                "waist":        (400, 5.0),
                "wrist_pitch":  (75,  1.5),
                "wrist_roll":   (60, 1.0),
                "wrist_yaw":    (20, 0.4),
                "hand_thumb":   (5, 0.8),
                "hand_index":   (5, 0.5),
                "hand_middle":  (5, 0.5),
                "palm":         (15, 1.0),
                "torso":        (300, 5.0),
            }

            # 遍历每个DOF名称，匹配并设置增益
            for j, name in enumerate(dof_names):
                for key, (kp, kd) in pd_table.items():
                    if key in name.lower():  # 只要包含关键词即可
                        props["stiffness"][j] = kp
                        props["damping"][j] = kd
                        break
                else:
                    # 若未匹配，默认设为0（可选：防止腿部误动）
                    props["stiffness"][j] = 0.0
                    props["damping"][j] = 0.0

            # 应用设置
            self.gym.set_actor_dof_properties(env, actor, props)
    
    def get_dof_names(self):
        return [self.gym.get_asset_dof_name(self.robot_asset, i) for i in range(self.num_dofs)]

    def reset_motion_goals(self):
        self.motion_goals = np.random.uniform(
            low=-0.5, high=0.5, size=(self.num_envs, len(self.upper_body_dof_indices))
        ).astype(np.float32)

    def reset(self):
        obs_buf = []
        self.motion_goals = []
        self.keypoint_goals = []

        for i, env in enumerate(self.envs):
            actor = self.actors[i]

            # 将所有关节设置为初始默认姿态
            self.gym.set_actor_dof_position_targets(env, actor, self.default_dof_positions)
            dof_states = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)
            dof_states["pos"][:] = self.default_dof_positions
            dof_states["vel"][:] = 0.0
            self.gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)



            # 随机生成一个上半身目标姿态
            q_ref = self.default_dof_positions.copy()
            for dof_idx in self.upper_body_dof_indices:
                    q_ref[dof_idx] += np.random.uniform(-0.5, 0.5)

            self.motion_goals.append(q_ref[self.upper_body_dof_indices])

            # 设置动作目标
            for j, dof_idx in enumerate(self.upper_body_dof_indices):
                self.gym.set_actor_dof_position_targets(env, actor, q_ref)

            # 获取当前关键点作为目标
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            keypoint = self.get_hand_keypoints(i)
            self.keypoint_goals.append(keypoint)

            # 初始观测 = 当前q + 目标q
            dof_states = self.gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
            q_actual = np.array([dof_states[dof_idx]['pos'] for dof_idx in self.upper_body_dof_indices])
            obs = np.concatenate([q_actual, self.motion_goals[i]])
            obs_buf.append(obs)

        return np.array(obs_buf), {}



    def get_hand_keypoints(self, env_index):
        env = self.envs[env_index]
        actor = self.actors[env_index]

        hand_links = [
            "left_hand_thumb_1_link",
            "left_hand_thumb_2_link",
            "left_hand_index_1_link",
            "left_hand_middle_1_link",
            "left_hand_palm_link",
            "right_hand_thumb_1_link",
            "right_hand_thumb_2_link",
            "right_hand_index_1_link",
            "right_hand_middle_1_link",
            "right_hand_palm_link"
        ]

        rb_states = self.gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)

        keypoints = []
        for name in hand_links:
            link_idx = self.gym.find_actor_rigid_body_handle(env, actor, name)
            if link_idx == -1:
                raise ValueError(f"[ERROR] 找不到 link 名称: {name}")
            state = rb_states[link_idx]
            pos = np.array([
                state['pose']['p'][0],
                state['pose']['p'][1],
                state['pose']['p'][2]
            ])
            keypoints.append(pos)

        return np.concatenate(keypoints)  # shape = (24,)
    
    def _get_obs(self):
        obs = []
        for i, env in enumerate(self.envs):
            actor = self.actors[i]
            dof_states = self.gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            joint_pos = np.array([dof_states[j]['pos'] for j in self.upper_body_dof_indices])
            joint_vel = np.array([dof_states[j]['vel'] for j in self.upper_body_dof_indices])
            goal = self.motion_goals[i]
            obs_i = np.concatenate([joint_pos, joint_vel, goal], axis=0)
            obs.append(obs_i.astype(np.float32))
        return np.stack(obs)

    def step(self, action):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        obs_buf = []
        done_buf = []

        # 把下半身所有关节置零
        for i, env in enumerate(self.envs):
            actor = self.actors[i]
            full_state = self.gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            new_state = full_state.copy()

            for j in range(self.num_dofs):
                if j not in self.upper_body_dof_indices:
                    new_state[j]['pos'] = self.default_dof_positions[j]
                    new_state[j]['vel'] = 0.0  # 防止受重力加速

            self.gym.set_actor_dof_states(env, actor, new_state, gymapi.STATE_ALL)

        for i, env in enumerate(self.envs):
            actor = self.actors[i]

            # 1. 应用 agent 输出的上半身动作（绝对位置控制）
            dof_target = self.default_dof_positions.copy()
            for j, dof_idx in enumerate(self.upper_body_dof_indices):
                dof_target[dof_idx] = action[i, j]
            self.gym.set_actor_dof_position_targets(env, actor, dof_target)

        # 2. 执行一步仿真
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        for i, env in enumerate(self.envs):
            actor = self.actors[i]

            # 3. 获取当前观测（上半身关节角度）
            dof_states = self.gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
            q_actual = np.array([dof_states[dof_idx]['pos'] for dof_idx in self.upper_body_dof_indices])
            q_ref = self.motion_goals[i]  # 当前目标动作

            # 4. 获取关键点（手部）状态
            p_actual = self.get_hand_keypoints(i)         # 当前手部位置
            p_ref = self.keypoint_goals[i]                # 目标手部位置

            # -------- Reward 计算 --------

            # --- DoF误差处理 ---
            dof_err = np.linalg.norm(q_ref - q_actual)

            # 对小误差使用sigmoid增强灵敏度（学习较小误差）
            if dof_err < 0.1:
                reward_dof = 1.0 - np.exp(-6.0 * dof_err)  # 更高灵敏度的奖励函数
            else:
                reward_dof = 1.0 - np.exp(-8.0 * dof_err ** 2)  # 减少大误差的惩罚力度

            # 对小误差的奖励更灵敏（增强小误差的影响）
            reward_dof += 2.5 / (1.0 + 3 * dof_err)

            # --- 手部关键点误差处理 ---
            keypoint_err = np.linalg.norm(p_ref - p_actual)

            # 小误差时，使用exp函数增强灵敏度
            if keypoint_err < 0.1:
                reward_keypoint = 1.0 - np.exp(-7.0 * keypoint_err)  # 对小误差非常灵敏
            else:
                reward_keypoint = 1.0 - np.tanh(5.0 * keypoint_err)  # 在误差较大时使用tanh

            # 减小惩罚力度，避免过度惩罚
            if dof_err > 0.3:
                reward_dof -= min(0.2 * dof_err, 0.2)  # 增加惩罚力度
            if keypoint_err > 0.2:
                reward_keypoint -= min(0.2 * keypoint_err, 0.2)

            # 对小误差进一步强化奖励
            reward_keypoint += 2.5 / (1.0 + 6 * keypoint_err)

            # 根据训练进度逐渐调整全身和关键部位的权重
            total_epochs = 19532
            epoch_progress = i / total_epochs

            # 初始阶段更多关注全身误差，逐步减少关键点误差的权重
            dynamic_weight_body = max(0.925 - 0.925 * epoch_progress**2, 0.2)
            dynamic_weight_keypoint = 1 - dynamic_weight_body

            reward = dynamic_weight_body * reward_dof + dynamic_weight_keypoint * reward_keypoint

            # 修改 joint_bonus，增强对小误差的响应
            joint_bonus = np.exp(-5 * (dof_err ** 2 + keypoint_err ** 2))  # 更平滑的精度奖励
            reward += 1.5 * joint_bonus  # 增强精确控制时的奖励，鼓励更精细的动作控制

            reward = np.clip(reward, 0.0, 5.0) 

            # 平滑奖励
            window_size = 5
            if i > window_size:
                reward_window = np.nanmean(rewards[max(0, i - window_size):i])
                reward = reward_window

            # 更新 reward
            rewards[i] = reward


            # 构造 obs
            delta_q = q_ref - q_actual
            obs = np.concatenate([q_actual, q_ref, delta_q])
            obs_buf.append(obs)
            done_buf.append(False)

        # 输出格式：(obs, reward, done, info)
        return np.array(obs_buf), rewards, np.array(done_buf), {}

    def render_step(self):
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

    def close(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def get_current_upperbody_pos(self):
        pos = []
        for i, env in enumerate(self.envs):
            actor = self.actors[i]
            dof_states = self.gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
            pos.append(np.array([dof_states[d]['pos'] for d in self.upper_body_dof_indices]))
        return pos[0]

if __name__ == "__main__":
    env = UnitreeTrackingEnv(urdf_path="unitree_g1.urdf", num_envs=128, render=True)
    try:
        for _ in range(1000):
            env.reset_motion_goals()
            action = env.motion_goals.copy()
            reward = env.step(action)
            env.render_step()
            print(f"reward: {reward}")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()