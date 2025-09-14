import numpy as np
from gym import Env, spaces
from isaacgym import gymapi, gymutil

class UnitreeTrackingEnv(Env):
    def __init__(self, urdf_path="unitree_g1.urdf", num_envs=8, render=True):
        self.urdf_path = urdf_path
        self.num_envs = num_envs
        self.render = render

        self.gym = gymapi.acquire_gym()
        self._create_sim()  # 创建物理模拟器
        self._load_robot_asset()  # 加载机器人模型
        self._create_envs()  # 创建环境实例
        
        if self.render:
            self._create_viewer()  # 如果渲染开启，创建渲染视图
        
        # 修正：total_dofs 是总关节数，num_upper_dofs 是上半身关节数
        self.total_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_upper_dofs = len(self.upper_body_dof_indices)
        
        # 根据机器人的总关节数量初始化 default_dof_positions
        self.default_dof_positions = np.zeros(self.total_dofs, dtype=np.float32)
        
        # 检查所有上半身关节索引是否都在有效范围内
        for idx in self.upper_body_dof_indices:
            if idx >= self.total_dofs:
                raise ValueError(f"上半身关节索引 {idx} 超出范围 (0-{self.total_dofs-1})")
        
        self.reset()  # 重置环境
        # 观察空间：上半身关节位置 + 速度 + 目标位置
        self.num_hand_keypoints = 10 * 3  # 10个关键点，每个点3个坐标
        obs_shape = self.num_upper_dofs * 2 + self.num_hand_keypoints  # 实际位置 + 目标位置 + 手部关键点
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_upper_dofs,), dtype=np.float32)
        test_obs = self.reset()
        print(f"观察空间形状: {test_obs.shape}")
        print(f"单个观察维度: {test_obs[0].shape if test_obs.size > 0 else 'N/A'}")

    def _create_sim(self):
        """
        创建物理模拟器
        """
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

    def _load_robot_asset(self):
        """
        加载机器人模型（URDF）
        """
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
        exclude_keywords = ["pelvis", "hip", "ankle", "knee", "waist"]

        # 筛选出上半身关节
        self.upper_body_dof_indices = [
            i for i, name in enumerate(dof_names)
            if any(kw in name.lower() for kw in include_keywords)
            and not any(bad in name.lower() for bad in exclude_keywords)
        ]

    def _create_envs(self):
        """
        创建环境实例
        """
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, 0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # 创建多个环境实例
        self.envs = []
        self.actors = []
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)) + 1)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 2.0)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            actor = self.gym.create_actor(env, self.robot_asset, pose, f"unitree_{i}", i, 1)
            self.envs.append(env)
            self.actors.append(actor)

    def _create_viewer(self):
        """
        创建渲染视图
        """
        if self.render:
            cam_props = gymapi.CameraProperties()
            self.viewer = self.gym.create_viewer(self.sim, cam_props)
            if self.viewer is None:
                raise RuntimeError("Failed to create viewer")
            self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(2, 2, 1.5), gymapi.Vec3(0, 0, 0.5))


    def reset(self):
        obs_buf = []
        self.motion_goals = []
        self.keypoint_goals = []

        for i, env in enumerate(self.envs):
            actor = self.actors[i]
        
            # 重置关节状态
            dof_states = np.zeros(self.total_dofs, dtype=gymapi.DofState.dtype)
            dof_states["pos"][:] = self.default_dof_positions
            dof_states["vel"][:] = 0.0
            self.gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

            # 初始化目标关节位置
            q_ref = self.default_dof_positions.copy()
            for dof_idx in self.upper_body_dof_indices:
                q_ref[dof_idx] += np.random.uniform(-0.5, 0.5)
        
            # 获取手部关键点初始位置
            p_actual = self.get_hand_keypoints(i)
        
            # 保存目标
            self.motion_goals.append(q_ref[self.upper_body_dof_indices])
            self.keypoint_goals.append(p_actual)

            # 获取当前关节位置
            dof_states = self.gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
            q_actual = np.array([dof_states[dof_idx]['pos'] for dof_idx in self.upper_body_dof_indices])
        
            # 构建观察值
            obs = np.concatenate([q_actual, self.motion_goals[i], p_actual])
            obs_buf.append(obs)

        return np.array(obs_buf)
    
    def step(self, action):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        obs_buf = []

        for i, env in enumerate(self.envs):
            actor = self.actors[i]
            # 设置目标位置
            dof_target = self.default_dof_positions.copy()
            for j, dof_idx in enumerate(self.upper_body_dof_indices):
                dof_target[dof_idx] = action[i, j] if action.ndim > 1 else action[j]
            self.gym.set_actor_dof_position_targets(env, actor, dof_target)

        # 执行仿真
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        for i, env in enumerate(self.envs):
            actor = self.actors[i]
            # 获取新状态
            dof_states = self.gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
            q_actual = np.array([dof_states[dof_idx]['pos'] for dof_idx in self.upper_body_dof_indices])
            p_actual = self.get_hand_keypoints(i)
            
            # 计算奖励
            reward = self.calculate_reward(
                q_ref=self.motion_goals[i],
                q_actual=q_actual,
                p_ref=self.keypoint_goals[i],
                p_actual=p_actual
            )
            rewards[i] = reward
            
            # 构建新观察值
            obs = np.concatenate([q_actual, self.motion_goals[i], p_actual])
            obs_buf.append(obs)

        return np.array(obs_buf), rewards, np.array([False] * self.num_envs), {}

    def get_hand_keypoints(self, env_index):
        """
        获取手部关键点
        """
        # 示例：这里只列出几个手部关键点，实际使用时需要根据机器人模型调整
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

        rb_states = self.gym.get_actor_rigid_body_states(self.envs[env_index], self.actors[env_index], gymapi.STATE_POS)
        keypoints = []

        for link in hand_links:
            link_idx = self.gym.find_actor_rigid_body_handle(self.envs[env_index], self.actors[env_index], link)
            if link_idx == -1:
                raise ValueError(f"找不到 link 名称: {link}")
            state = rb_states[link_idx]
            pos = np.array([state['pose']['p'][0], state['pose']['p'][1], state['pose']['p'][2]])
            keypoints.append(pos)

        return np.concatenate(keypoints)

    def render_step(self):
        """
        渲染每一步（如果启用渲染）
        """
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

    def close(self):
        """
        关闭模拟器
        """
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def calculate_reward(self, q_ref, q_actual, p_ref, p_actual):
        # 归一化误差（避免绝对值过大）
        joint_error = np.linalg.norm(q_ref - q_actual) / len(q_ref)
        hand_error = np.linalg.norm(p_ref - p_actual) / len(p_ref)

        # 奖励平滑
        smooth_factor = 0.9
        joint_error = smooth_factor * joint_error + (1 - smooth_factor) * joint_error
        hand_error = smooth_factor * hand_error + (1 - smooth_factor) * hand_error

        # 基于成功率的奖励设计
        success_threshold = 0.1  # 误差阈值
        joint_success = joint_error < success_threshold
        hand_success = hand_error < success_threshold

        # 分级奖励
        if joint_success and hand_success:
            return 15.0  # 完全成功
        elif joint_success or hand_success:
            return 5.0   # 部分成功
        else:
            return -0.2 * joint_error - 0.1 * hand_error  # 微小惩罚
