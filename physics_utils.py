# physics_utils.py
import numpy as np
from isaacgym import gymapi

class DomainRandomizer:
    def __init__(self, config):
        # 从配置加载随机化参数（论文附录A）
        self.gravity_range = config.get('gravity_range', (-0.1, 0.1))  # ±0.1m/s²
        self.friction_range = config.get('friction_range', (0.6, 2.0))
        
    def randomize_gravity(self, env_handle):
        """随机化重力（论文3.1节）"""
        gravity = np.random.uniform(*self.gravity_range)
        self.gym.set_sim_params(env_handle, gymapi.SimParams(gravity=gymapi.Vec3(0, 0, -9.81 + gravity)))

    def randomize_rigid_body_properties(self, env_handle, actor_handle):
        """随机化质量、摩擦等（论文3.1节）"""
        props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
        for p in props:
            p.mass *= np.random.uniform(0.8, 1.2)  # 质量变化±20%
        self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, props)