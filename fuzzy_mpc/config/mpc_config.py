import numpy as np

class MPCConfig:
    def __init__(self):
        # 预测时域
        self.N = 20
        self.dt = 1.0 / 5

        # 目标参数
        self.target_distance = 200.0
        self.target_update_tolerance = 10.0

        # 车辆参数
        self.L = 5.0
        self.vehicle_length = 5.0
        self.vehicle_width = 2.0

        # 道路参数 - 修改为实际道路边界
        self.lane_width = 4.0
        self.y_min = 0.0    # 道路最左侧
        self.y_max = 12.0   # 道路最右侧，3车道 * 4米 = 12米

        # 状态约束 - 提高自车速度
        self.v_min = 25.0  # 提高最小速度
        self.v_max = 35.0  # 提高最大速度
        self.delta_min = -np.pi / 6
        self.delta_max = np.pi / 6

        # 控制约束
        self.a_min = -5.0
        self.a_max = 4.0
        self.delta_dot_max = np.pi / 8
        self.theta_dot_max = np.pi / 6

        # 安全参数
        self.safety_distance = 3.0
        self.max_obs = 10

        # 成本权重 - 大幅增加车道中央权重
        self.w_x = 1.0
        self.w_y = 200.0  # 大幅增加横向位置权重
        self.w_theta = 5.0
        self.w_v = 0.5

        # 控制平滑性权重
        self.w_a = 0.3
        self.w_delta = 0.5
        self.w_delta_dot = 1.0

        # 约束惩罚权重
        self.w_bound = 500.0
        self.w_obstacle = 200.0

        # 策略权重
        self.w_threat = 200.0
        self.w_lane_keep = 10.0
        self.w_lane_change = 1.0
        self.w_strategy_evaluation = 2.0
        self.w_trajectory_tracking = 50.0  # 增加轨迹跟踪权重

        # 终端成本权重
        self.w_fx = 1.0
        self.w_fy = 200.0  # 增加终端横向位置权重
        self.w_ftheta = 8.0
        self.w_fv = 1.0

        # 障碍物参数
        self.obstacle_decay_rate = 5.0
        self.obstacle_smooth_beta = 3.0

        # 求解器参数
        self.max_iter = 500