import numpy as np


class DynamicTargetManager:
    def __init__(self, config):
        self.config = config
        self.current_target_x = 0.0
        self.current_target_lane = 0.0
        self.base_target_x = 0.0
        self.target_initialized = False
        self.target_update_count = 0
        self.total_distance_traveled = 0.0
        self.last_ego_x = 0.0
        self.strategy_target_lane = None
        self.target_speed = 30.0  # 设置目标速度
        self.lane_width = 4.0  # 车道宽度

    def initialize_target(self, current_x, current_lane):
        """初始化第一个目标"""
        self.base_target_x = current_x
        self.current_target_x = current_x + 200.0
        self.current_target_lane = self._get_lane_center(current_lane)
        self.target_initialized = True
        self.target_update_count = 0
        self.last_ego_x = current_x
        print(
            f"初始目标设定: x={self.current_target_x:.1f}m, 车道={self.current_target_lane:.1f}m, 目标速度={self.target_speed:.1f}m/s")

    def _get_lane_center(self, lane_number):
        """根据车道编号获取车道中央位置"""
        lane_number = max(0, min(lane_number, 2))
        # 每个车道的中央位置：车道0: 2米, 车道1: 6米, 车道2: 10米
        return (lane_number + 0.5) * self.lane_width

    def update_target(self, current_x, current_lane):
        """更新目标位置"""
        if not self.target_initialized:
            self.initialize_target(current_x, current_lane)
            return

        # 计算行驶距离
        distance_traveled = current_x - self.last_ego_x
        self.total_distance_traveled += distance_traveled
        self.last_ego_x = current_x

        distance_to_target = abs(current_x - self.current_target_x)

        if distance_to_target < self.config.target_update_tolerance:
            self.target_update_count += 1
            self.base_target_x = self.current_target_x
            self.current_target_x = self.base_target_x + 200.0
            self.current_target_lane = self._get_lane_center(current_lane)
            print(
                f"目标更新 #{self.target_update_count}: x={self.current_target_x:.1f}m, 车道={self.current_target_lane:.1f}m")

    def get_target_trajectory(self, current_x, current_y, current_heading, current_speed, horizon, dt):
        """生成参考轨迹 - 确保在车道中央行驶"""
        if not self.target_initialized:
            self.initialize_target(current_x, 1)

        ref_trajectory = []

        # 计算当前应该在哪条车道
        current_lane_num = int(current_y // self.lane_width)
        current_lane_num = max(0, min(current_lane_num, 2))

        # 如果策略要求变道，使用策略的目标车道
        if self.strategy_target_lane is not None:
            target_lane_center = self.strategy_target_lane
        else:
            target_lane_center = self.current_target_lane

        # 使用目标速度而不是当前速度
        target_speed = self.target_speed

        for k in range(horizon + 1):
            ref_x = current_x + target_speed * k * dt  # 使用目标速度
            ref_y = target_lane_center  # 车道中央位置
            ref_theta = 0.0
            ref_v = min(target_speed, self.config.v_max)  # 使用目标速度
            ref_trajectory.append([ref_x, ref_y, ref_theta, ref_v])

        print(f"参考轨迹: 车道{current_lane_num}, 中央Y={target_lane_center:.1f}, 目标速度={target_speed:.1f}m/s")
        return np.array(ref_trajectory).T

    def set_strategy_target_lane(self, target_lane_center):
        """设置策略目标车道"""
        self.strategy_target_lane = target_lane_center

    def set_target_speed(self, speed):
        """设置目标速度"""
        self.target_speed = max(self.config.v_min, min(speed, self.config.v_max))
        print(f"更新目标速度: {self.target_speed:.1f}m/s")

    def get_current_target_info(self):
        return {
            'target_x': self.current_target_x,
            'target_lane': self.current_target_lane,
            'target_speed': self.target_speed,
            'progress_count': self.target_update_count,
            'total_distance': self.total_distance_traveled
        }

    def should_reset_evaluation(self, reset_interval):
        """检查是否应该重置评价"""
        return self.total_distance_traveled >= reset_interval

    def reset_evaluation(self):
        """重置评价计数器"""
        self.total_distance_traveled = 0.0