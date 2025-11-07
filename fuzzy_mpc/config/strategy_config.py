STRATEGY_CONFIG = {
    "target_distance": 200.0,
    "threat_distances": [100.0, 80.0, 50.0, 30.0, 15.0],  # 调整威胁距离
    "threat_penalties": [0.0, -0.3, -1.0, -2.0, -4.0],  # 调整惩罚强度
    "lane_keeping_reward": 0.1,  # 每20米奖励0.1
    "lane_keeping_interval": 20.0,  # 每20米给予奖励
    "prediction_horizon": 3,
    "min_safe_distance": 15.0,
    "must_change_threshold": -0.5,  # 当评价分数低于-0.5时触发变道
    "lane_change_cost": -0.1,
    "reset_interval": 200.0,
    "trajectory_tracking_reward": 0.05,
    "vehicle_count_weight": 0.05,
    "min_lane_change_advantage": 0.1,
    "look_ahead_distance": 150.0,
    "target_speed": 30.0,
    "speed_tolerance": 2.0,
}