import casadi as csd
import numpy as np
import math


def filter_obs(obs):
    """过滤观测数据，移除不相关的智能体"""
    if obs is None or len(obs) == 0:
        return np.array([]), np.array([])

    relevant_obs = obs[obs[:, 0] != 0]
    if len(relevant_obs) == 0:
        return np.array([]), np.array([])

    ego_obs = relevant_obs[0]
    other_obs = relevant_obs[1:]
    return ego_obs, other_obs


def analyze_vehicle_speeds(ego_speed, other_vehicles):
    """分析车辆速度分布"""
    if other_vehicles is None or len(other_vehicles) == 0:
        return {
            'slow_vehicles': 0,
            'fast_vehicles': 0,
            'avg_other_speed': 0,
            'speed_difference': 0
        }

    slow_count = 0
    fast_count = 0
    other_speeds = []

    for vehicle in other_vehicles:
        if len(vehicle) >= 5:
            vx, vy = vehicle[3], vehicle[4]
            speed = math.sqrt(vx ** 2 + vy ** 2)
            other_speeds.append(speed)

            if speed < ego_speed - 2:  # 比自车慢2m/s以上
                slow_count += 1
            elif speed > ego_speed + 2:  # 比自车快2m/s以上
                fast_count += 1

    avg_other_speed = np.mean(other_speeds) if other_speeds else 0
    speed_difference = ego_speed - avg_other_speed if other_speeds else 0

    return {
        'slow_vehicles': slow_count,
        'fast_vehicles': fast_count,
        'avg_other_speed': avg_other_speed,
        'speed_difference': speed_difference
    }


def safe_smooth_max(z, beta=5.0, epsilon=1e-6):
    """安全的平滑最大函数"""
    z_safe = csd.fmin(csd.fmax(z, -100), 100)
    return (1 / beta) * csd.log(1 + csd.exp(beta * z_safe)) + epsilon


def safe_obstacle_penalty(ego_x, ego_y, ego_heading, obs_x, obs_y, obs_heading,
                          obs_length, obs_width, flag, config):
    """安全的障碍物惩罚计算"""
    epsilon = 1e-6

    dx = ego_x - obs_x + epsilon
    dy = ego_y - obs_y + epsilon

    cos_obs = csd.cos(obs_heading)
    sin_obs = csd.sin(obs_heading)

    rot_dx = dx * cos_obs + dy * sin_obs
    rot_dy = -dx * sin_obs + dy * cos_obs

    margin_x = obs_length / 2 + config.vehicle_length / 2 + config.safety_distance + epsilon
    margin_y = obs_width / 2 + config.vehicle_width / 2 + config.safety_distance + epsilon

    norm_dx = rot_dx / margin_x
    norm_dy = rot_dy / margin_y
    norm_dist = norm_dx ** 2 + norm_dy ** 2 + epsilon

    safe_penalty = csd.exp(-config.obstacle_decay_rate * csd.fmax(0, 1 - norm_dist))
    penalty_bounded = csd.fmin(safe_penalty, 1000.0)

    return safe_smooth_max(flag - 0.5) * penalty_bounded


def calculate_lane_center_casadi(y_position, lane_width=4.0):
    """使用CasADi函数计算车道中央"""
    lane_num = csd.floor(y_position / lane_width)
    lane_num = csd.fmax(0, csd.fmin(lane_num, 2))  # 限制在0-2车道
    lane_center = (lane_num + 0.17) * lane_width  # 车道中央位置
    return lane_num, lane_center


def safe_boundary_penalty(y_position, y_min, y_max, config):
    """安全的边界惩罚函数"""
    epsilon = 1e-6

    lane_width = 4.0
    lane_num, lane_center = calculate_lane_center_casadi(y_position, lane_width)

    center_penalty = (y_position - lane_center) ** 2
    boundary_penalty_left = safe_smooth_max(y_min - y_position)
    boundary_penalty_right = safe_smooth_max(y_position - y_max)

    total_penalty = (config.w_y * center_penalty +
                     config.w_bound * (boundary_penalty_left ** 2 + boundary_penalty_right ** 2))

    return total_penalty


def predict_vehicle_trajectories(ego_x, ego_y, other_vehicles, horizon, dt):
    """预测前方车辆轨迹"""
    predictions = []

    if other_vehicles is None or len(other_vehicles) == 0:
        return predictions

    relevant_vehicles = sorted(other_vehicles,
                               key=lambda v: v[1] - ego_x if v[1] > ego_x else float('inf'))[:5]

    for vehicle in relevant_vehicles:
        if len(vehicle) < 6:
            continue

        other_x, other_y, other_vx, other_vy, other_heading = vehicle[1], vehicle[2], vehicle[3], vehicle[4], vehicle[5]
        other_speed = math.sqrt(other_vx ** 2 + other_vy ** 2)

        vehicle_trajectory = []
        for k in range(horizon + 1):
            pred_x = other_x + other_speed * math.cos(other_heading) * k * dt
            pred_y = other_y + other_speed * math.sin(other_heading) * k * dt
            vehicle_trajectory.append([pred_x, pred_y, other_heading, other_speed])

        predictions.append(np.array(vehicle_trajectory).T)

    return predictions


def check_obstacle_collision(ego_x, ego_y, other_vehicles, safety_distance=5.0):
    """检查是否与障碍物发生碰撞"""
    if other_vehicles is None or len(other_vehicles) == 0:
        return False, None

    for vehicle in other_vehicles:
        if len(vehicle) < 3:
            continue

        other_x, other_y = vehicle[1], vehicle[2]
        distance = math.sqrt((ego_x - other_x) ** 2 + (ego_y - other_y) ** 2)

        if distance < safety_distance:
            return True, vehicle

    return False, None