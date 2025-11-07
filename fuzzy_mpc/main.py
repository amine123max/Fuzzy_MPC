# ==================== 导入模块 ====================
import gymnasium as gym
import pprint, copy, math
import numpy as np
import casadi as csd
from matplotlib import pyplot as plt
import matplotlib
import pygame
import cv2
import sys
import os
import time

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入自定义模块
from config.mpc_config import MPCConfig
from config.simulation_config import SIMULATION_CONFIG
from config.strategy_config import STRATEGY_CONFIG
from managers.target_manager import DynamicTargetManager
from controllers.mpc_controller import MPCController
from controllers.strategy_evaluator import FuzzyStrategyEvaluator
from utils.math_utils import filter_obs, check_obstacle_collision
from utils.visualization import plot_simulation_results

try:
    from highway_env_mpc.envs.env import HighwayEnvMPC

    print("Successfully imported HighwayEnvMPC class")
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying to import directly from env.py...")
    sys.path.append(os.path.join(os.path.dirname(__file__), 'highway_env_mpc', 'envs'))
    from env import HighwayEnvMPC


# ==================== 主函数模块 ====================
def main():
    # ==================== 系统初始化模块 ====================
    print("Fuzzy-MPC 自动驾驶控制系统启动 - 修复评价系统版本")

    # 创建配置实例
    config = MPCConfig()

    # 创建环境实例
    print("创建环境实例...")
    env = HighwayEnvMPC(render_mode='rgb_array')

    # 环境配置 - 设置其他车辆较慢
    env.config.update({
        "duration": 100,
        "vehicles_count": 10,
        # 设置其他车辆较慢
        "vehicle_config": {
            "max_speed": 22,  # 其他车辆最大速度较低
            "min_speed": 15,
            "speed_range": [15, 22],  # 其他车辆速度范围
        },
        # 控制车辆生成
        "vehicles_density": 0.8,
        "initial_spacing": 3.0,
        # 控制相对速度
        "relative_lane_speed": 0.8,
    })

    pprint.pprint(env.config)

    # 初始化观测
    obs, _ = env.reset()
    ego_obs, other_obs = filter_obs(obs)
    N = config.N
    ego_actions = np.zeros((N, 2))
    dt = config.dt
    print(f"时间步长: {dt}")

    # 创建管理器和控制器
    target_manager = DynamicTargetManager(config)
    strategy_evaluator = FuzzyStrategyEvaluator(STRATEGY_CONFIG)
    mpc_controller = MPCController(config)

    # 设置目标速度比其他车辆快
    target_manager.set_target_speed(25.0)  # 自车目标速度30m/s

    # 创建图片保存目录
    save_dir = r"C:\Users\Amine\Desktop\fuzzy_mpc\pic"
    os.makedirs(save_dir, exist_ok=True)

    # 数据记录
    trajectory_data = []
    speed_data = []
    time_data = []
    lane_change_log = []
    fuzzy_tendency_data = []
    obstacle_detection_log = []
    evaluation_history = []
    cumulative_reward_history = []
    threat_penalty_history = []

    # 仿真计时
    start_time = time.time()
    image_count = 0

    # ==================== 主控制循环模块 ====================
    print("开始Fuzzy-MPC控制循环...")
    for it in range(SIMULATION_CONFIG["max_iterations"]):
        current_time = time.time() - start_time
        print(f"迭代: {it}, 已用时间: {current_time:.2f}s")

        if current_time >= SIMULATION_CONFIG["max_duration"]:
            print("达到最大仿真时间")
            break

        # 获取当前状态
        current_x = ego_obs[1]
        current_y = ego_obs[2]
        current_speed = max(math.sqrt(ego_obs[3] ** 2 + ego_obs[4] ** 2), config.v_min)

        # 记录数据
        trajectory_data.append((current_x, current_y))
        speed_data.append(current_speed)
        time_data.append(current_time)

        # 获取其他车辆信息
        _, other_obs_current = filter_obs(obs)

        # ==================== 速度优势分析模块 ====================
        if len(other_obs_current) > 0:
            # 分析周围车辆速度
            other_speeds = []
            for vehicle in other_obs_current:
                if len(vehicle) >= 5:
                    vx, vy = vehicle[3], vehicle[4]
                    speed = math.sqrt(vx ** 2 + vy ** 2)
                    other_speeds.append(speed)

            if other_speeds:
                avg_other_speed = np.mean(other_speeds)
                max_other_speed = np.max(other_speeds)
                speed_advantage = current_speed - avg_other_speed

                print(
                    f"速度分析: 自车={current_speed:.1f}m/s, 其他车平均={avg_other_speed:.1f}m/s, 优势={speed_advantage:.1f}m/s")

                # 如果速度优势不足，提高目标速度
                if speed_advantage < 5.0 and current_speed < config.v_max - 2.0:
                    new_target_speed = min(config.v_max, current_speed + 2.0)
                    target_manager.set_target_speed(new_target_speed)

        # ==================== 障碍物检测模块 ====================
        print(f"检测到 {len(other_obs_current)} 辆其他车辆")

        # 检查最近的障碍物并分析速度
        if len(other_obs_current) > 0:
            front_vehicles = [v for v in other_obs_current if v[1] > current_x]
            if front_vehicles:
                closest_vehicle = min(front_vehicles, key=lambda v: v[1] - current_x)
                distance_to_closest = closest_vehicle[1] - current_x
                closest_lane = int(round(closest_vehicle[2] / 4.0))
                current_lane = int(round(current_y / 4.0))

                other_speed = math.sqrt(closest_vehicle[3] ** 2 + closest_vehicle[4] ** 2)
                speed_difference = current_speed - other_speed

                print(f"最近前方车辆: 距离 {distance_to_closest:.1f}m, 车道 {closest_lane}, 速度 {other_speed:.1f}m/s")
                print(f"自车速度: {current_speed:.1f}m/s, 速度差: {speed_difference:.1f}m/s")

                if distance_to_closest < 100.0:
                    if closest_lane == current_lane:
                        if speed_difference > 2.0:
                            print(
                                f"超车机会: 同车道前方有慢车! 距离: {distance_to_closest:.1f}m, 速度差: {speed_difference:.1f}m/s")
                        else:
                            print(f"注意: 同车道前方有车辆! 距离: {distance_to_closest:.1f}m")
                    else:
                        print(f"注意: 相邻车道有车辆! 距离: {distance_to_closest:.1f}m")

                obstacle_detection_log.append({
                    'iteration': it,
                    'distance': distance_to_closest,
                    'closest_lane': closest_lane,
                    'current_lane': current_lane,
                    'other_speed': other_speed,
                    'ego_speed': current_speed,
                    'speed_difference': speed_difference
                })

        # 检查碰撞
        collision, collided_vehicle = check_obstacle_collision(current_x, current_y, other_obs_current)
        if collision:
            collision_distance = math.sqrt(
                (current_x - collided_vehicle[1]) ** 2 + (current_y - collided_vehicle[2]) ** 2)
            print(f"碰撞警告! 与车辆距离: {collision_distance:.1f}m")

        # ==================== 策略决策模块 ====================
        strategy_result = strategy_evaluator.fuzzy_comprehensive_evaluation(
            current_x, current_y, current_speed, other_obs_current, it
        )

        fuzzy_tendency_data.append(strategy_result['fuzzy_tendency'])
        evaluation_history.append(strategy_result['cumulative_evaluation'])
        cumulative_reward_history.append(strategy_result['cumulative_reward'])
        threat_penalty_history.append(strategy_result['threat_penalty'])

        # 记录变道决策
        if strategy_result['should_change_lane']:
            lane_change_log.append((it, current_x, current_y, strategy_result['best_lane']))
            print(f"记录变道决策: 迭代 {it}")

        # 关键修复：先更新奖励和变道评价
        distance_traveled = current_speed * dt
        lane_keeping_reward = strategy_evaluator.update_lane_keeping_reward(current_y, distance_traveled)
        trajectory_tracking_reward = strategy_evaluator.update_trajectory_tracking_reward(
            current_y, target_manager.current_target_lane
        )

        # 更新变道评价标准（累积奖励 - 当前惩罚）
        should_reset = strategy_evaluator.update_lane_evaluation(
            distance_traveled,
            strategy_result['threat_penalty'],  # 当前瞬时惩罚
            lane_keeping_reward
        )

        if should_reset:
            print(f"重置评价标准")

        # 根据策略结果更新目标车道
        if strategy_result['should_change_lane']:
            target_manager.current_target_lane = strategy_result['target_lane']
            target_manager.set_strategy_target_lane(strategy_result['target_lane'])
            strategy_evaluator.reset_evaluation()
            print(f"执行变道: 移动到车道 {strategy_result['target_lane'] / 4:.0f}")
            print(
                f"评价分数: {strategy_result['cumulative_evaluation']:.2f} = 累计奖励{strategy_result['cumulative_reward']:.2f} + 当前惩罚{strategy_result['threat_penalty']:.2f}")
            print(f"当前车道评分: {strategy_result['current_lane_score']:.2f}")
            print(f"最佳车道评分: {strategy_result['best_lane_score']:.2f} (车道 {strategy_result['best_lane']})")
            print(f"车道车辆数量: {strategy_result['lane_vehicle_counts']}")
            print(f"模糊倾向: {strategy_result['fuzzy_tendency']:.2f}, 建议: {strategy_result['fuzzy_recommendation']}")
            print(f"变道优势: {strategy_result['lane_change_advantage']:.2f}")
        else:
            target_manager.set_strategy_target_lane(None)
            print(
                f"保持车道: 评价分数{strategy_result['cumulative_evaluation']:.2f} = 累计奖励{strategy_result['cumulative_reward']:.2f} + 当前惩罚{strategy_result['threat_penalty']:.2f}")

        # 更新目标位置
        target_manager.update_target(current_x, strategy_result['target_lane'] / 4)
        target_info = target_manager.get_current_target_info()

        # 生成参考轨迹
        ref_trajectory = target_manager.get_target_trajectory(
            current_x, current_y, ego_obs[5], current_speed, N, dt
        )

        # 车道跟踪监控
        current_lane, lane_center = strategy_evaluator.calculate_lane_center(current_y)
        lane_error = abs(current_y - lane_center)

        print(f"车道跟踪: 当前位置 Y={current_y:.2f}, 车道{current_lane}中心={lane_center:.1f}, 误差={lane_error:.2f}m")

        if lane_error > 0.2:
            print(f"警告: 车辆偏离车道中心! 误差: {lane_error:.2f}m")
        elif lane_error < 0.05:
            print(f"优秀: 车辆精确保持在车道中心")

        # ==================== MPC控制模块 ====================
        # 构建MPC优化问题
        current_state = [current_x, current_y, current_speed, ego_obs[5]]
        opti, variables = mpc_controller.build_optimization_problem(
            current_state, ref_trajectory, other_obs_current, strategy_result,
            lane_keeping_reward, trajectory_tracking_reward, dt
        )

        # 设置初始猜测和求解器
        mpc_controller.set_initial_guess(opti, variables, current_state, ego_actions, N, dt)
        mpc_controller.configure_solver(opti)

        try:
            # 求解MPC优化问题
            sol = opti.solve()

            # 轨迹可视化数据提取
            if hasattr(env, 'viewer') and env.viewer is not None:
                x_sol, y_sol, _, _, _, _ = variables
                env.viewer.ego_traj = np.stack((sol.value(x_sol), sol.value(y_sol)), axis=1)

            # 更新控制序列
            acc_sol, delta_sol = variables[4], variables[5]
            ego_actions = np.stack((sol.value(acc_sol), sol.value(delta_sol)), axis=1)

            # 执行控制动作
            action = ego_actions[0]
            print(f"控制动作 - 加速度: {action[0]:.2f} m/s², 转向: {np.degrees(action[1]):.1f}°")
            print(
                f"策略评分: {strategy_result['current_lane_score']:.2f}, 威胁惩罚: {strategy_result['threat_penalty']:.2f}")
            print(f"变道评价标准: {strategy_evaluator.lane_evaluation_score:.2f} (累计奖励 + 当前惩罚)")
            print(f"保持车道奖励: {lane_keeping_reward:.2f}, 轨迹跟踪奖励: {trajectory_tracking_reward:.2f}")
            print(
                f"模糊控制: 倾向={strategy_result['fuzzy_tendency']:.2f}, 建议='{strategy_result['fuzzy_recommendation']}'")

            distance_to_target = target_info['target_x'] - current_x
            print(
                f"进度: 当前位置 x={current_x:.1f}m, 目标位置 x={target_info['target_x']:.1f}m, 距离={distance_to_target:.1f}m")
            print(f"速度: 当前={current_speed:.1f}m/s, 目标={target_info['target_speed']:.1f}m/s")

            obs, reward, done, truncated, _ = env.step(action)

            if done or truncated:
                print("仿真结束!")
                break

            ego_obs, other_obs = filter_obs(obs)

            # 渲染与数据保存
            if it % SIMULATION_CONFIG["image_save_frequency"] == 0:
                try:
                    render_img = env.render()
                    image_count += 1
                    cv2.imwrite(os.path.join(save_dir, f"img_{image_count:06d}.png"), render_img)
                    print(f"保存图像: img_{image_count:06d}.png")
                except Exception as e:
                    print(f"渲染失败，跳过图像保存: {e}")

        except Exception as e:
            print(f"MPC优化失败: {e}")
            # 使用紧急避障策略
            if len(other_obs_current) > 0:
                front_vehicles = [v for v in other_obs_current if v[1] > current_x]
                if front_vehicles:
                    closest_vehicle = min(front_vehicles, key=lambda v: v[1] - current_x)
                    distance_to_closest = closest_vehicle[1] - current_x

                    if distance_to_closest < 20.0:
                        backup_action = np.array([-3.0, 0.1])
                        print("执行紧急避障策略")
                    else:
                        backup_action = np.array([-1.0, 0.0])
                else:
                    backup_action = np.array([-0.5, 0.0])
            else:
                backup_action = np.array([-0.5, 0.0])

            obs, reward, done, truncated, _ = env.step(backup_action)
            ego_obs, other_obs = filter_obs(obs)

            if done or truncated:
                print("仿真结束!")
                break

    # ==================== 仿真后处理模块 ====================
    env.close()
    total_time = time.time() - start_time
    final_target_info = target_manager.get_current_target_info()

    print(f"仿真完成! 总时间: {total_time:.2f}s, 总图像数: {image_count}")
    print(f"最终进度: 完成 {final_target_info['progress_count']} 个目标点")
    print(f"总行驶距离: {final_target_info['total_distance']:.1f}m")
    print(f"最终策略评分: {strategy_evaluator.last_evaluation_score:.2f}")
    print(f"最终保持车道奖励: {strategy_evaluator.total_lane_keeping_reward:.2f}")
    print(f"最终车道: {strategy_evaluator.current_lane}")
    print(f"总变道次数: {len(lane_change_log)}")
    print(f"平均速度: {np.mean(speed_data):.1f}m/s")
    print(f"最大速度: {np.max(speed_data):.1f}m/s")

    # 评价系统统计
    if evaluation_history:
        avg_evaluation = np.mean(evaluation_history)
        min_evaluation = np.min(evaluation_history)
        max_evaluation = np.max(evaluation_history)
        final_cumulative_reward = cumulative_reward_history[-1] if cumulative_reward_history else 0
        final_threat_penalty = threat_penalty_history[-1] if threat_penalty_history else 0

        print(f"评价系统统计:")
        print(f"  - 平均评价分数: {avg_evaluation:.2f}")
        print(f"  - 最小评价分数: {min_evaluation:.2f}")
        print(f"  - 最大评价分数: {max_evaluation:.2f}")
        print(f"  - 最终累计奖励: {final_cumulative_reward:.2f}")
        print(f"  - 最终威胁惩罚: {final_threat_penalty:.2f}")
        print(f"  - 变道触发次数: {len(lane_change_log)}")

        # 分析评价分数构成
        positive_evaluations = len([e for e in evaluation_history if e > 0])
        negative_evaluations = len([e for e in evaluation_history if e < 0])
        print(f"  - 正评价比例: {positive_evaluations / len(evaluation_history) * 100:.1f}%")
        print(f"  - 负评价比例: {negative_evaluations / len(evaluation_history) * 100:.1f}%")

    # 障碍物检测统计
    if obstacle_detection_log:
        min_distance = min([log['distance'] for log in obstacle_detection_log if log['distance'] > 0])
        avg_distance = np.mean([log['distance'] for log in obstacle_detection_log if log['distance'] > 0])
        avg_other_speed = np.mean([log['other_speed'] for log in obstacle_detection_log])
        avg_ego_speed = np.mean([log['ego_speed'] for log in obstacle_detection_log])
        avg_speed_difference = np.mean([log['speed_difference'] for log in obstacle_detection_log])

        print(f"障碍物检测统计:")
        print(f"  - 最小距离: {min_distance:.1f}m")
        print(f"  - 平均距离: {avg_distance:.1f}m")
        print(f"  - 平均其他车辆速度: {avg_other_speed:.1f}m/s")
        print(f"  - 平均自车速度: {avg_ego_speed:.1f}m/s")
        print(f"  - 平均速度差: {avg_speed_difference:.1f}m/s")
        print(f"  - 速度优势: {avg_ego_speed - avg_other_speed:.1f}m/s")

    # 车道中心保持统计
    lane_errors = []
    for x, y in trajectory_data:
        lane_num, lane_center = strategy_evaluator.calculate_lane_center(y)
        lane_error = abs(y - lane_center)
        lane_errors.append(lane_error)

    avg_lane_error = np.mean(lane_errors)
    max_lane_error = np.max(lane_errors)
    print(f"车道中心保持统计:")
    print(f"  - 平均车道误差: {avg_lane_error:.3f}m")
    print(f"  - 最大车道误差: {max_lane_error:.3f}m")
    print(f"  - 优秀保持率: {len([e for e in lane_errors if e < 0.1]) / len(lane_errors) * 100:.1f}%")

    # 奖励统计
    total_lane_keeping_reward = strategy_evaluator.total_lane_keeping_reward
    total_trajectory_reward = strategy_evaluator.trajectory_tracking_reward
    total_reward = total_lane_keeping_reward + total_trajectory_reward
    print(f"奖励统计:")
    print(f"  - 总保持车道奖励: {total_lane_keeping_reward:.2f}")
    print(f"  - 总轨迹跟踪奖励: {total_trajectory_reward:.2f}")
    print(f"  - 总奖励: {total_reward:.2f}")

    # 获取策略评分历史
    strategy_scores_history = strategy_evaluator.get_strategy_scores_history()
    fuzzy_tendency_history = strategy_evaluator.get_fuzzy_tendency_history()

    # 绘制结果图表
    plot_simulation_results(trajectory_data, speed_data, time_data, lane_change_log,
                            strategy_scores_history, fuzzy_tendency_history, save_dir)

    # 额外绘制评价系统分析图
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(time_data, evaluation_history, 'b-', linewidth=2, label='评价分数')
    plt.axhline(y=0, color='red', linestyle='--', label='零线')
    plt.axhline(y=STRATEGY_CONFIG["must_change_threshold"], color='orange', linestyle='--', label='变道阈值')
    plt.xlabel('时间 (s)')
    plt.ylabel('评价分数')
    plt.title('评价分数随时间变化')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(time_data, cumulative_reward_history, 'g-', linewidth=2, label='累计奖励')
    plt.plot(time_data, threat_penalty_history, 'r-', linewidth=2, label='威胁惩罚')
    plt.xlabel('时间 (s)')
    plt.ylabel('数值')
    plt.title('奖励与惩罚分解')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    # 标记变道点
    if lane_change_log:
        lc_times = [time_data[min(it, len(time_data) - 1)] for it, _, _, _ in lane_change_log]
        lc_scores = [evaluation_history[min(it, len(evaluation_history) - 1)] for it, _, _, _ in lane_change_log]
        plt.scatter(lc_times, lc_scores, c='red', s=50, marker='*', label='变道点')
    plt.plot(time_data, evaluation_history, 'b-', linewidth=2, label='评价分数')
    plt.xlabel('时间 (s)')
    plt.ylabel('评价分数')
    plt.title('变道点与评价分数关系')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(time_data, fuzzy_tendency_data, 'purple', linewidth=2, label='模糊倾向')
    plt.axhline(y=0, color='red', linestyle='--', label='中性')
    plt.axhline(y=0.3, color='green', linestyle='--', label='变道阈值')
    plt.xlabel('时间 (s)')
    plt.ylabel('模糊倾向')
    plt.title('模糊控制倾向分析')
    plt.grid(True)
    plt.legend()

    plt.suptitle('评价系统分析 - 累计奖励 + 当前惩罚', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '评价系统分析.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print("Fuzzy-MPC仿真分析图表保存成功")


if __name__ == "__main__":
    main()