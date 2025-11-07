import matplotlib.pyplot as plt
import numpy as np
import os

def plot_simulation_results(trajectory_data, speed_data, time_data, lane_change_log, 
                          strategy_scores, fuzzy_tendency_data, save_dir):
    """绘制中文图表 - 融合模糊意图预测的MPC控制"""
    plt.figure(figsize=(16, 12))

    # 轨迹图
    plt.subplot(2, 3, 1)
    trajectory_array = np.array(trajectory_data)
    plt.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b-', linewidth=2, label='车辆轨迹')

    # 标记变道点
    if lane_change_log:
        lc_array = np.array(lane_change_log)
        plt.scatter(lc_array[:, 1], lc_array[:, 2], c='red', s=50, marker='*', label='变道点')

    plt.xlabel('X位置 (m)')
    plt.ylabel('Y位置 (m)')
    plt.title('Fuzzy-MPC车辆轨迹和变道点')
    plt.grid(True)
    plt.legend()

    # 速度图
    plt.subplot(2, 3, 2)
    plt.plot(time_data, speed_data, 'r-', linewidth=2, label='速度')
    plt.xlabel('时间 (s)')
    plt.ylabel('速度 (m/s)')
    plt.title('车辆速度随时间变化')
    plt.grid(True)
    plt.legend()

    # 车道变化图
    plt.subplot(2, 3, 3)
    lane_numbers = [int(round(y / 4.0)) for _, y in trajectory_data]
    plt.plot(time_data, lane_numbers, 'g-', linewidth=2, label='车道编号')
    plt.xlabel('时间 (s)')
    plt.ylabel('车道编号')
    plt.title('车道变化历史')
    plt.grid(True)
    plt.legend()

    # 策略评价图
    plt.subplot(2, 3, 4)
    if len(strategy_scores) > len(time_data):
        strategy_scores = strategy_scores[:len(time_data)]
    elif len(strategy_scores) < len(time_data):
        strategy_scores.extend([0] * (len(time_data) - len(strategy_scores)))

    plt.plot(time_data, strategy_scores, 'purple', linewidth=2, label='策略评分')
    plt.axhline(y=0, color='red', linestyle='--', label='变道阈值')
    plt.xlabel('时间 (s)')
    plt.ylabel('策略评分')
    plt.title('车道策略评价 (惩罚+奖励)')
    plt.grid(True)
    plt.legend()

    # 模糊控制倾向图
    plt.subplot(2, 3, 5)
    if len(fuzzy_tendency_data) > len(time_data):
        fuzzy_tendency_data = fuzzy_tendency_data[:len(time_data)]
    elif len(fuzzy_tendency_data) < len(time_data):
        fuzzy_tendency_data.extend([0] * (len(time_data) - len(fuzzy_tendency_data)))

    plt.plot(time_data, fuzzy_tendency_data, 'orange', linewidth=2, label='模糊倾向')
    plt.axhline(y=0, color='red', linestyle='--', label='中性')
    plt.axhline(y=1, color='green', linestyle='--', label='建议变道')
    plt.axhline(y=-1, color='blue', linestyle='--', label='不建议变道')
    plt.xlabel('时间 (s)')
    plt.ylabel('模糊倾向')
    plt.title('模糊控制变道倾向分析')
    plt.grid(True)
    plt.legend()

    plt.suptitle('Fuzzy-MPC：融合模糊意图预测的自动驾驶模型预测控制', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fuzzy_MPC仿真分析结果.png'), dpi=300, bbox_inches='tight')
    plt.show()