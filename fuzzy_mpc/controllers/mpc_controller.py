import casadi as csd
import numpy as np
from utils.math_utils import safe_smooth_max, safe_obstacle_penalty, predict_vehicle_trajectories, \
    safe_boundary_penalty, check_obstacle_collision


class MPCController:
    def __init__(self, config):
        self.config = config

    def build_optimization_problem(self, current_state, ref_trajectory, other_vehicles, strategy_result,
                                   lane_keeping_reward, trajectory_tracking_reward, dt):
        """构建MPC优化问题"""
        opti = csd.Opti()
        N = self.config.N

        # 控制变量和状态变量定义
        acc = opti.variable(N)
        delta = opti.variable(N)
        x = opti.variable(N + 1)
        y = opti.variable(N + 1)
        speed = opti.variable(N + 1)
        heading = opti.variable(N + 1)

        # 车辆动力学模型
        for k in range(N):
            x_next = x[k] + speed[k] * csd.cos(heading[k]) * dt
            y_next = y[k] + speed[k] * csd.sin(heading[k]) * dt
            heading_next = heading[k] + (speed[k] / self.config.L) * csd.tan(delta[k]) * dt
            speed_next = speed[k] + acc[k] * dt

            opti.subject_to(x[k + 1] == x_next)
            opti.subject_to(y[k + 1] == y_next)
            opti.subject_to(heading[k + 1] == heading_next)
            opti.subject_to(speed[k + 1] == speed_next)

        # 硬约束设置
        opti.subject_to(opti.bounded(self.config.a_min, acc, self.config.a_max))
        opti.subject_to(opti.bounded(self.config.delta_min, delta, self.config.delta_max))
        opti.subject_to(opti.bounded(self.config.v_min, speed, self.config.v_max))
        opti.subject_to(opti.bounded(-np.pi / 3, heading, np.pi / 3))
        opti.subject_to(opti.bounded(self.config.y_min, y, self.config.y_max))

        # 成本函数
        total_cost = self._build_cost_function(
            x, y, heading, speed, acc, delta, ref_trajectory,
            other_vehicles, strategy_result, lane_keeping_reward,
            trajectory_tracking_reward, dt, current_state
        )

        opti.minimize(total_cost)

        # 初始状态约束
        opti.subject_to(x[0] == current_state[0])
        opti.subject_to(y[0] == current_state[1])
        opti.subject_to(speed[0] == current_state[2])
        opti.subject_to(heading[0] == current_state[3])

        return opti, (x, y, heading, speed, acc, delta)

    def _build_cost_function(self, x, y, heading, speed, acc, delta, ref_trajectory,
                             other_vehicles, strategy_result, lane_keeping_reward,
                             trajectory_tracking_reward, dt, current_state):
        """构建成本函数 - 修复CasADi条件判断问题"""
        total_cost = 0
        N = self.config.N

        # 1. 轨迹跟踪成本
        for k in range(N):
            state_x = x[k]
            state_y = y[k]
            state_theta = heading[k]
            state_v = speed[k]

            ref_x = ref_trajectory[0, k]
            ref_y = ref_trajectory[1, k]
            ref_theta = ref_trajectory[2, k]
            ref_v = ref_trajectory[3, k]

            total_cost += self.config.w_x * (state_x - ref_x) ** 2
            total_cost += self.config.w_y * (state_y - ref_y) ** 2
            total_cost += self.config.w_theta * csd.sin((state_theta - ref_theta) / 2) ** 2
            total_cost += self.config.w_v * (state_v - ref_v) ** 2

        # 2. 控制平滑成本
        for k in range(N):
            control_acc = acc[k]
            control_delta = delta[k]
            total_cost += self.config.w_a * control_acc ** 2
            total_cost += self.config.w_delta * control_delta ** 2

            if k > 0:
                delta_dot = (delta[k] - delta[k - 1]) / dt
                total_cost += self.config.w_delta_dot * delta_dot ** 2

        # 3. 边界约束惩罚
        for k in range(N):
            state_y = y[k]
            total_cost += safe_boundary_penalty(state_y, self.config.y_min, self.config.y_max, self.config)

        # 4. 车道中央吸引力 - 修改为车道中央
        for k in range(N):
            state_y = y[k]
            lane_width = 4.0
            lane_num = csd.floor(state_y / lane_width)  # 确定当前车道
            lane_num = csd.fmax(0, csd.fmin(lane_num, 2))  # 限制在0-2车道
            lane_center = (lane_num + 0.17) * lane_width  # 车道中央位置
            lane_center_penalty = (state_y - lane_center) ** 2  # 使用平方惩罚
            total_cost += 50.0 * lane_center_penalty  # 增加权重确保车辆在车道中央

        # 5. 策略相关成本
        strategy_penalty = strategy_result['current_lane_score']
        if strategy_penalty < 0:
            total_cost += self.config.w_strategy_evaluation * abs(strategy_penalty)

        # 6. 奖励（负成本）
        if lane_keeping_reward > 0:
            total_cost -= self.config.w_lane_keep * lane_keeping_reward
        if trajectory_tracking_reward > 0:
            total_cost -= self.config.w_trajectory_tracking * trajectory_tracking_reward

        # 7. 障碍物规避 - 修复CasADi条件判断
        if other_vehicles is not None and len(other_vehicles) > 0:
            vehicle_predictions = predict_vehicle_trajectories(
                current_state[0], current_state[1], other_vehicles, N, dt
            )

            # 检查所有预测的障碍物
            max_obstacles_to_check = min(len(vehicle_predictions), self.config.max_obs)

            for k in range(N):  # 检查所有时间步
                for i in range(max_obstacles_to_check):
                    if k < vehicle_predictions[i].shape[1]:
                        other_x_pred = vehicle_predictions[i][0, k]
                        other_y_pred = vehicle_predictions[i][1, k]
                        other_heading = vehicle_predictions[i][2, k]

                        # 计算与障碍物的距离
                        distance_x = csd.fabs(x[k] - other_x_pred)
                        distance_y = csd.fabs(y[k] - other_y_pred)

                        # 使用CasADi兼容的距离检测 - 修复条件判断
                        # 创建距离权重函数，距离越近权重越大
                        distance_weight = csd.exp(-0.1 * (distance_x + distance_y))

                        # 只在合理距离内施加惩罚
                        penalty = safe_obstacle_penalty(
                            x[k], y[k], heading[k],
                            other_x_pred, other_y_pred, other_heading,
                            self.config.vehicle_length, self.config.vehicle_width, 1.0, self.config
                        )

                        # 使用距离权重来调整惩罚
                        weighted_penalty = penalty * distance_weight
                        total_cost += self.config.w_obstacle * weighted_penalty

                        # 额外添加近距离紧急避障惩罚
                        emergency_weight = csd.exp(-0.2 * (distance_x + distance_y))
                        emergency_penalty = 100.0 * emergency_weight
                        total_cost += emergency_penalty

        # 8. 终端成本
        final_state_x = x[N]
        final_state_y = y[N]
        final_state_theta = heading[N]
        final_state_v = speed[N]

        final_ref_x = ref_trajectory[0, N]
        final_ref_y = ref_trajectory[1, N]
        final_ref_theta = ref_trajectory[2, N]
        final_ref_v = ref_trajectory[3, N]

        total_cost += self.config.w_fx * (final_state_x - final_ref_x) ** 2
        total_cost += self.config.w_fy * (final_state_y - final_ref_y) ** 2
        total_cost += self.config.w_ftheta * csd.sin((final_state_theta - final_ref_theta) / 2) ** 2
        total_cost += self.config.w_fv * (final_state_v - final_ref_v) ** 2

        return total_cost

    def set_initial_guess(self, opti, variables, current_state, prev_actions, N, dt):
        """设置初始猜测"""
        x, y, heading, speed, acc, delta = variables

        if prev_actions is not None and len(prev_actions) > 0:
            opti.set_initial(acc, np.clip(prev_actions[:, 0], self.config.a_min, self.config.a_max))
            opti.set_initial(delta, np.clip(prev_actions[:, 1], self.config.delta_min, self.config.delta_max))
        else:
            opti.set_initial(acc, np.ones(N) * 0.5)
            opti.set_initial(delta, np.zeros(N))

        opti.set_initial(x, np.linspace(current_state[0], current_state[0] + current_state[2] * N * dt, N + 1))
        opti.set_initial(y, np.ones(N + 1) * current_state[1])
        opti.set_initial(speed, np.ones(N + 1) * current_state[2])
        opti.set_initial(heading, np.ones(N + 1) * current_state[3])

    def configure_solver(self, opti):
        """配置求解器"""
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": self.config.max_iter,
            "print_level": 0,
            "tol": 1e-3,
            "acceptable_tol": 1e-2,
            "bound_relax_factor": 1e-6,
            "mu_strategy": "adaptive"
        }
        opti.solver('ipopt', p_opts, s_opts)