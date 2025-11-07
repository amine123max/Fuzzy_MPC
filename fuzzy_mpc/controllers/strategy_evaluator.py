import math
from .fuzzy_controller import FuzzyController


class StrategyEvaluator:
    def __init__(self, strategy_config):
        self.strategy_config = strategy_config
        self.lane_keeping_distance = 0.0
        self.total_lane_keeping_reward = 0.0  # 累积奖励
        self.current_lane = 0
        self.last_evaluation_score = 0.0
        self.lane_change_cooldown = 0
        self.trajectory_tracking_reward = 0.0  # 累积轨迹跟踪奖励
        self.evaluation_reset_pending = False
        self.last_lane_change_iteration = -100

        # 变道评价相关属性
        self.lane_evaluation_score = 0.0
        self.evaluation_reset_distance = 200.0
        self.distance_since_last_reset = 0.0
        self.strategy_scores_history = []

    def calculate_lane_center(self, y_position):
        """根据y位置计算当前车道编号和车道中央位置"""
        lane_width = 4.0
        lane_num = int(y_position // lane_width)  # 使用整数除法确定车道
        lane_num = max(0, min(lane_num, 2))  # 限制在0-2车道范围内
        # 车道中央位置 = 车道编号 * 车道宽度 + 车道宽度/2
        lane_center = (lane_num + 0.17) * lane_width
        return lane_num, lane_center

    def count_vehicles_in_lane(self, target_lane, ego_x, other_vehicles, look_ahead=200.0):
        """统计目标车道上的车辆数量"""
        if other_vehicles is None or len(other_vehicles) == 0:
            return 0

        count = 0
        for vehicle in other_vehicles:
            other_x, other_y = vehicle[1], vehicle[2]
            other_lane, _ = self.calculate_lane_center(other_y)

            if (other_lane == target_lane and
                    other_x > ego_x and
                    other_x - ego_x < look_ahead):
                count += 1
        return count

    def update_lane_evaluation(self, distance_traveled, current_threat_penalty, lane_keeping_reward):
        """更新变道评价标准 - 修复：累积奖励 - 当前惩罚"""
        self.distance_since_last_reset += distance_traveled

        # 关键修复：累积奖励 - 当前惩罚
        # total_reward 是累积的奖励，current_threat_penalty 是当前的瞬时惩罚
        total_reward = self.total_lane_keeping_reward + self.trajectory_tracking_reward
        self.lane_evaluation_score = total_reward + current_threat_penalty  # 惩罚为负值，所以是相加

        self.strategy_scores_history.append(self.lane_evaluation_score)

        if self.distance_since_last_reset >= self.evaluation_reset_distance:
            self.reset_evaluation()
            return True
        return False

    def should_change_lane(self):
        """判断是否必须变道"""
        return self.lane_evaluation_score < self.strategy_config["must_change_threshold"]

    def update_lane_keeping_reward(self, current_y, distance_traveled):
        """更新保持车道奖励 - 修复：正确累积奖励"""
        _, lane_center = self.calculate_lane_center(current_y)
        self.current_lane = _

        if abs(current_y - lane_center) < 0.5:
            self.lane_keeping_distance += distance_traveled

            if self.lane_keeping_distance >= self.strategy_config["lane_keeping_interval"]:
                reward_count = int(self.lane_keeping_distance // self.strategy_config["lane_keeping_interval"])
                reward = reward_count * self.strategy_config["lane_keeping_reward"]
                self.total_lane_keeping_reward += reward  # 累积奖励
                self.lane_keeping_distance %= self.strategy_config["lane_keeping_interval"]
                print(f"获得保持车道奖励: {reward:.2f}, 累计奖励: {self.total_lane_keeping_reward:.2f}")
                return reward
        else:
            self.lane_keeping_distance = 0

        return 0.0

    def update_trajectory_tracking_reward(self, current_y, target_y):
        """更新轨迹跟踪奖励 - 修复：正确累积奖励"""
        tracking_error = abs(current_y - target_y)
        if tracking_error < 0.2:
            reward = self.strategy_config["trajectory_tracking_reward"]
            self.trajectory_tracking_reward += reward  # 累积奖励
            print(f"获得轨迹跟踪奖励: {reward:.2f}, 累计奖励: {self.trajectory_tracking_reward:.2f}")
            return reward
        return 0.0

    def assess_threat_penalty(self, ego_x, ego_y, other_vehicles):
        """评估前方车辆威胁惩罚 - 返回当前瞬时惩罚"""
        if other_vehicles is None or len(other_vehicles) == 0:
            return 0.0, None

        max_penalty = 0.0
        most_threatening_vehicle = None

        for vehicle in other_vehicles:
            other_x, other_y = vehicle[1], vehicle[2]
            ego_lane, _ = self.calculate_lane_center(ego_y)
            other_lane, _ = self.calculate_lane_center(other_y)

            # 放宽车道匹配条件，考虑相邻车道的影响
            if abs(ego_lane - other_lane) <= 1:
                distance_x = other_x - ego_x

                if distance_x > 0 and distance_x < 150.0:
                    penalty = self.calculate_penalty_by_distance(distance_x)

                    if penalty < max_penalty:
                        max_penalty = penalty
                        most_threatening_vehicle = vehicle

        if max_penalty < 0:
            print(f"当前威胁惩罚: {max_penalty:.2f}, 距离: {most_threatening_vehicle[1] - ego_x:.1f}m")

        return max_penalty, most_threatening_vehicle

    def calculate_penalty_by_distance(self, distance):
        """根据距离计算威胁惩罚 - 距离越小惩罚越大"""
        threat_distances = self.strategy_config["threat_distances"]
        threat_penalties = self.strategy_config["threat_penalties"]

        if distance >= threat_distances[0]:
            return threat_penalties[0]
        elif distance >= threat_distances[1]:
            t = (distance - threat_distances[1]) / (threat_distances[0] - threat_distances[1])
            return threat_penalties[1] * (1 - t) + threat_penalties[0] * t
        elif distance >= threat_distances[2]:
            t = (distance - threat_distances[2]) / (threat_distances[1] - threat_distances[2])
            return threat_penalties[2] * (1 - t) + threat_penalties[1] * t
        elif distance >= threat_distances[3]:
            t = (distance - threat_distances[3]) / (threat_distances[2] - threat_distances[3])
            return threat_penalties[3] * (1 - t) + threat_penalties[2] * t
        else:
            return threat_penalties[4]

    def calculate_lane_change_cost(self, from_lane, to_lane, ego_x, ego_y):
        """计算变道成本"""
        base_cost = self.strategy_config["lane_change_cost"]
        lane_diff = abs(to_lane - from_lane)
        return base_cost * lane_diff

    def evaluate_lane_safety(self, target_lane, ego_x, ego_y, other_vehicles):
        """评估目标车道安全性"""
        safety_score = 0.0
        ego_lane, _ = self.calculate_lane_center(ego_y)

        if target_lane != ego_lane:
            lane_change_cost = self.calculate_lane_change_cost(ego_lane, target_lane, ego_x, ego_y)
            safety_score += lane_change_cost

        vehicle_count = self.count_vehicles_in_lane(target_lane, ego_x, other_vehicles,
                                                    look_ahead=self.strategy_config["look_ahead_distance"])
        safety_score -= vehicle_count * self.strategy_config["vehicle_count_weight"]

        # 评估目标车道上的具体威胁
        if other_vehicles is not None and len(other_vehicles) > 0:
            for vehicle in other_vehicles:
                other_x, other_y = vehicle[1], vehicle[2]
                other_lane, _ = self.calculate_lane_center(other_y)

                if other_lane == target_lane:
                    distance_x = other_x - ego_x
                    distance_abs = abs(distance_x)

                    if distance_x > 0 and distance_abs < 100.0:
                        penalty = self.calculate_penalty_by_distance(distance_abs)
                        safety_score += penalty

        return safety_score

    def comprehensive_strategy_evaluation(self, ego_x, ego_y, other_vehicles, target_manager, current_iteration):
        """综合策略评价"""
        if target_manager.should_reset_evaluation(
                self.strategy_config["reset_interval"]) or self.evaluation_reset_pending:
            self.reset_evaluation()
            target_manager.reset_evaluation()
            self.evaluation_reset_pending = False
            print("评价标准重置")

        current_lane, lane_center = self.calculate_lane_center(ego_y)
        current_lane_threat_penalty, _ = self.assess_threat_penalty(ego_x, ego_y, other_vehicles)
        current_lane_safety = self.evaluate_lane_safety(current_lane, ego_x, ego_y, other_vehicles)
        current_lane_score = current_lane_threat_penalty + current_lane_safety

        lanes_count = 3
        lane_scores = {}
        lane_vehicle_counts = {}

        for target_lane in range(lanes_count):
            vehicle_count = self.count_vehicles_in_lane(target_lane, ego_x, other_vehicles)
            lane_vehicle_counts[target_lane] = vehicle_count

            if target_lane == current_lane:
                lane_scores[target_lane] = current_lane_score
            else:
                lane_score = self.evaluate_lane_safety(target_lane, ego_x, ego_y, other_vehicles)
                lane_scores[target_lane] = lane_score

        best_lane = max(lane_scores.items(), key=lambda x: (x[1], -lane_vehicle_counts[x[0]]))
        best_score = best_lane[1]
        best_lane_num = best_lane[0]

        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1

        should_change = False
        target_lane_center = (current_lane + 0.5) * 4.0  # 车道中央位置

        iterations_since_last_change = current_iteration - self.last_lane_change_iteration
        min_change_interval = 30

        # 关键修复：使用正确的评价分数进行变道判断
        lane_change_advantage = best_score - current_lane_score
        should_change_due_to_evaluation = self.should_change_lane()

        print(f"变道条件检查:")
        print(
            f"  当前评价分数: {self.lane_evaluation_score:.2f} (累计奖励: {self.total_lane_keeping_reward + self.trajectory_tracking_reward:.2f} + 当前惩罚: {current_lane_threat_penalty:.2f})")
        print(f"  变道阈值: {self.strategy_config['must_change_threshold']}")
        print(f"  当前车道评分: {current_lane_score:.2f}, 最佳车道评分: {best_score:.2f}")
        print(f"  变道优势: {lane_change_advantage:.2f}")

        if (should_change_due_to_evaluation and
                best_lane_num != current_lane and
                lane_change_advantage > self.strategy_config["min_lane_change_advantage"] and
                self.lane_change_cooldown == 0 and
                iterations_since_last_change >= min_change_interval):
            should_change = True
            target_lane_center = (best_lane_num + 0.5) * 4.0  # 目标车道中央位置
            self.lane_change_cooldown = 15
            self.last_lane_change_iteration = current_iteration
            self.evaluation_reset_pending = True
            print(f"执行变道: 当前评价分数 {self.lane_evaluation_score:.2f} 低于阈值")

        self.last_evaluation_score = current_lane_score

        return {
            'should_change_lane': should_change,
            'target_lane': target_lane_center,
            'current_lane_score': current_lane_score,
            'best_lane_score': best_score,
            'best_lane': best_lane_num,
            'all_lane_scores': lane_scores,
            'lane_vehicle_counts': lane_vehicle_counts,
            'threat_penalty': current_lane_threat_penalty,
            'lane_change_advantage': lane_change_advantage,
            'cumulative_evaluation': self.lane_evaluation_score,
            'cumulative_reward': self.total_lane_keeping_reward + self.trajectory_tracking_reward
        }

    def reset_evaluation(self):
        """重置评价状态"""
        self.lane_keeping_distance = 0.0
        self.total_lane_keeping_reward = 0.0  # 重置累积奖励
        self.trajectory_tracking_reward = 0.0  # 重置累积奖励
        self.lane_evaluation_score = 0.0
        self.distance_since_last_reset = 0.0
        print("评价状态重置 - 所有累积奖励清零")

    def get_strategy_scores_history(self):
        """获取策略评分历史"""
        return self.strategy_scores_history


class FuzzyStrategyEvaluator(StrategyEvaluator):
    def __init__(self, strategy_config):
        super().__init__(strategy_config)
        self.fuzzy_controller = FuzzyController()
        self.fuzzy_tendency_history = []
        self.last_fuzzy_recommendation = ""

    def fuzzy_comprehensive_evaluation(self, ego_x, ego_y, ego_speed, other_vehicles, current_iteration):
        """基于模糊控制的综合策略评价"""
        current_lane, lane_center = self.calculate_lane_center(ego_y)

        # 计算当前车道的威胁惩罚
        current_lane_threat_penalty, threat_vehicle = self.assess_threat_penalty(ego_x, ego_y, other_vehicles)
        current_lane_safety = self.evaluate_lane_safety(current_lane, ego_x, ego_y, other_vehicles)
        current_lane_score = current_lane_threat_penalty + current_lane_safety

        closest_threat_distance = self._get_closest_threat_distance(ego_x, ego_y, other_vehicles)

        # 使用累积评价分数作为变道评价标准
        lc_evaluation = self.lane_evaluation_score

        # 获取模糊控制倾向 - 使用新的接口
        fuzzy_tendency = self.fuzzy_controller.infer_lane_change_tendency(
            distance=closest_threat_distance,
            lc_evaluation=lc_evaluation
        )

        self.fuzzy_tendency_history.append(fuzzy_tendency)
        self.last_fuzzy_recommendation = self.fuzzy_controller.get_fuzzy_recommendation(fuzzy_tendency)

        lanes_count = 3
        lane_scores = {}
        lane_fuzzy_scores = {}
        lane_vehicle_counts = {}

        for target_lane in range(lanes_count):
            vehicle_count = self.count_vehicles_in_lane(target_lane, ego_x, other_vehicles)
            lane_vehicle_counts[target_lane] = vehicle_count

            if target_lane == current_lane:
                lane_scores[target_lane] = current_lane_score
                lane_fuzzy_scores[target_lane] = current_lane_score + fuzzy_tendency * 0.3
            else:
                lane_score = self.evaluate_lane_safety(target_lane, ego_x, ego_y, other_vehicles)
                lane_scores[target_lane] = lane_score
                lane_fuzzy_scores[target_lane] = lane_score + fuzzy_tendency * 0.8

        best_lane = max(lane_fuzzy_scores.items(), key=lambda x: (x[1], -lane_vehicle_counts[x[0]]))
        best_fuzzy_score = best_lane[1]
        best_lane_num = best_lane[0]

        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1

        should_change = False
        target_lane_center = (current_lane + 0.5) * 4.0  # 当前车道中央位置

        iterations_since_last_change = current_iteration - self.last_lane_change_iteration
        min_change_interval = 20

        # 模糊变道条件
        fuzzy_change_condition = (
                fuzzy_tendency > 0.3 or
                (fuzzy_tendency > 0.1 and self.should_change_lane())
        )

        current_fuzzy_score = lane_fuzzy_scores.get(current_lane, 0)
        lane_change_advantage = best_fuzzy_score - current_fuzzy_score

        print(f"模糊变道条件检查:")
        print(f"  模糊倾向: {fuzzy_tendency:.2f}, 条件: {fuzzy_change_condition}")
        print(f"  累积评价分数: {self.lane_evaluation_score:.2f}")
        print(f"  最佳车道: {best_lane_num}, 当前车道: {current_lane}")
        print(f"  变道优势: {lane_change_advantage:.2f}")
        print(f"  冷却时间: {self.lane_change_cooldown}, 间隔: {iterations_since_last_change}")

        if (fuzzy_change_condition and
                best_lane_num != current_lane and
                lane_change_advantage > self.strategy_config["min_lane_change_advantage"] and
                self.lane_change_cooldown == 0 and
                iterations_since_last_change >= min_change_interval):
            should_change = True
            target_lane_center = (best_lane_num + 0.5) * 4.0  # 目标车道中央位置
            self.lane_change_cooldown = 10
            self.last_lane_change_iteration = current_iteration
            self.evaluation_reset_pending = True

            print(f"执行模糊控制变道决策")
            print(f"  模糊倾向: {fuzzy_tendency:.2f}, 建议: '{self.last_fuzzy_recommendation}'")
            print(f"  当前模糊评分: {current_fuzzy_score:.2f}, 最佳模糊评分: {best_fuzzy_score:.2f}")
            print(f"  变道优势: {lane_change_advantage:.2f}")
            print(f"  目标车道: {best_lane_num} (Y={target_lane_center:.1f})")

        self.last_evaluation_score = current_lane_score

        return {
            'should_change_lane': should_change,
            'target_lane': target_lane_center,
            'current_lane_score': current_lane_score,
            'best_lane_score': best_fuzzy_score,
            'best_lane': best_lane_num,
            'all_lane_scores': lane_scores,
            'lane_vehicle_counts': lane_vehicle_counts,
            'threat_penalty': current_lane_threat_penalty,
            'fuzzy_tendency': fuzzy_tendency,
            'fuzzy_recommendation': self.last_fuzzy_recommendation,
            'closest_threat_distance': closest_threat_distance,
            'lane_change_advantage': lane_change_advantage,
            'cumulative_evaluation': self.lane_evaluation_score,
            'cumulative_reward': self.total_lane_keeping_reward + self.trajectory_tracking_reward
        }

    def _get_closest_threat_distance(self, ego_x, ego_y, other_vehicles):
        """获取最近威胁车辆的距离"""
        if other_vehicles is None or len(other_vehicles) == 0:
            return 500.0

        min_distance = float('inf')
        for vehicle in other_vehicles:
            other_x, other_y = vehicle[1], vehicle[2]
            ego_lane, _ = self.calculate_lane_center(ego_y)
            other_lane, _ = self.calculate_lane_center(other_y)

            if abs(ego_lane - other_lane) <= 1 and other_x > ego_x:
                distance = other_x - ego_x
                if distance < min_distance:
                    min_distance = distance

        return min_distance if min_distance != float('inf') else 500.0

    def get_fuzzy_tendency_history(self):
        """获取模糊倾向历史"""
        return self.fuzzy_tendency_history


# 添加导出语句
__all__ = ['StrategyEvaluator', 'FuzzyStrategyEvaluator']