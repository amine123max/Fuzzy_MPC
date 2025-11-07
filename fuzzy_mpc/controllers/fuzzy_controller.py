class FuzzyController:
    def __init__(self):
        # 模糊集定义 - 根据需求文档调整
        self.distance_sets = {
            'VF': (150, 200),  # 非常远 150-200m
            'F': (100, 150),  # 远 100-150m
            'N': (50, 100),  # 近 50-100m
            'VN': (20, 50)  # 非常近 20-50m
        }

        # 变道评价标准模糊集
        self.lc_evaluation_sets = {
            'P': (0, 2),  # 正变道评价标准 >0
            'N': (-5, 0)  # 负变道评价标准 <0
        }

        # 模糊规则库 - 根据需求文档
        self.rules = self._build_fuzzy_rules()

    def _build_fuzzy_rules(self):
        """构建模糊规则库 - 根据需求文档"""
        rules = []

        # 变道综合评定标准：
        # 非常远+正变道评价标准=变道意愿
        # 近/非常近+负变道评价标准=高变道意愿

        # 规则格式: (距离, 变道评价) -> 变道倾向
        # 变道倾向: -2(强烈不建议), -1(不建议), 0(中性), 1(建议), 2(强烈建议)

        # 非常远 + 正评价 = 变道意愿
        rules.append((('VF', 'P'), -1))

        # 远 + 正评价 = 轻微变道意愿
        rules.append((('F', 'P'), 0))

        # 近 + 负评价 = 高变道意愿
        rules.append((('N', 'N'), 2))

        # 非常近 + 负评价 = 高变道意愿
        rules.append((('VN', 'N'), 2))

        # 其他情况保持中性或轻微不建议
        rules.append((('ANY', 'ANY'), 0))

        return rules

    def fuzzy_membership(self, value, fuzzy_set):
        """计算模糊隶属度"""
        low, high = fuzzy_set
        if value <= low:
            return 1.0 if value == low else 0.0
        elif value >= high:
            return 1.0 if value == high else 0.0
        else:
            # 简单的三角形隶属函数
            if value <= (low + high) / 2:
                return 1.0 - (value - low) / (high - low)
            else:
                return (value - low) / (high - low)

    def fuzzify_distance(self, distance):
        """距离模糊化"""
        membership = {}
        for set_name, set_range in self.distance_sets.items():
            membership[set_name] = self.fuzzy_membership(distance, set_range)
        return membership

    def fuzzify_lc_evaluation(self, evaluation):
        """变道评价标准模糊化"""
        membership = {}
        for set_name, set_range in self.lc_evaluation_sets.items():
            membership[set_name] = self.fuzzy_membership(evaluation, set_range)
        return membership

    def evaluate_rule(self, rule, distance_mf, lc_eval_mf):
        """评估单条规则"""
        conditions, output = rule
        dist_cond, lc_cond = conditions

        # 计算规则激活度
        activation = 1.0

        if dist_cond != 'ANY':
            activation = min(activation, distance_mf.get(dist_cond, 0))
        if lc_cond != 'ANY':
            activation = min(activation, lc_eval_mf.get(lc_cond, 0))

        return activation, output

    def infer_lane_change_tendency(self, distance, lc_evaluation):
        """推理变道倾向"""
        # 模糊化输入
        distance_mf = self.fuzzify_distance(distance)
        lc_eval_mf = self.fuzzify_lc_evaluation(lc_evaluation)

        print(f"模糊输入 - 距离: {distance:.1f}m, 变道评价: {lc_evaluation:.2f}")
        print(f"距离隶属度: { {k: f'{v:.2f}' for k, v in distance_mf.items() if v > 0} }")
        print(f"变道评价隶属度: { {k: f'{v:.2f}' for k, v in lc_eval_mf.items() if v > 0} }")

        # 应用模糊规则
        outputs = []
        activations = []

        for rule in self.rules:
            activation, output = self.evaluate_rule(rule, distance_mf, lc_eval_mf)
            if activation > 0.1:
                outputs.append(output)
                activations.append(activation)
                conditions, _ = rule
                print(f"规则激活: {conditions} -> {output}, 激活度: {activation:.2f}")

        if not outputs:
            print("没有规则激活，返回中性倾向")
            return 0

        # 加权平均去模糊化
        weighted_sum = sum(o * a for o, a in zip(outputs, activations))
        total_activation = sum(activations)

        result = weighted_sum / total_activation if total_activation > 0 else 0
        print(f"模糊推理结果: {result:.3f}")

        return result

    def get_fuzzy_recommendation(self, tendency):
        """获取模糊建议描述"""
        if tendency >= 1.3:
            return "高变道意愿"
        elif tendency >= 0.5:
            return "变道意愿"
        elif tendency >= 0.3:
            return "轻微变道意愿"
        elif tendency >= -0.5:
            return "保持当前车道"
        elif tendency >= -1.5:
            return "不建议变道"
        else:
            return "强烈不建议变道"