# Fuzzy-MPC 高速公路自动驾驶控制系统

## 项目简介

基于模糊逻辑和模型预测控制(MPC)的高速公路环境智能车辆决策系统，实现安全高效的自动驾驶控制。

## 主要特性

-  **模型预测控制** - 精确的轨迹跟踪和车辆动力学控制
- **模糊逻辑决策** - 智能的变道决策和威胁评估
-  **综合评价系统** - 结合奖励和惩罚的动态评分机制
-  **实时安全监控** - 多障碍物检测和碰撞避免
-  **性能可视化** - 完整的仿真数据记录和分析

## 系统架构

fuzzy_mpc/
├── config/ # 配置参数模块
│ ├── mpc_config.py # MPC控制器参数
│ ├── strategy_config.py # 策略评价参数
│ └── simulation_config.py # 仿真环境参数
├── controllers/ # 控制器模块
│ ├── mpc_controller.py # MPC轨迹控制器
│ ├── fuzzy_controller.py # 模糊逻辑控制器
│ └── strategy_evaluator.py # 策略评价器
├── managers/ # 管理器模块
│ └── target_manager.py # 动态目标管理器
├── utils/ # 工具函数模块
│ ├── math_utils.py # 数学计算工具
│ └── visualization.py # 数据可视化工具
├── highway_env_mpc/ # 高速公路环境
├── main.py # 主程序入口
└── transformer.py # 视频生成工具

## 核心算法

### 模型预测控制 (MPC)
- 预测时域：20步
- 控制时域：20步
- 优化目标：轨迹跟踪 + 障碍物规避 + 控制平滑性

### 模糊决策系统
- 输入变量：威胁距离、变道评价分数
- 输出变量：变道倾向度
- 模糊规则库：基于专家经验的决策规则

### 动态策略评价
- 累积奖励机制：车道保持奖励 + 轨迹跟踪奖励
- 实时威胁惩罚：基于距离的威胁等级评估
- 综合评价：奖励 - 惩罚的动态评分

## 安装要求

```bash
# 核心依赖
Python 3.8+
gymnasium
casadi
numpy
matplotlib

# 可视化依赖
pygame
opencv-python

# 安装命令
pip install -r requirements.txt
```

## 快速开始

### 基本运行

```
# 启动Fuzzy-MPC控制系统
python main.py
```

## 引用说明

如果您在研究或项目中使用本代码，请注明来源
