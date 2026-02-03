# MAPOLight: Robust Traffic Signal Control with PPO & SUMO

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![SUMO](https://img.shields.io/badge/Simulator-SUMO-orange)
![Ray](https://img.shields.io/badge/RL-Ray%2FRLlib-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📖 Introduction (项目简介)

**MAPOLight** 是一个基于深度强化学习 (Deep Reinforcement Learning) 的智能交通信号控制系统。项目利用 **Ray/RLlib** 框架中的 **PPO (Proximal Policy Optimization)** 算法，在 **SUMO** 仿真环境中训练智能体以优化交通流。

与传统控制系统不同，本项目集成了**事故模拟机制 (Accident Simulation)**，能够测试并训练 AI 在极端路况（如车道因事故封闭）下的应急疏导能力，从而实现更具鲁棒性的交通控制。

## ✨ Key Features (核心功能)

* **PPO 算法控制**：使用稳定且高效的 PPO 算法进行交通信号决策。
* **事故模拟机制**：支持随机或固定时间/地点生成故障车辆，模拟真实道路事故场景。
* **极速训练 (Headless)**：利用 `LibSumo` 技术，无需图形界面即可进行高并发训练。
* **可视化评估**：提供带有 SUMO GUI 的评估脚本，直观展示 AI 在事故场景下的表现。
* **兼容性修复**：针对 Ray/RLlib 新版 API 进行了适配，确保在最新环境下稳定运行。
* **CAV 扩展接口**：预留了网联车 (CAV) 感知和多灯协同 (Cooperative Control) 的代码接口。

## 📂 Project Structure (项目结构)

```text
MAPOLight/
├── train.py                # [核心] 训练脚本：定义 PPO 配置、训练循环及停止条件
├── evaluate.py             # [核心] 评估脚本：加载模型并在 GUI 中演示效果
├── env/                    # 环境定义模块
│   ├── SignalEnv.py        # SUMO 环境主逻辑 (Step, Reset, 事故生成)
│   ├── TrafficSignal.py    # 智能体行为定义 (动作空间、状态空间)
│   └── networkdata.py      # 路网数据解析工具
├── utils/                  # 工具包
│   └── mypettingzoo.py     # PettingZoo 到 RLlib 的环境适配器
├── sumo_files/             # SUMO 仿真文件
│   ├── single.net.xml      # 路网拓扑结构
│   └── 1groutes.xml        # 交通流/路由定义
└── requirements.txt        # 项目依赖列表