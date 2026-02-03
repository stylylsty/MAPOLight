# MAPOLight: Robust Traffic Signal Control with PPO & SUMO

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![SUMO](https://img.shields.io/badge/Simulator-SUMO-orange)
![Ray](https://img.shields.io/badge/RL-Ray%2FRLlib-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Introduction（项目简介）

MAPOLight 是一个基于深度强化学习（Deep Reinforcement Learning）的交通信号控制系统。  
本项目使用 Ray / RLlib 框架中的 PPO（Proximal Policy Optimization）算法，在 SUMO 仿真环境中训练智能体，以优化交通流效率。

与传统方法不同，本项目引入了事故模拟机制（Accident Simulation），用于评估和提升模型在突发事故、车道封闭等极端交通场景下的鲁棒性。

---

## Key Features（核心功能）

- 基于 PPO 的交通信号控制
- 支持随机或固定位置的事故模拟
- 基于 LibSUMO 的无界面高速训练
- 支持 SUMO GUI 的可视化评估
- 兼容新版 Ray / RLlib API
- 预留 CAV / 多路口协同扩展接口

---

## Project Structure（项目结构）

MAPOLight/  
├── train.py                训练脚本（PPO 配置与训练流程）  
├── evaluate.py             评估脚本（SUMO GUI 可视化）  
├── env/                    环境定义模块  
│   ├── SignalEnv.py        SUMO 环境主逻辑  
│   ├── TrafficSignal.py    智能体定义  
│   └── networkdata.py      路网解析工具  
├── utils/                  工具模块  
│   └── mypettingzoo.py     PettingZoo 到 RLlib 适配器  
├── sumo_files/             SUMO 仿真文件  
│   ├── single.net.xml      路网结构  
│   └── 1groutes.xml        交通流定义  
└── requirements.txt        Python 依赖列表  

---

## Prerequisites（环境依赖）

### 操作系统
- 推荐 Linux（Ubuntu / WSL2）

### SUMO 仿真器
- 已正确安装 SUMO
- 已配置 SUMO_HOME 环境变量

### Python 依赖
建议在虚拟环境中安装以下依赖：

pip install ray[rllib] pettingzoo gymnasium sumolib traci torch pandas numpy

---

## Usage（使用方法）

### 模型训练

运行以下命令开始训练（默认不启用 GUI）：

python train.py

训练配置说明：
- 使用 CPU 训练（num_gpus = 0）
- 停止条件：平均奖励 ≥ 500 或训练轮数达到 500
- 模型保存路径：~/ray_results/PPO/

---

### 模型评估与可视化

使用训练完成的模型进行评估：

python evaluate.py --checkpoint 路径/到/checkpoint

示例：

python evaluate.py --checkpoint /home/wsl/ray_results/PPO/PPO_signal_env_xxxx/checkpoint_000500

说明：
- SUMO GUI 打开后，点击左上角绿色 Play 按钮开始仿真
- 默认在第 1800 秒，-gneE10 车道会发生模拟事故

---

## Configuration（关键参数）

accident_num（train.py）：事故数量  
accident_duration（train.py）：事故持续时间（秒）  
reward_fn（SignalEnv.py）：奖励函数类型（推荐 average-speed）  
_enable_learner_api（train.py）：必须设为 False  
num_gpus（train.py）：无 GPU 环境设为 0  

---

## Contributing

欢迎提交 Issue 或 Pull Request。

---

## License

MIT License
