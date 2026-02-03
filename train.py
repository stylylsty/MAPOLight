import os
import shutil

# 1. 强制使用 LibSumo 加速 (无界面模式)
os.environ["LIBSUMO_AS_TRACI"] = "True"

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune import Tuner
from ray.air import RunConfig, CheckpointConfig
from env.SignalEnv import env as signal_env
from utils.mypettingzoo import MyPettingZooEnv

# 2. 路径处理
net_file = os.path.abspath("sumo_files/single.net.xml")
route_file = os.path.abspath("sumo_files/1groutes.xml")
co = False

# 3. 创建一个临时环境来获取智能体列表
print("Generating dummy environment to fetch agent list...")
dummy_env = signal_env(
    net_file=net_file,
    route_file=route_file,
    pr=1,
    use_gui=False,
    begin_time=10,
    num_seconds=3600,
    max_depart_delay=0,
    time_to_teleport=-1,
    delta_time=5,
    yellow_time=3,
    min_green=5,
    max_green=50,
    reward_fn="average-speed",
    sumo_warnings=False, # 训练时关掉警告
    cav_env=False,
    cav_compare=False,
    collaborate=co,
)
# 获取智能体列表，并转为集合 (Set) 供 RLlib 使用
agents = list(dummy_env.possible_agents)
print(f"Agents detected: {agents}")
dummy_env.close() 

# 4. 注册训练环境
register_env(
    "signal_env",
    lambda _: MyPettingZooEnv(
        signal_env(
            net_file=net_file,
            route_file=route_file,
            pr=1,
            use_gui=False,
            begin_time=10,
            num_seconds=3600,
            max_depart_delay=0,
            time_to_teleport=-1,
            delta_time=5,
            yellow_time=3,
            min_green=5,
            max_green=50,
            reward_fn="average-speed",
            sumo_warnings=False, 
            cav_env=False,
            cav_compare=False,
            collaborate=co,
            accident_num=2,
            accident_duration=600,
        )
    ),
)

# 5. 注册评估环境
register_env(
    "eval_signal_env",
    lambda _: MyPettingZooEnv(
        signal_env(
            net_file=net_file,
            route_file=route_file,
            pr=1,
            use_gui=False,
            begin_time=10,
            num_seconds=3600,
            max_depart_delay=0,
            time_to_teleport=-1,
            delta_time=5,
            yellow_time=3,
            min_green=5,
            max_green=50,
            reward_fn="average-speed",
            sumo_warnings=False,
            cav_env=False,
            cav_compare=False,
            collaborate=co,
            accident_edge=["-gneE10"],
            accident_duration=600,
            accident_time=["1800"],
        )
    ),
)

# 6. 配置 PPO 算法 (核心修复部分)
config = (
    PPOConfig()
    .environment("signal_env")
    .framework("torch") # 明确指定使用 PyTorch
    
    # ==========================================================
    # 【核心修复 1】将 GPU 数量改为 0
    # 您的环境检测不到 GPU，必须设为 0 才能运行，否则会一直 PENDING
    # ==========================================================
    .resources(num_gpus=0) 
    
    # ==========================================================
    # 【核心修复 2】强制关闭 RLModule API 和 Learner API
    # 解决 num_env_steps_trained = 0 (不训练) 的问题
    # ==========================================================
    .rl_module(_enable_rl_module_api=False) 
    .training(
        _enable_learner_api=False, 
        train_batch_size=4000,   # 凑齐 4000 步就开始训练
        sgd_minibatch_size=128,  # 每次梯度下降的数据量
        num_sgd_iter=30,         # 每次训练循环 30 次
        lr=5e-5,                 # 学习率
        model={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
        }
    )
    # ==========================================================

    .rollouts(
        num_rollout_workers=5, 
        rollout_fragment_length="auto" 
    )
    .multi_agent(
        policies=set(agents), # 传入集合
        # 确保 agent_id 能正确对应到 policy
        policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
    )
    .evaluation(
        evaluation_interval=10, # 每 10 轮评估一次
        evaluation_duration=1,
        evaluation_config={"env": "eval_signal_env"},
        evaluation_parallel_to_training=False, # 避免多进程冲突
    )
)

# 7. 开始训练
print("Starting training...")
Tuner(
    trainable="PPO",
    param_space=config.to_dict(),
    run_config=RunConfig(
        verbose=1, # 减少日志输出，只看关键信息
        stop={
            "episode_reward_mean": 500, # 目标分数：500分自动停止
            "training_iteration": 500,  # 保底：跑500轮自动停止
        },
        checkpoint_config=CheckpointConfig(
            num_to_keep=3, # 只保留最新的3个模型，节省硬盘
            checkpoint_frequency=10,
        ),
    ),
).fit()

import numpy as np
np.bool8 = np.bool_