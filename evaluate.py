import argparse
import os
import sys

# 1. 强制使用 LibSumo 加速 (虽然 evaluate 需要 GUI，但代码内部依赖这个标记)
# 注意：如果是为了看 GUI，稍后代码里会覆盖 use_gui=True
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from utils.mypettingzoo import MyPettingZooEnv
from env.SignalEnv import env as signal_env

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--net", type=str, default="single")
parser.add_argument("--pr", type=str, default="1")
parser.add_argument("--algorithm", type=str, default="ppo")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
args = parser.parse_args()

# 路径处理
net_file = os.path.abspath("sumo_files/single.net.xml")
route_file = os.path.abspath("sumo_files/1groutes.xml")

# 注册环境
register_env(
    "signal_env",
    lambda _: MyPettingZooEnv(
        signal_env(
            net_file=net_file,
            route_file=route_file,
            pr=1,
            # 【重点】评估时开启 GUI
            use_gui=True,  
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
            collaborate=False,
            # 固定事故位置，方便观察
            accident_edge=["-gneE10"],
            accident_duration=600,
            accident_time=["1800"],
        )
    ),
)

# 配置算法
# 【核心修复】必须与 train.py 保持完全一致的配置！
config = (
    PPOConfig()
    .environment("signal_env")
    .framework("torch")
    # 显式设置为 0，防止报错
    .resources(num_gpus=0)
    
    # ==========================================================
    # 【关键修复】这里必须和训练时一样关闭新 API
    # 否则无法加载 Checkpoint
    # ==========================================================
    .rl_module(_enable_rl_module_api=False)
    .training(
        _enable_learner_api=False,
        model={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
        }
    )
    # ==========================================================
    
    .rollouts(
        num_rollout_workers=1, # 评估时用 1 个 worker 即可
        rollout_fragment_length="auto"
    )
    .multi_agent(
        # 简单的策略映射，只要能跑通即可
        policy_mapping_fn=(lambda agent_id, *args, **kwargs: "gneJ2"), 
        policies={"gneJ2"}
    )
)

print(f"Loading checkpoint from: {args.checkpoint}")

# 构建算法并加载模型
algo = config.build()
algo.restore(args.checkpoint)

print("Starting evaluation loop...")

# 手动运行评估循环
env = MyPettingZooEnv(
    signal_env(
        net_file=net_file,
        route_file=route_file,
        pr=1,
        use_gui=True, # 确保 GUI 开启
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
        collaborate=False,
        accident_edge=["-gneE10"],
        accident_duration=600,
        accident_time=["1800"],
    )
)

obs, info = env.reset()
done = {"__all__": False}
total_reward = 0

while not done["__all__"]:
    # 获取动作
    # 注意：MyPettingZooEnv 的 obs 是个字典 {agent_id: obs}
    # compute_actions 需要接收字典
    actions = {}
    for agent_id, agent_obs in obs.items():
        # 单个智能体计算动作
        action = algo.compute_single_action(agent_obs, policy_id="gneJ2")
        actions[agent_id] = action

    # 环境推演
    obs, reward, terminated, truncated, info = env.step(actions)
    
    # 统计奖励
    for r in reward.values():
        total_reward += r
        
    done = terminated # PettingZoo 的 terminated 包含了所有智能体的结束状态

print(f"Evaluation finished. Total Reward: {total_reward}")
env.close()