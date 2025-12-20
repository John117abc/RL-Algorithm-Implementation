import gymnasium as gym
import highway_env
import numpy as np
import torch

from AgentHighway import AgentHighWay
# render_mode="rgb_array",
env = gym.make("highway-v0",render_mode="rgb_array",config={
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted"
        },
        "action": {
            "type": "DiscreteMetaAction", # 离散型动作
        },
        "vehicles_density":1.5,
        "lanes_count": 4,   # 车道数
        "vehicles_count": 30,   # 车辆数
        "duration": 200,  # [s]  # 时间s
        "initial_spacing": 2,   # 初始间距
        "collision_reward": -10,  # 碰撞奖励
        "reward_speed_range": [10, 50],  # [m/s] 从这个范围到最高速度的奖励呈线性关系 [0, HighwayEnv.HIGH_SPEED_REWARD].
        "simulation_frequency": 15,  # [Hz] 模拟频率
        "policy_frequency": 2,  # [Hz]  策略频率
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",   # 其他车辆类型
        "screen_width": 600,  # [px]  # 显示宽度
        "screen_height": 150,  # [px] # 显示高度
        "centering_position": [0.3, 0.5],   # 居中位置
        "scaling": 5.5,   # 缩放
        "show_trajectories": False,   # 显示轨迹
        "render_agent": True,   # 渲染智能体
        "offscreen_rendering": False    # 屏幕外渲染
    })

def normalize_obs(obs):
    # obs: (10, 7) or (70,)
    obs = obs.reshape(-1, 7)
    obs[:, 1] /= 100.0  # x
    obs[:, 2] /= 100.0  # y
    obs[:, 3] /= 20.0   # vx
    obs[:, 4] /= 20.0   # vy
    # presence, cos_h, sin_h 已经归一化了
    return obs.flatten()

episode_rewards = []
rollout_steps = 20
# 使用策略梯度法进行训练
agent = AgentHighWay(state_dim=70,value_output_dim=1,policy_lr=0.0001,value_lr=0.001)
for episode in range(10000):
    obs, _ = env.reset()
    state = normalize_obs(obs)
    done = False
    total_reward = 0
    step_count = 0
    actions_taken = []
    while not done:  # 加个上限防死循环
        env.render()
        action, log_prob, value = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        # 新增：惩罚倒车或极低速
        if info['speed'] < 10.0:  # 低于 10 m/s (~18 km/h) 就惩罚
            reward -= 10.0
        if info['speed'] < 0:  # 倒车重罚
            reward -= 5.0
        done = terminated or truncated

        agent.store_transition(state, action, reward, log_prob, value, done)
        state = normalize_obs(next_state)
        actions_taken.append(action)
        if len(agent.memory) == rollout_steps or done:
            agent.update_policy()
            agent.update_value()
            agent.clean_mem()

        total_reward += reward
        step_count += 1
    # 打印终止原因
    if done:
        print(f"Episode ended at step {step_count}:")
        print(f"  terminated={terminated}, truncated={truncated}")
        print(f"  info={info}")  # highway-env 会在这里返回原因！
        print("Actions:", actions_taken[:20])  # 看前20步

    # 打印进度
    episode_rewards.append(total_reward)
    if episode % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"回合 {episode}, 平均奖励: {avg_reward:.2f}")

env.close()
# 存储训练参数

print('开出存储训练参数')
torch.save(agent.policy_model.state_dict(),'./data/highway-policy.pth')
torch.save(agent.value_model.state_dict(),'./data/highway-value.pth')