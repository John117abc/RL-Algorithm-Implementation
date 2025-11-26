import numpy as np
import time
from rl_utils.GNGridWorldEnv import GridWorldEnv

# ε-Greedy策略生成
def epsilon_greedy_action(q_table, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(0, n_actions)
    else:
        return np.argmax(q_table[state])

# 设置环境
grid_world_size = 10
obstacle_count = 20
env = GridWorldEnv(size=grid_world_size,obstacle_count=obstacle_count)
n_actions = env.action_space.n
q_table = np.zeros((grid_world_size * grid_world_size, n_actions))

env.render()  # 显示窗口


# 初始参数
gamma = 0.95
epsilon = 1.0   # 随机epsilon，强探索
epsilon_decay = 0.9995   # eps的每轮衰减率
epsilon_min = 0.01 # 最小的eps
max_iterate = 10000
seed = 20
learning_rate = 0.01

state_count = np.zeros((grid_world_size * grid_world_size, n_actions))  # 访问次数
for it_index in range(max_iterate):

    observation, _ = env.reset(seed=seed, options={'enable_random_pos': True})
    state = int((observation['agent'][0] * grid_world_size) + observation['agent'][1])
    done = False
    episode_max = 200
    episode = []
    while not done and episode_max >0:
        action = epsilon_greedy_action(q_table, state, epsilon, n_actions)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = int((next_state['agent'][0] * grid_world_size) + next_state['agent'][1])
        episode.append((state, action, reward))
        # q-learning的核心步骤
        q_table[state][action] += learning_rate * (reward + gamma * q_table[next_state][np.argmax(q_table[next_state])] - q_table[state][action])
        state = next_state
        done = terminated or truncated
        episode_max -= 1

    # 更新epsilon
    epsilon = max(epsilon * epsilon_decay,epsilon_min)
    # 打印进度
    if it_index % 100 == 0:
        print(f"Episode {it_index}, epsilon: {epsilon:.4f}")


print("\n正在用新策略运行一个可视化 episode...")
test_env = GridWorldEnv(size=grid_world_size, render_mode='human',obstacle_count=obstacle_count)
n_actions = test_env.action_space.n
observation, _ = test_env.reset(seed=seed, options={'enable_random_pos': True})
state = int((observation['agent'][0] * grid_world_size) + observation['agent'][1])
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state])
    next_state, reward, terminated, truncated, _ = test_env.step(action)
    state = int((next_state['agent'][0] * grid_world_size) + next_state['agent'][1])
    total_reward += reward
    done = terminated or truncated
    time.sleep(0.5)

print(f"测试 episode 总奖励: {total_reward}")
test_env.close()
env.close()