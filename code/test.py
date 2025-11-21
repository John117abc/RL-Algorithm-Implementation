import gymnasium as gym

# 创建环境
env = gym.make("FrozenLake-v1", render_mode="human")
# 其余代码不变，运行时会弹出窗口

# 重置
state, info = env.reset()
print(f"起始状态: {state}")

done = False
total_reward = 0
step = 0

while not done:
    action = env.action_space.sample()  # 随机动作
    next_state, reward, terminated, truncated, info = env.step(action)

    print(f"Step {step}: state={state} → action={action} → next={next_state}, reward={reward}")

    total_reward += reward
    done = terminated or truncated
    state = next_state
    step += 1

print(f"Episode 结束，总奖励: {total_reward}")
env.close()