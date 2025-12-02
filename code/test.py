import time
import torch
import random
from torch import nn
from rl_utils.GNGridWorldEnv import GridWorldEnv

# 定义神经网络
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# 设置环境
grid_world_size = 10
obstacle_count = 20
test_env = GridWorldEnv(size=grid_world_size, render_mode='human', obstacle_count=obstacle_count)
n_actions = test_env.action_space.n

epsilon_min = 0.01  # 最小探索率

# 测试学习到的策略
print("\n正在用学习到的策略运行测试 episode...")
observation, _ = test_env.reset(seed=99,options={'enable_random_pos': True})
state = observation['agent'] / grid_world_size
done = False
total_reward = 0
# 设备选择
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else "cpu"
)
print(f"使用设备: {device}")
# ε-Greedy动作选择
def select_action_test(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, n_actions-1)

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = predict_network(state_tensor)
    return torch.argmax(q_values).item()

# 加载模型参数
model_path = '../data/deep-q-learning/dqn-model.pth'
print(f"正在从 {model_path} 加载模型...")
params = torch.load(model_path, map_location=device)

predict_network = NeuralNetwork(input_dim=2,hidden_dim=1024, output_dim=4).to(device)
predict_network.load_state_dict(params)
predict_network.eval()  # 设置为评估模式

while not done:
    action = select_action_test(state, epsilon_min)  # 使用最小探索率
    next_observation, reward, terminated, truncated, _ = test_env.step(action)
    state = next_observation['agent'] / grid_world_size
    total_reward += reward
    done = terminated or truncated
    time.sleep(0.3)

print(f"测试 episode 总奖励: {total_reward:.2f}")
test_env.close()
