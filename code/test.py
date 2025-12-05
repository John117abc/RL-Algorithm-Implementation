import numpy as np
import torch
from torch import nn
from collections import deque
from rl_utils.GNGridWorldEnv import GridWorldEnv

# 设备选择
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else "cpu"
)
print(f"使用设备: {device}")


# 定义神经网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim,hidden_dim ,action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出动作概率

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # 输出概率分布


class PolicyAgent:
    def __init__(self,state_dim,hidden_dim,action_dim,lr = 0.01):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.lr = lr

        # 初始化神经网络
        self.model = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = lr)

        self.gamma = 0.99
        self.memory = deque()

    # 使用神经网络输出概率分布，再使用贪心策略选择action
    def select_action(self,state):
        state = torch.tensor(state,dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            predict = self.model(state)
        action = np.random.choice(self.action_dim,p=predict.squeeze().numpy())
        return action

    # 存储每一次action后得state,action,reward
    def save_memory(self,state,action,reward):
        self.memory.append((state,action,reward))

    def calculate_returns(self):
        returns = []
        G = 0.0
        # 从后往前遍历
        for state, action, reward in reversed(self.memory):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    # 更新策略
    def update_policy(self):
        if len(self.memory) == 0:
            return  # 防止空经验更新
        states, actions, rewards = zip(*self.memory)
        returns = self.calculate_returns()

        states = torch.tensor(states,dtype=torch.float32)
        actions = torch.tensor(actions,dtype=torch.long)
        returns = torch.tensor(returns,dtype=torch.float32)

        # 取每一次被选择的那个动作的概率
        choose = self.model(states)[torch.arange(len(states)), actions]
        log = torch.log(choose + 1e-8)

        # 定义损失函数，因为要做梯度上升，所以加上负号
        loss_func = -(log * returns).mean()

        self.optimizer.zero_grad()
        loss_func.backward()
        self.optimizer.step()

# 设置环境
grid_world_size = 10
obstacle_count = 20
env = GridWorldEnv(size=grid_world_size, obstacle_count=obstacle_count)
n_actions = env.action_space.n

# 训练循环
global_step = 0
episode_rewards = []


# 使用策略梯度法进行训练
agent = PolicyAgent(state_dim=2,hidden_dim=64,action_dim=4,lr = 0.0001)
for episode in range(10000):
    # 重置环境
    observation, _ = env.reset(seed=99,options={'enable_random_pos': True})
    state = observation['agent'] / (grid_world_size - 1 + 1e-8)  # 归一化到[0,1]
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < 1000:  # 最大步数限制
        # 选择动作
        action = agent.select_action(state)

        # 执行动作
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = next_observation['agent'] / (grid_world_size - 1 + 1e-8)
        done = terminated or truncated

        # 存储经验
        agent.save_memory(state,action,reward)
        # 更新状态
        state = next_state
        total_reward += reward
        global_step += 1
        step_count += 1

    agent.update_policy()
    agent.memory.clear()
    # 更新探索率
    episode_rewards.append(total_reward)

    # 打印进度
    if episode % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"回合 {episode}, 平均奖励: {avg_reward:.2f}, 步数: {global_step}")

env.close()

# 保存模型
torch.save(agent.model.state_dict(), '../data/value-function-method/vf-model.pth')