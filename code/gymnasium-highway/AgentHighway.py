import torch
import numpy as np
from torch import nn
# 设备选择
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else "cpu"
)
print(f"使用设备: {device}")

# 定义神经网络
class PolicyNetwork(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim = 5):
        super(PolicyNetwork,self).__init__()
        self.an1 = nn.Linear(state_dim,hidden_dim)
        self.an2 = nn.Linear(hidden_dim,hidden_dim)
        self.an3 = nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x = torch.relu(self.an1(x))
        x = torch.relu(self.an2(x))
        return self.an3(x)

class ValueNetwork(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim = 1):
        super(ValueNetwork,self).__init__()
        self.an1 = nn.Linear(input_dim,hidden_dim)
        self.an2 = nn.Linear(hidden_dim,hidden_dim)
        self.an3 = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x = torch.relu(self.an1(x))
        x = torch.relu(self.an2(x))
        return self.an3(x)

class AgentHighWay:
    def __init__(self,value_output_dim,state_dim,
                 hidden_dim = 256,action_dim = 5,policy_lr = 0.01,value_lr = 0.01):
        self.advantage = None
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.policy_lr = policy_lr

        self.value_input_dim = state_dim
        self.value_output_dim = value_output_dim
        self.value_lr = value_lr

        # 初始化神经网络
        self.policy_model = PolicyNetwork(self.state_dim,self.hidden_dim,self.action_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(),lr = self.policy_lr)

        self.value_model = ValueNetwork(self.value_input_dim,self.hidden_dim,self.value_output_dim).to(device)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(),lr = self.value_lr)

        self.gamma = 0.99
        self.memory = []

    # 存储记录
    def store_transition(self, state, action, reward, log_prob, value, done):
        self.memory.append((state,action,reward,log_prob,value,done))

    def calculate_returns(self):
        returns = []
        G = 0.0

        # 如果最后一步不是终止状态，用 critic 估计 V(last_state)
        if not self.memory[-1][5]:  # 第6个元素是 done
            last_state = torch.FloatTensor(self.memory[-1][0]).unsqueeze(0).to(device)
            with torch.no_grad():
                G = self.value_model(last_state).item()

        # 从后往前计算
        for i in reversed(range(len(self.memory))):
            reward = self.memory[i][2]
            G = reward + self.gamma * G
            returns.insert(0, G)

        return returns

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.policy_model(state)
            value = self.value_model(state)
        dist = torch.distributions.Categorical(logits=logits)   # 直接传 logits，如果 logits = [0.1, 0.7, 0.2]，那么 dist 就是一个“以10%概率选0、70%选1、20%选2”的分布。
        action = dist.sample()   # 从 dist 中随机采样一个动作。
        log_prob = dist.log_prob(action)    # 计算刚刚采样出的 action 在该分布下的对数概率
        return action.item(), log_prob.item(), value.item()

    # 使用A2C算法更新策略
    def update_policy(self):
        if not self.memory:
            return
        states,actions,rewards,log_probs,values,dones = zip(*self.memory)
        values = torch.tensor(values, dtype=torch.float32).to(device)

        states = torch.from_numpy(np.array(states)).to(device).float()
        actions = torch.tensor(actions,dtype=torch.long).to(device)
        # 重新计算当前策略下的 log_prob
        logits = self.policy_model(states)
        dist = torch.distributions.Categorical(logits=logits)
        current_log_probs = dist.log_prob(actions).to(device)

        # 计算优势函数
        returns = self.calculate_returns()
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = returns - values

        loss_func = -(current_log_probs * advantages.detach()).mean()

        self.policy_optimizer.zero_grad()
        loss_func.backward()
        self.policy_optimizer.step()

    def update_value(self):
        if not self.memory:
            return

        states,actions,rewards,log_probs,values,dones = zip(*self.memory)
        states = torch.from_numpy(np.array(states)).to(device).float()
        # 重新计算values
        current_values = self.value_model(states).squeeze(-1)

        returns = self.calculate_returns()  # 应该返回 list of floats
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        loss_func = nn.MSELoss()(current_values, returns)

        self.value_optimizer.zero_grad()
        loss_func.backward()
        self.value_optimizer.step()

    def clean_mem(self):
        self.memory.clear()