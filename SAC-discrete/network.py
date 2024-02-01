import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=32):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = F.relu(self.fc1(state))  # 256x6车x256
        x = F.relu(self.fc2(x))  # 256x6车x256
        action_probs = self.softmax(self.fc3(x))  # 256x6车x5--(5个动作)
        return action_probs

    # s→ evaluate → 动作；动作概率；log 动作概率
    def evaluate(self, state, epsilon=1e-6):
        # action_probs 是一个表示动作概率分布的张量，它的形状为 (batch_size, action_size)
        action_probs = self.forward(state)  # s→ Actor类 forward→ 动作概率
        dist = Categorical(action_probs)
        # action形状(batch_size)  dist.sample() 方法从动作概率分布中采样，表示选取的动作在动作空间中的索引。
        action = dist.sample().to(state.device)
        z = action_probs == 0.0  # 若action_probs==0-true=1  不等于0-false=0  即为0时z是1，防止出现log 0 的情况
        z = z.float() * 1e-8
        # log_action_probabilities = action_probs加入noise z ;防止log 0
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities  # 返回动作;动作概率;动作log概率 -- 这三个都是数组！

# 似乎没用
    def get_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities

    # s → 计算概率分布 → action(1个数字 张量)

    def get_det_action(self, state):
        action_probs = self.forward(state)  # Actor-forward (s→ →动作概率分布)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)  # 按照概率选
        return action.detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32):
        super(Critic, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """ 每个(state, action)对 -> Q-values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
