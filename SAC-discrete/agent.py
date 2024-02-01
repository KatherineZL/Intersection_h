import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from network import Critic, Actor
import copy
import numpy as np
# loss公式 ; 参数 按照sac包

class SAC(nn.Module):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 device
                 ):
        super(SAC, self).__init__()

        # 参数
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = 0.99  # γ 折扣因子
        self.tau = 1e-2  # τ
        hidden_size = 256
        learning_rate = 5e-4  # 学习率 0.0005
        self.clip_grad_param = 1  # 防止梯度爆炸

        # α
        self.target_entropy = -action_size  # -dim(A)
        # self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98

        self.log_alpha = torch.tensor([0.0], requires_grad=True)  # log(α) 的初始值为 0，并且需要计算梯度
        self.alpha = self.log_alpha.exp().detach()  # 显然α = exp(log α) 不需要梯度
        # self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=0.01) # 0.002学习率太小

        # Actor Network

        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        # self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_size).to(device)

        # 要求 这两个网参数不一样
        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())  # 将critic1中的参数全都复制到critic1_target

        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())  # 将critic2 复制到critic2_target

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

    # s → (get_det_action) → action 一个数字----与官方对比过 +z版
    def get_action(self, state):
        # state = torch.from_numpy(state).float().to(self.device)  # 转化成float型张量
        with torch.no_grad():
            # actor_local = Actor类的实例
            # action = self.actor_local.get_det_action(state)
            # 试试这样呢?
            action,_,_ = self.actor_local.get_action(state)
        return action.numpy()

    # 计算动作policy的loss----与官方对比过
    def calc_policy_loss(self, states, alpha):
        # # evaluate(s) → 动作；动作概率(256x6车x5)；log 动作概率(256x6车x5)
        _, action_probs, log_pis = self.actor_local.evaluate(states)  # Actor类
        q1 = self.critic1(states)  # 256x5
        q2 = self.critic2(states)  # 256x5
        # shape
        min_Q = torch.min(q1, q2)  # 选取最小的q
        # 对应 Jπ(Φ)=E(...) 式子
        actor_loss = (action_probs * (alpha * log_pis - min_Q)).sum(1).mean()

        # 公式
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi, action_probs

    def learn(self, experiences, gamma, batch_size=256):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)   -----正常loss的计算
        Actor_loss = α * log_pi(a|s) - Q(s,a) ---- 公式
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences



        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)

            # -------计算E(V(S_))------------
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            # 这里计算的是 V(s_); Q_target_next为256x1x5  --乘概率了 是平均值
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)
            # 以下两句
            Q_target_next=torch.squeeze(Q_target_next,dim=1)  # E(V(S_)); Q_target_next为256x1
            Q_target_next=Q_target_next.sum(dim=1)
            Q_target_next=Q_target_next.unsqueeze(-1)

            Q_targets = rewards + ((gamma * (1 - dones)) * Q_target_next) # 256x1

            # Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1))

        # -------计算q1 q2------------

        # gather 取出 每个s-所选动作的 q值
        # q1→256x1x5
        # actions = torch.unsqueeze(actions, dim=2)
        value1=self.critic1(states).squeeze(dim=1)  # 256*1*5
        value2=self.critic2(states).squeeze(dim=1)  # 256*1*5

        q1 = value1.gather(1, actions.long())
        q2 = value2.gather(1, actions.long())

        # -------计算JQ(θ)------------
        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ---------------------------- update actor 策略 ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)  # 复制α
        # 得到
        actor_loss, log_pis, action_probs = self.calc_policy_loss(states, current_alpha.to(self.device))
        # 在每次-训练-之前将梯度清零，以确保每次-反向传播-都是从零开始的。
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # .step()根据当前的梯度信息,更新entropy_alpha的值(actor_optimizer定义时params=[self.log_alpha]),以最小化损失函数
        self.actor_optimizer.step()

        # -------------------Compute alpha loss---------------------
        # 用公式J(a) 但是是连续版本
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), current_alpha

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
