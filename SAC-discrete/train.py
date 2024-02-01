from agent import SAC
from buffer import ReplayBuffer
from tool import collect_random
import torch
import gymnasium as gym
import highway_env

highway_env.register_highway_envs()

import matplotlib.pyplot as plt

import time
import math

if __name__ == "__main__":
    env = gym.make("intersection-v1", render_mode='rgb_array')
    # 根据一共10个车,7个参数 所以表格是10*7的
    env.configure({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 6,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,  # 全是相对我车来说的
            "order": "sorted"
        }
    })
    vehicles_count = env.config["observation"]["vehicles_count"]
    # env.config["collision_reward"]默认为-1

    env.reset()
    state_size = env.observation_space.shape[1] * env.observation_space.shape[0]
    action_size = env.action_space.shape[0] # (-1 , 1 , 2)

    # 参数设置
    buffer_size = 50000
    batch_size = 256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = SAC(state_size, action_size, device)
    buffer = ReplayBuffer(buffer_size, batch_size, device)
    collect_random(env, buffer, 256)  # 先存满buffer

    episode = 20
    for i in range(episode):
        state = env.reset()
        state = state[0]  # 类型numpy.ndarray
        state = torch.from_numpy(state).float()
        state = state.view(1, 42)
        step_count = 0
        reward_count = 0.0
        speed_cnt = 0
        while True:
            # -----------------选取action------------------------
            action = agent.get_action(state)
            print(action)
            next_state, reward, done, truncated, info = env.step(action)
            step_count = step_count + 1
            # -----------------学习------------------------
            next_state = next_state.reshape(1, 42)
            buffer.add(state, action, reward, next_state, done)
            actor_loss, alpha_loss, critic1_loss, critic2_loss, current_alpha = agent.learn(buffer.sample(), gamma=0.99,
                                                                                            batch_size=256)
            state = next_state
            state = torch.from_numpy(state).float()
            state = state.view(1, 42)

            if done:
                break
            env.render()

