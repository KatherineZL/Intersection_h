import torch


# 这个函数是为了  train前先随机抽样，充满buffer
def collect_random(env, dataset, num_samples=256):
    state = env.reset()
    state = state[0]
    state = state.reshape(1, 42)
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
            state = state[0]
            state = state.reshape(1, 42)

# def save(args, save_name, model, wandb, ep=None):
#     import os
#     save_dir = './trained_models/'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     if not ep == None:
#         torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
#         wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
#     else:
#         torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
#         wandb.save(save_dir + args.run_name + save_name + ".pth")
