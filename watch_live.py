"""Live simulation of the agent while it trains.

Before every episode, reloads tmp/ddpg/Actor_ddpg (saved by main_torch.py every
25 episodes) and flies it WITHOUT exploration noise, so you can watch the policy
improve during training.
"""
import os

import gymnasium as gym
import torch as T

from ddpg_torch import ActorNetwork

# A single-sample forward pass gains nothing from many threads; they only spin
# and steal CPU from the training process.
T.set_num_threads(1)

env = gym.make('LunarLander-v3', continuous=True, render_mode='human')
actor = ActorNetwork(0.000025, [8], 400, 300, n_actions=2, name='Actor')
ckpt = actor.checkpoint_file
last_mtime, ep = None, 0

while True:
    if os.path.exists(ckpt) and os.path.getmtime(ckpt) != last_mtime:
        try:
            actor.load_state_dict(T.load(ckpt, map_location='cpu'))
            last_mtime = os.path.getmtime(ckpt)
            print('>> new checkpoint loaded', flush=True)
        except Exception:
            pass  # file mid-write; retry next episode
    actor.eval()
    obs, _ = env.reset()
    done, score = False, 0.0
    while not done:
        with T.no_grad():
            a = actor(T.tensor(obs, dtype=T.float)).numpy()
        obs, r, term, trunc, _ = env.step(a)
        score += r
        done = term or trunc
    ep += 1
    print(f'watch episode {ep}: score {score:.1f}', flush=True)
