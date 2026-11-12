"""Evaluate a saved actor without noise, and/or record one episode to MP4.

  python3 record_episode.py eval  [CKPT_DIR] [N]        # mean score over N seeds
  python3 record_episode.py video [CKPT_DIR] [SEED] [OUT.mp4]
"""
import sys

import gymnasium as gym
import imageio
import numpy as np
import torch as T

from ddpg_torch import ActorNetwork

T.set_num_threads(1)
mode = sys.argv[1]
ckpt_dir = sys.argv[2] if len(sys.argv) > 2 else 'tmp/ddpg'

actor = ActorNetwork(0.000025, [8], 400, 300, n_actions=2, name='Actor', chkpt_dir=ckpt_dir)
actor.load_state_dict(T.load(actor.checkpoint_file, map_location='cpu'))
actor.eval()


def run(env, seed, frames=None):
    obs, _ = env.reset(seed=seed)
    done, score = False, 0.0
    while not done:
        if frames is not None:
            frames.append(env.render())
        with T.no_grad():
            a = actor(T.tensor(obs, dtype=T.float)).numpy()
        obs, r, term, trunc, _ = env.step(a)
        score += r
        done = term or trunc
    return score


if mode == 'eval':
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    env = gym.make('LunarLander-v3', continuous=True)
    scores = [run(env, s) for s in range(n)]
    print(ckpt_dir, 'mean %.1f' % np.mean(scores), 'landed(>200) %d/%d' % (sum(x > 200 for x in scores), n),
          ' '.join('%.0f' % x for x in scores))
else:
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    out = sys.argv[4] if len(sys.argv) > 4 else 'episode.mp4'
    env = gym.make('LunarLander-v3', continuous=True, render_mode='rgb_array')
    frames = []
    score = run(env, seed, frames)
    imageio.mimsave(out, frames, fps=50, macro_block_size=1)
    print(f'seed {seed}: score {score:.1f}, {len(frames)} frames -> {out}')
