"""Resume training from the last checkpoint (same agent/hyperparameters as main_torch.py).

  python3 -u resume_torch.py START_EPISODE [END_EPISODE] >> train.log

Loads tmp/ddpg/* (saved by main_torch.py every 25 episodes) and continues from
START_EPISODE up to END_EPISODE (default 1000). The 100-episode average is
seeded from the existing train.log so it continues smoothly.

Limitation: the replay buffer and OU-noise state are NOT saved by the original
code, so learning restarts with an empty buffer (it refills within a few episodes).
"""
import os
import re
import sys

import gymnasium as gym
import numpy as np

from ddpg_torch import Agent
from utils import plotLearning

start = int(sys.argv[1])
end = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

env = gym.make('LunarLander-v3', continuous=True)
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)
agent.load_models()

# previous scores for episodes < start, so the trailing average is continuous
score_history = []
pat = re.compile(r'^episode\s+(\d+)\s+score\s+(-?[\d.]+)')
if os.path.exists('train.log'):
    for line in open('train.log'):
        m = pat.match(line)
        if m and int(m[1]) < start:
            score_history.append(float(m[2]))

for i in range(start, end):
    obs, _ = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
    score_history.append(score)

    if i % 25 == 0:
        agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]), flush=True)

agent.save_models()  # always keep the final model
plotLearning(score_history, 'LunarLander-alpha000025-beta00025-400-300.png', window=100)
