import numpy as np
import matplotlib.pyplot as plt

epsilon = 1
max_eps = 1
min_eps = 0.01
eps_decay = 0.003/3
episode = 4000

epsilon_values = [(max_eps - min_eps) * np.exp(-eps_decay * ep) + min_eps for ep in range(episode)]

plt.plot(range(episode), epsilon_values)
plt.xlabel('episode')
plt.ylabel('epsilon')
plt.show()