####################################
#            IMPORT
####################################
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
import logging
logging.basicConfig(level=logging.WARNING)


####################################
#            ENV
####################################
DOWN= 0
UP= 1
RIGHT = 2
LEFT= 3
PICK= 4
DROP_OFF= 5

# env = gym.make('Taxi-v3', render_mode="human")
env = gym.make('Taxi-v3')

####################################
#            FUNCTION
####################################
def update_epsilon(epsilon, episode):
    epsilon = 1
    max_eps = 1
    min_eps = 0.05
    eps_decay = 0.003/3

    if(epsilon > min_eps):
        epsilon =(max_eps - min_eps) * np.exp(-eps_decay*episode) + min_eps

    return epsilon

def update_reward(qtable, state, action, new_state, alpha, gamma):
    old_reward = qtable[state][action]
    # max[a](Q[s+1,a]) --> B
    next_better_value = max(qtable[new_state])
    # gamma * B --> C
    new_reward = next_better_value * gamma
    # reward + C - A --> D
    new_reward = new_reward + res[1] - old_reward
    # alpha * D --> E
    new_reward = new_reward * alpha
    # A + E --> Q[s,a]
    new_reward = new_reward + old_reward

    # if(action in [0,1,2,3]):
    #     if(state == new_state):
    #         new_reward = new_reward - 2

    logging.info(f"old_reward: {old_reward}")
    logging.info(f"new_reward: {new_reward}\n")

    return new_reward

def show(list_result):
    arrayX = []
    arrayY = list_result
    nb_total = len(list_result)
    for i in range(nb_total):
        arrayX.append(i)

    print(max(list_result))
    
    plt.plot(arrayX, arrayY)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("")

    plt.show()

    
####################################
#         Init variables
####################################
state_size = env.observation_space.n
action_size = env.action_space.n

# Init the qTable
qtable = [[0 for _ in range(action_size)] for _ in range(state_size)]

# Init parameters
max_episode = 4000
alpha = 0.5
gamma = 0.9
max_step = 150

epsilon = 1
decay_rate = 0.005

list_reward = []

## Train ##
for e in range(max_episode):
    # restart the env
    env.reset() 
    # init the first state
    state = 0
    done = False 

    # reward buffer for each episode
    reward = 0
    for _ in range(max_step):

        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            better_value = max(qtable[state])
            action =  qtable[state].index(better_value)

        res = env.step(action)

        # q_learning update formule:
        # Q[s,a] = Q[s,a] + alpha*(reward + gamma*max[a](Q[s+1,a]) - Q[s,a])
        new_reward = update_reward(qtable, state, action, res[0], alpha, gamma)
 
        qtable[state][action] = new_reward

        reward += res[1]

        state = res[0]
        done = res[2]
        if(done):
            break

    epsilon = update_epsilon(epsilon, e)
    list_reward.append(reward)
    print(f"reward {e}: {reward}")
       
show(list_reward)
# env = gym.make('Taxi-v3', render_mode="human")
# for _ in range(5):
#     res = env.reset()
#     done = False
#     while(not done):
#         state = res[0]

#         better_value = max(qtable[state])
#         action =  qtable[state].index(better_value)
#         print(action)

#         res = env.step(action)
#         env.render()

#         state = res[0]
#         done = res[2]
