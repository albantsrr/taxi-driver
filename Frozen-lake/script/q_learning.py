####################################
#            IMPORT
####################################
import gymnasium as gym
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.WARNING)

####################################
#           FUNCTION
####################################
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

    logging.info(f"old_reward: {old_reward}")
    logging.info(f"new_reward: {new_reward}\n")

    return new_reward

def show(list_result):
    arrayX = []
    arrayY = []
    nb_total = len(list_result)
    for i in range(nb_total):
        result = list_result[:i].count(1)
        # value = result/(i+1)
        arrayY.append(result) 
        arrayX.append(i)

    plt.plot(arrayX, arrayY)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("")

    plt.show()

####################################
#                ENV
####################################
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

custom_map = [  'FFHHFFFFFFFF',
                'HFFFHHFFFFFF',
                'HFHFFHFFFFFF',
                'FFFFHFFFFFFF',
                'FFFFFFHFFFFF',
                'HFFFFFFFFFFF',
                'FFGFHFFFFFFF']

# create the environment
# environment = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="human")
environment = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False)

# create matrix [state*action]
state_size = environment.observation_space.n 
action_size = environment.action_space.n  

logging.info(f"State size: {state_size}")
logging.info(f"Action size: {action_size}\n")

# Init the qTable
qtable = [[0 for _ in range(action_size)] for _ in range(state_size)]
# qtable =[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.3375, 0.0, 0.0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0.0, 0.225, 0.0], [0.0, 0, 0.875, 0], [0, 0, 0, 0]]


# Init parameters
episode = 3000
alpha = 0.5
gamma = 0.9

logging.info(f"number episode: {episode}")
logging.info(f"value alpha: {alpha}")
logging.info(f"value gamma: {gamma}\n")

list_result = []
list_saved_path = []

## Train ##
for _ in range(episode):
    # restart the env
    environment.reset() 
    # init the first state
    state = 0
    done = False 

    # path buffer for each episode
    path = []
    while(not done):        
        if(max(qtable[state]) > 0):
            better_value = max(qtable[state])
            action =  qtable[state].index(better_value)
        else:
            action = environment.action_space.sample()

        res = environment.step(action)

        # q_learning update formule:
        # Q[s,a] = Q[s,a] + alpha*(reward + gamma*max[a](Q[s+1,a]) - Q[s,a])
        new_reward = update_reward(qtable, state, action, res[0], alpha, gamma)
 
        qtable[state][action] = new_reward - 0.0001

        path.append(action)

        state = res[0]
        done = res[2]

    if(res[1] == 1):
        list_saved_path.append(path)
        list_result.append(1)
    else:
        list_result.append(0)
    





show(list_result)

number_good_path = len(list_saved_path)
# print(qtable)
print(list_saved_path[number_good_path - 3:])
