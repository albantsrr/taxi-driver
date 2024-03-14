import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

def show(x, y, titre="Victoire/Temp"):
    plt.plot(x, y)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(titre)

    plt.show()

## GLOBAL ##
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

## Create env ##
environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
# environment = gym.make("FrozenLake-v1", is_slippery=False)
environment.reset()

## Create matrix [state*action] ##
nb_states = environment.observation_space.n  # = 16
nb_actions = environment.action_space.n      # = 4

# print(f"Observation space: {nb_states}")
# print(f"Action space: {nb_actions}")

qtable = [[0 for _ in range(nb_actions)] for _ in range(nb_states)]

# qtable =[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.3375, 0.0, 0.0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0.0, 0.225, 0.0], [0.0, 0, 0.875, 0], [0, 0, 0, 0]]

## Variables ##
number = 200
alpha = 0.5
gamma = 0.9

list_result = []

## Train ##
# print (qtable) 
list_saved_path = []
for _ in range(number):
    environment.reset() 
    state = 0
    done = False 

    if(max(qtable[state]) > 0):
        better_value = max(qtable[state])
        action =  qtable[state].index(better_value)
    else:
        action = environment.action_space.sample()

    path = []
    while(not done):
        # First action
        res = environment.step(action)

        # Second action
        if(max(qtable[res[0]]) > 0):
            next_better_value = max(qtable[res[0]])
            next_action =  qtable[res[0]].index(next_better_value)
        else:
            next_action = environment.action_space.sample() 

        next_value = qtable[res[0]][next_action]
        old_value = qtable[state][action]
        new_value = next_value * gamma
        new_value = new_value + res[1] - old_value
        new_value = new_value * alpha
        new_value = new_value + old_value

        qtable[state][action] = new_value - 0.0001

        # if(res[1] == 1):
        #     print(qtable, "\n")
        path.append(action)
        action = next_action
        state = res[0]
        done = res[2]


    if(res[1] == 1):
        list_result.append(1)
        list_saved_path.append(path)
    else:
        list_result.append(0)
    
arrayX = []
arrayY = []
nb_total = len(list_result)
for i in range(nb_total):
    result = list_result[:i].count(1)
    # value = result/(i+1)
    arrayY.append(result) 
    arrayX.append(i)

show(arrayX, arrayY)

number_good_path = len(list_saved_path)
# print(qtable)
print(list_saved_path[number_good_path - 3:])
