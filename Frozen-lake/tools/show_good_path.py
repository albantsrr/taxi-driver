import gymnasium as gym

list_path = [[2, 1, 1, 1, 1, 2, 1, 1], [2, 1, 1, 1, 1, 2, 1, 1], [2, 1, 1, 1, 1, 2, 1, 1]]

custom_map = [  'FFHHFFFFFFFF',
                'HFFFHHFFFFFF',
                'HFHFFHFFFFFF',
                'FFFFHFFFFFFF',
                'FFFFFFHFFFFF',
                'HFFFFFFFFFFF',
                'FFGFHFFFFFFF']

## Create env ##
environment = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="human")
# environment = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
for i in range(len(list_path)):
    good_path = list_path[i]
    environment.reset()
    for j in range(len(good_path)):
        action = good_path[j]
        environment.step(action)