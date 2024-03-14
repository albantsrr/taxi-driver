import logging
logging.basicConfig(level=logging.ERROR)

from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from collections import deque

import random
import numpy as np
import gymnasium as gym

####################################
#           ENV
####################################

# env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
env = gym.make("FrozenLake-v1", is_slippery=False)
train_episodes=4000
# train_episodes=100
test_episodes=100
max_steps=300
state_size = env.observation_space.n
action_size = env.action_space.n
batch_size=32

logging.warning(f"state size: {state_size}")
logging.warning(f"action size: {action_size}")

####################################
#           MODEL
####################################
class cAgent:
    def __init__(self, state_size, action_size):
        self.learning_rate = 0.0001
        self.memory = deque(maxlen=2500)
        self.learning_rate=0.001
        self.epsilon=1
        self.max_eps=1
        self.min_eps=0.01
        # self.eps_decay = 0.001/3
        self.eps_decay = 0.1/3
        self.gamma=0.9
        self.state_size= state_size
        self.action_size= action_size
        self.epsilon_lst=[]


        self.model = self.__build_model()

    def __build_model(self):
        model=Sequential()
        model.add(Dense(10, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model


    def add_memory(self, data):
        logging.warning(f"data: {data}")
        logging.warning(f"new_state, reward, done, state, index_best_action\n")

        self.memory.append(data)

    def replay(self,batch_size):
        # get random batch from memory 
        list_batch=random.sample(self.memory, batch_size)
        for i in range(len(list_batch)):
            # get data info from the batch list 
            data = list_batch[i]
            new_state = data[0]
            reward = data[1]
            done = data[2]
            state = data[3]
            index_best_action = data[4]

            # keep the old value
            new_reward = reward
            if(done == True):
                # Apply the q learning formule
                # get value of the next best move 
                # max[a](Q[s+1,a]) --> A
                next_best_value = np.amax(self.model.predict(new_state))
                # gamma * A --> B
                new_reward = next_best_value * self.gamma
                # r + B --> C
                new_reward = reward + new_reward 

            # get the vector of value for actions
            array_action_value = self.model.predict(state)
            # update the value of the next best move 
            array_action_value[0][index_best_action] = new_reward

            # adjust the weight to match with the array_action_value updated
            self.model.fit(state, array_action_value, epochs=1, verbose=0)

        if self.epsilon > self.min_eps:
            self.epsilon=(self.max_eps - self.min_eps) * np.exp(-self.eps_decay*episode) + self.min_eps

        self.epsilon_lst.append(self.epsilon)



    def step(self, state):
        # greedy epsilon >> to get some exploration 
        random_value = np.random.rand()
        if(random_value < self.epsilon):
            random_action = np.random.randint(0,self.action_size)
            logging.warning(f"GREEDY EPSILON : {random_action}\n")
            return random_action
        else:
            return self.__predict(state)
            
    def __predict(self, state):
        array_action_value = self.model.predict(state)
        idx_best_action = np.argmax(array_action_value)

        logging.warning(f"state : {state}")
        logging.error(f"array_action_value : {array_action_value}")
        logging.warning(f"idx_best_action : {idx_best_action}\n")
        return idx_best_action


c_agent = cAgent(state_size, action_size)


####################################
#           CODE
####################################
for episode in range(train_episodes):
    # get state from the start
    state= env.reset()
    state = state[0]

    # create the vector for input layer
    array_state=np.zeros(state_size)
    # set the state of the agent
    array_state[state] = 1
    # reshape vector for inut layer
    state= np.reshape(array_state, [1, state_size])

    reward = 0
    done = False
    for t in range(30):
        # env.render()
        # get the index of the next best action
        index_best_action = c_agent.step(state)
        
        # move in the env and collect info
        res = env.step(index_best_action)
        new_state = res[0]
        reward = res[1]
        done = res[2]

        # build new vector for input layer
        new_array_state=np.zeros(state_size)
        new_array_state[new_state] = 1
        new_state= np.reshape(new_array_state, [1, state_size])

        # store data in a temp memory
        data = [new_state, reward, done, state, index_best_action]
        c_agent.add_memory(data)
        state = new_state

        if(done == True):
            print(f'Episode: {episode:4}/{train_episodes} and step: {t:4}. Eps: {float(c_agent.epsilon):.2}, reward {reward}')
            break

    if (len(c_agent.memory)> batch_size):
        c_agent.replay(batch_size)
        
    
