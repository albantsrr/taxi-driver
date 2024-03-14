
epsilon = 1
decay_rate = 0.005
start_episode = 0

learning_rate = 0.9
discount_rate = 0.8


num_episodes = 200
num_steps = 99


for episode in range(start_episode, num_episodes):
    state, _ = env.reset()
    total_rewards = 0
    steps = 0
    done = False

    for step in range(num_steps):
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state,:])

        new_state, reward, done, info = env.step(action)[:4]

        total_rewards += reward
        steps+=1

        qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:]) - qtable[state,action])

        state = new_state

        if done:
            break

    epsilon = np.exp(-decay_rate * episode)

