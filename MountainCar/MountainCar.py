import gym
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def posToQ(pos):
    pos += 1.2
    pos = int(pos * (states_n-1) / 1.8)
    return pos


def velToQ(vel):
    vel += 0.07
    vel = int(vel * (states_n-1) / 0.14)
    return vel


def plott(q):
    X = []
    Y = []
    Z = []
    for pos in range(states_n):
        for vel in range(states_n):
            X.append(pos)
            Y.append(vel)
            inputs = True
            for p in q[pos][vel]:
                if p != 0.0:
                    inputs = False  # means there are values
                    break
            if inputs:
                Z.append(3)
            else:
                Z.append(np.argmax(q[pos][vel]))
    Z = pd.Series(Z)
    colors = {0: 'red', 1: 'yellow', 2: 'lime', 3: 'black'}
    out = Z.map(lambda x: colors[x])
    labels = ['Left', 'Nothing', 'Right']

    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap


    fig = plt.figure(3, figsize=[7, 7])
    ax = fig.gca()
    plt.set_cmap('brg')
    surf = ax.scatter(X, Y, c=out)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Policy')
    fig.savefig('Policy.png')
    plt.show(block=False)


env = gym.make('MountainCar-v0')
# observation
# pos -1.2 .. 0.6
# vel -0.07 .. 0.07

# in how many sections the observation states will be sectionnned
states_n = 100

learn = False

q = 0
if learn:
    q = np.zeros((states_n, states_n, 3))
else:
    q = np.load('trained_Q_DONE.npy')
    states_n = len(q[0, :])
    plott(q)
    # trained file

next_state = [0, 0]
gamma = 0.95  # discount reward rate
alpha = 0.8  # learning rate
epsilon = 1
epsilon_decay = 0.99

# iterate through episodes
for episode in range(10000000):
    state = env.reset()

    # iterate through steps
    while True:
        if episode % 1000 == 0 or not learn :
            env.render()
        if learn:
            # best valued action for a specific position and a velocity
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
                epsilon *= epsilon_decay
            else:
                action = int(np.argmax(q[posToQ(state[0])][velToQ(state[1])]))
                input = True
                for i in q[posToQ(state[0])][velToQ(state[1])]:
                    if i != 0.0:
                        input = False
                        break

                if input:
                    action = env.action_space.sample()
                    epsilon *= epsilon_decay
        else:
            action = int(np.argmax(q[posToQ(state[0])][velToQ(state[1])]))


        '''noise = np.random.randn(1, 3) * (1. / (episode+1)) ** 0.75
        action = 0

        input = True
        for i in q[posToQ(state[0])][velToQ(state[1])]:
            if i != 0.0:
                input = False  # means there are values
                break
        if not input:
            action = random.randint(0, 2)
        else:
            action = np.argmax(q[posToQ(state[0]), velToQ(state[1]), :]+noise)'''

        # take a step
        next_state, reward, done, info = env.step(action)

        if learn:
            next_max = np.max(q[posToQ(next_state[0]), velToQ(next_state[1]), :])

            reward = posToQ(next_state[0]) / 40.0
            # reward = abs(next_state[1])/0.07
            #if next_state[0] > 0.55:
            #    reward = 10
            #    print("YESS WE DID IT!!!")
            # update Q table with reward
            next_value = (1-alpha)*q[posToQ(state[0]), velToQ(state[1]), action] + alpha * (
                reward+gamma * next_max-q[posToQ(state[0]), velToQ(state[1]), action])

            q[posToQ(next_state[0])][velToQ(next_state[1])][action] = next_value
            state = next_state

        #if done:
        #    np.save('trained_Q_DONE', q)
        #    break
    if (episode+1) % 100 == 0:
        print("stage {}".format(episode+1))
    if episode % 1000 == 0 and learn:
        import threading

        t = threading.Thread(target=plott(q))
        t.daemon = True
        t.start()
    if episode % 5000 == 0:
        np.save('trained_Qs', q)
env.close()

if episode % 498 == 0:
    import matplotlib.pyplot as plt2

    plt2.imshow(q, interpolation='nearest', cmap=plt.cm.ocean)
    plt2.colorbar()
    plt2.show()

# Observation has 4 spots. so it's   Obs[0]*y1 + Obs[1]*y2 + Obs[2]*y3 + Obs[3]*y4 = [-infinity..+infinity]
