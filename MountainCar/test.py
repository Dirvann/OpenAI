import gym
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


env = gym.make('MountainCar-v0')


# iterate through episodes
for episode in range(10000000):
    state = env.reset()

    # iterate through steps
    while True:
        env.render()

        action = 2
        # take a step
        next_state, reward, done, info = env.step(action)

        if done:
            break
env.close()

# Observation has 4 spots. so it's   Obs[0]*y1 + Obs[1]*y2 + Obs[2]*y3 + Obs[3]*y4 = [-infinity..+infinity]
