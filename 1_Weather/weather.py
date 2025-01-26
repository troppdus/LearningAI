# From tutorial at: https://www.youtube.com/watch?v=UuTkioxL9bQ
# First test from 26.01.25

# Problem: You're going out, do you need an umbrella?

# States: Rainy, Sunny and Cloudy
# Actions: take umbrella, don't take umbrella
# Rewards: Dict mapping with each state and action pair
# Transition: Prob
# Discount

import numpy as np
import matplotlib.pyplot as plt

states = ['Rainy', 'Cloudy', 'Sunny']
actions = ['Umbrella', 'No Umbrella']
rewards = {
        'Rainy': {'Umbrella': -1, 'No Umbrella': -5},
        'Cloudy': {'Umbrella': -1, 'No Umbrella': -1},
        'Sunny': {'Umbrella': -5, 'No Umbrella': -1}
    }
transitions = {
        'Rainy': {'Umbrella': {'Rainy': 0.7, 'Cloudy': 0.3, 'Sunny': 0},
                  'No Umbrella': {'Rainy': 0.3, 'Cloudy': 0.4, 'Sunny': 0.3}},
        'Cloudy': {'Umbrella': {'Rainy': 0.4, 'Cloudy': 0.6, 'Sunny': 0},
                  'No Umbrella': {'Rainy': 0, 'Cloudy': 0.7, 'Sunny': 0.3}},
        'Sunny': {'Umbrella': {'Rainy': 0, 'Cloudy': 0, 'Sunny': 1},
                  'No Umbrella': {'Rainy': 0, 'Cloudy': 0.4, 'Sunny': 0.6}}
    }
discount = 0.9

def value_iteration():
    V = {s: 0 for s in states}
    while True:
        V_new = {}
        for s in states:
            values = []
            for a in actions:
                value = rewards[s][a]
                for s_ in states:
                    value += discount * transitions[s][a][s_] * V[s_]
                values.append(value)
            V_new[s] = max(values)

        if all(abs(V[s] - V_new[s]) < 0.0001 for s in states):
            return V_new
        V = V_new
        print("V not finished yet = ", V)

V = value_iteration()
print("V finished = ", V)

policy = {}
for s in states:
    values = []
    for a in actions:
        value = rewards[s][a]
        for s_ in states:
            value += discount * transitions[s][a][s_] * V[s_]
        values.append(value)
    policy[s] = actions[np.argmax(values)]

print("Optimal policy:")
print(policy)

policy_values = np.zeros((len(states), len(actions)))
for i, s in enumerate(states):
    for j, a in enumerate(actions):
        policy_values[i, j] = rewards[s][a]
        for s_ in states:
            policy_values[i, j] += discount * transitions[s][a][s_] * V[s_]

plt.imshow(policy_values, cmap='Greys')
plt.show()