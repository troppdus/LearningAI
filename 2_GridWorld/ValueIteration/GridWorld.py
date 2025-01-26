# From tutorial at: https://medium.com/@ngao7/markov-decision-process-policy-iteration-42d35ee87c82
# First test from 26.01.25

# Problem: Standard GridWorld problem

import numpy as np
import matplotlib.pyplot as plt

# Problem 
gridWorld = [
             [-1, -1, -1, -1, -1, -1],
             [-1,  0,  0,  0,  1, -1],
             [-1,  0, -1,  0,  2, -1],
             [-1,  0,  0,  0,  0, -1],
             [-1, -1, -1, -1, -1, -1]
            ]
gridWorldWidth = len(gridWorld[0])
def printGridWorld():
    for row in gridWorld:
        print(row)

def posToState(x, y):
    return x * gridWorldWidth + y

def stateToPos(state):
    return state // gridWorldWidth, state % gridWorldWidth

# States
def initStates():
    states = []
    goalState = []
    errorState = []
    i = 0
    for row in gridWorld:
        for val in row:
            if val != -1:
                states.append(i)

            if val == 1:
                goalState.append(i)
            elif val == 2:
                errorState.append(i)
            i += 1

    return states, goalState, errorState

states, goalState, errorState = initStates()

# Actions
actions = ['Right', 'Left', 'Up', 'Down']

# Rewards
# rewards = {
#         0: {'Right': -0.04, 'Left': -0.04, 'Up': -0.04, 'Down': -0.04},
#         1: {'Right': -0.04, 'Left': -0.04, 'Up': -0.04, 'Down': -0.04},
#         2: {'Right': -0.04, 'Left': -0.04, 'Up': -0.04, 'Down': -0.04},
#         3: {'Right': 1.0, 'Left': 1.0, 'Up': 1.0, 'Down': 1.0},
#         4: {'Right': -0.04, 'Left': -0.04, 'Up': -0.04, 'Down': -0.04},
#         6: {'Right': -0.04, 'Left': -0.04, 'Up': -0.04, 'Down': -0.04},
#         7: {'Right': -1.0, 'Left': -1.0, 'Up': -1.0, 'Down': -1.0},
#         8: {'Right': -0.04, 'Left': -0.04, 'Up': -0.04, 'Down': -0.04},
#         9: {'Right': -0.04, 'Left': -0.04, 'Up': -0.04, 'Down': -0.04},
#         10: {'Right': -0.04, 'Left': -0.04, 'Up': -0.04, 'Down': -0.04},
#         11: {'Right': -0.04, 'Left': -0.04, 'Up': -0.04, 'Down': -0.04},
#     }
def rewards(state):
    if state in goalState:
        return 1
    elif state in errorState:
        return -1
    else:
        return -0.04

# Transition
# transitions = {
#         0: {'Right': {1: 0.8, 4: 0.1, 0: 0.1},
#             'Left': {0: 0.9, 4: 0.1},
#             'Up': {0: 0.9, 1: 0.1},
#             'Down': {4: 0.8, 0: 0.1, 1: 0.1}},
#     }
def transitions(state, action):
    if state in goalState or state in errorState:
        return [(state, 1)]
    # [(state_, prob)]
    trans = []
    x, y = stateToPos(state)

    if action == 'Right':
        prob = 0
        if gridWorld[x][y+1] == -1:
            prob += 0.8
        else:
            trans.append((posToState(x, y+1), 0.8))
            
        if gridWorld[x-1][y] == -1:
            prob += 0.1
        else:
            trans.append((posToState(x-1, y), 0.1))
            
        if gridWorld[x+1][y] == -1:
            prob += 0.1
        else:
            trans.append((posToState(x+1, y), 0.1))

        if prob > 0:
            trans.append((state, prob))
    
    elif action == 'Left':
        prob = 0
        if gridWorld[x][y-1] == -1:
            prob += 0.8
        else:
            trans.append((posToState(x, y-1), 0.8))
            
        if gridWorld[x-1][y] == -1:
            prob += 0.1
        else:
            trans.append((posToState(x-1, y), 0.1))
            
        if gridWorld[x+1][y] == -1:
            prob += 0.1
        else:
            trans.append((posToState(x+1, y), 0.1))

        if prob > 0:
            trans.append((state, prob))
    
    elif action == 'Up':
        prob = 0
        if gridWorld[x-1][y] == -1:
            prob += 0.8
        else:
            trans.append((posToState(x-1, y), 0.8))
            
        if gridWorld[x][y-1] == -1:
            prob += 0.1
        else:
            trans.append((posToState(x, y-1), 0.1))
            
        if gridWorld[x][y+1] == -1:
            prob += 0.1
        else:
            trans.append((posToState(x, y+1), 0.1))

        if prob > 0:
            trans.append((state, prob))
    
    elif action == 'Down':
        prob = 0
        if gridWorld[x+1][y] == -1:
            prob += 0.8
        else:
            trans.append((posToState(x+1, y), 0.8))
            
        if gridWorld[x][y-1] == -1:
            prob += 0.1
        else:
            trans.append((posToState(x, y-1), 0.1))
            
        if gridWorld[x][y+1] == -1:
            prob += 0.1
        else:
            trans.append((posToState(x, y+1), 0.1))

        if prob > 0:
            trans.append((state, prob))
            
    return trans

discount = 0.5



def value_iteration():
    V = {s: 0 for s in states}
    while True:
        V_new = {}
        for s in states:
            values = []
            for a in actions:
                value = 0
                for s_, prob in transitions(s, a):
                    value += prob * V[s_]
                values.append(value)
            V_new[s] = rewards(s) + (discount * max(values))

        if all(abs(V[s] - V_new[s]) < 0.0001 for s in states):
            return V_new
        V = V_new
        print("V = ", V)

V = value_iteration()
print("V finished = ", V)

policy = {}
for s in states:
    values = []
    for a in actions:
        value = 0
        for s_, prob in transitions(s, a):
            value += prob * V[s_]
        values.append(rewards(s) + (discount * value))
    policy[s] = actions[np.argmax(values)]

print("Optimal policy:")
print(policy)