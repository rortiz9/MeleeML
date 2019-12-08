import pandas as pd
import numpy as np

import argparse
import os

from melee import enums

def get_data_from_logs(file_dir, both=True, one_hot_actions = False):
    file_list = os.listdir(file_dir)

    states = list()
    actions = list()

    for file in file_list:
        if 'game' in file and '.csv' in file:
            state, action = read_data(file_dir + file)
            num_actions = action.shape[1]//2
            num_states = state.shape[1]//2
            if both:
                states.append(state)
                states.append(np.hstack([state[:, 0:1], state[:, -num_states:], state[:, 1:num_states + 1]]))
                actions.append(action[:, :num_actions])
                actions.append(action[:, num_actions:])

            else:
                # Use only player1's actions
                states.append(state)
                actions.append(action[:, :num_actions])

    # Preprocess
    states = preprocess_states(np.vstack(states))
    if one_hot_actions:
        action_set, actions =  preprocess_actions(np.vstack(actions))
        return states, actions, action_set
    return states, np.vstack(actions)


def read_data(file_path, num_states=31, num_actions=32):
    data = np.array(pd.read_csv(file_path, header=None))
    states = data[:, 1:num_states + 1]
    actions = data[:, -num_actions:]
    return states, actions


def preprocess_states(states):
    # Convert Actions to One Hot Encodeing
    num_actions = 233
    action2idx_mapping = dict()

    # Convert Value to Indexes
    for action in enumerate(enums.Action):
        action2idx_mapping[action[1].value] = int(action[0])

    for ii in range(states.shape[0]):
        states[ii][7] = action2idx_mapping[states[ii][7]]
        states[ii][22] = action2idx_mapping[states[ii][22]]

    # Convert to One Hot Encodeing
    p1_actions = np.eye(num_actions)[states[:,7].astype(int)]
    p2_actions = np.eye(num_actions)[states[:,22].astype(int)]

    # Merge
    new_states = np.hstack([states[:, :7], p1_actions, states[:, 8:22], p2_actions, states[:, 23:]])

    return new_states

def preprocess_actions(actions):
    '''
    First we compile the actions into an intermediary representation where
    x, y are lumped together
    L, R are lumped together and are either 0, 1, 2 (no shield, half shield, power shield)
    Analog_x, analog_y are discretized between 0 - 4
    (far left, tilt left, neutral, tilt right, far right) or down/up respectively
    C-stick is discretized between 0 - 8
    (neutral, left, left-down, left-up, right, right-down, right-up, up, down)

    Then we generate a unique set of (intermediary) actions. Each action then gets a one-hot
    encoding based on its index in this set
    '''
    # A, B, Jump, Grab, SHIELD, ANALOG_X, ANALOG_Y, C-STICK
    intermediary_actions = np.zeros((actions.shape[0], 8))
    intermediary_actions[:, :2] = actions[:, :2] # A, B
    intermediary_actions[:, 2] = np.logical_or(actions[:, 2], actions[:, 3]).astype(int) #Jump
    intermediary_actions[:, 3] = actions[:, 3] #Grab

    shield = np.maximum(actions[:, 10], actions[:, 11])
    intermediary_actions[shield > 59, 4] = 1 # half-shield
    intermediary_actions[shield > 121, 4] = 2 # full shield

    intermediary_actions[actions[:, 12] > 50, 5] = 1 #ANA: X,tilt-left
    intermediary_actions[actions[:, 12] > 101, 5] = 2
    intermediary_actions[actions[:, 12] > 152, 5] = 3
    intermediary_actions[actions[:, 12] > 203, 5] = 4
    intermediary_actions[actions[:, 13] > 50, 6] = 1 # ANA: Y, tilt-down
    intermediary_actions[actions[:, 13] > 101, 6] = 2
    intermediary_actions[actions[:, 13] > 152, 6] = 3
    intermediary_actions[actions[:, 13] > 203, 6] = 4

    L_smash = actions[:, 14] < 101
    R_smash = actions[:, 14] > 152
    down_smash = actions[:, 15] < 101
    up_smash = actions[:, 15] > 152
    intermediary_actions[np.logical_and(L_smash, np.logical_not(down_smash), np.logical_not(up_smash)),7] = 1
    intermediary_actions[np.logical_and(L_smash, down_smash),7] = 2
    intermediary_actions[np.logical_and(L_smash, up_smash), 7] = 3
    intermediary_actions[np.logical_and(R_smash, np.logical_not(down_smash), np.logical_not(up_smash)),7] = 4
    intermediary_actions[np.logical_and(R_smash, down_smash), 7] = 5
    intermediary_actions[np.logical_and(R_smash, up_smash), 7] = 6
    intermediary_actions[np.logical_and(up_smash, np.logical_not(R_smash), np.logical_not(L_smash)), 7] = 7
    intermediary_actions[np.logical_and(down_smash, np.logical_not(R_smash), np.logical_not(L_smash)), 7] = 8

    action_set = np.unique(intermediary_actions, axis = 0)

    # can't find a good vectorized way to do this
    one_hot_actions = np.zeros((actions.shape[0], action_set.shape[0]))
    for i in range(one_hot_actions.shape[0]):
        # https://stackoverflow.com/questions/18927475/numpy-array-get-row-index-searching-by-a-row
        idx = np.where(np.all(action_set == intermediary_actions[i], axis = 1))
        one_hot_actions[i, idx] = 1
    return action_set, one_hot_actions
