import pandas as pd
import numpy as np

import argparse
import os

from melee import enums


def get_data_from_logs(file_dir, both=True, one_hot_actions = False):
    file_list = list()
    for root, dirnames, filenames in os.walk(file_dir):
        for filename in filenames:
            if 'game' in filename and '.csv' in filename:
                file_list.append(os.path.join(root, filename))

    states = list()
    actions = list()

    for file_path in file_list:
        try:
            state, action = read_data(file_path)
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

        except:
            print("File with possibly no data")
            print(file_path)

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


"""
Params: states
Return: New Normalized States
495 new state size
indexes [2, 3, 240, 245, 246, 249, 250, 487, 492, 493] Normalize from -1 to 1
rest is 0 to 1
"""
def preprocess_states(states):
    if len(states.shape) == 1:
        states = states.reshape(1,  -1)
    # Normalize Data
    states[:, 0] /= 26
    # character, x, y, percent, stock, action_frame, jumps_left, speed_x, speed_y
    nommalizing_per_player = np.array([66, 300, 250, 1000, 4, 120, 10, 100, 100])
    nommalizing_indexes = np.array([0, 1, 2, 3, 4, 7, 10, 12, 13])
    states[:, 1:16][:, nommalizing_indexes] = states[:, 1:16][:, nommalizing_indexes] / nommalizing_per_player
    states[:, 16:][:, nommalizing_indexes] = states[:, 16:][:, nommalizing_indexes] / nommalizing_per_player

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

#Todo kinda works
# Param: batch, seq, features
# Returns a list of rewards
def get_rewards(states):
    rewards = [-0.01]

    for jj in range(states.shape[0]):
        batch_state = states[jj]
        for ii in range(1, states.shape[1]):
            p1_stock = batch_state[ii][5]
            p1_percent = batch_state[ii][4]
            p2_stock = batch_state[ii][252]
            p2_percent = batch_state[ii][251]
            prev_p1_stock = batch_state[ii - 1][5]
            prev_p1_percent = batch_state[ii - 1][4]
            prev_p2_stock = batch_state[ii - 1][252]
            prev_p2_percent = batch_state[ii - 1][251]

            if prev_p1_stock >= p1_stock and prev_p2_stock >= p2_stock:
                # Percent
                reward = 1000 * (prev_p1_percent - p1_percent - prev_p2_percent + p2_percent)

                # Stock
                reward += -4000 * (prev_p1_stock - p1_stock - prev_p2_stock + p2_stock)

            else:
                reward = 0

            if reward == 0:
                reward = -0.01

            rewards.append(reward)
        
    return rewards