import pandas as pd
import numpy as np

import argparse
import os

from melee import enums


def get_data_from_logs(file_dir, both=True):
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