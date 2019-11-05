import pandas as pd
import numpy as np

import argparse
import os


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


    return np.vstack(states), np.vstack(actions)


def read_data(file_path, num_states=31, num_actions=32):
    data = np.array(pd.read_csv(file_path, header=None))
    states = data[:, 1:num_states + 1]
    actions = data[:, -num_actions:]
    return states, actions
