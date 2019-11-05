import pandas as pd
import numpy as np

import argparse
import os


def get_data_from_logs(file_dir):
    file_list = os.listdir(file_dir)

    states = list()
    actions = list()

    for file in file_list:
        if 'game' in file and '.csv' in file:
            print("Reading " + file_dir + file)
            state, action = read_data(file_dir + file)
            num_actions = action.shape[1]//2
            num_states = state.shape[1]//2

            if args.both:
                states.append(state)
                states.append(np.hstack([state[:, 0:1], state[:, -num_states:], state[:, 1:num_states + 1]]))
                actions.append(action[:, :num_actions])
                actions.append(action[:, num_actions:])

            else:
                # Use only player1's actions
                states.append(state)
                actions.append(action[:, :num_actions])


    return np.array(np.vstack(states)), np.vstack(actions)


def read_data(file_path, num_states=31, num_actions=32):
    data = np.array(pd.read_csv(file_path, header=None))
    states = data[:, 1:num_states + 1]
    actions = data[:, -num_actions:]
    return states, actions


def main(args):
    states, actions = get_data_from_logs(args.data_dir)
    print(states.shape)
    print(actions.shape)
    np.savetxt("state_examples.csv", states, delimiter=",")
    np.savetxt("action_examples.csv", actions, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read gamestate and GC actions')

    parser.add_argument("--data_dir", type=str, help="path to data", default="logs/")
    parser.add_argument('--both', '-b', default=False, action='store_true', help='Use both player 1 and player 2s actions as data')

    args = parser.parse_args()

    main(args)