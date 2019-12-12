import sys
import melee
import torch
import gym
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from models.ActorCriticPolicy import ActorCriticPolicy
from models.GAIL import  GAIL
from models.cbow import CBOW_state
from envs.dataset import *
from envs.melee_env import MeleeEnv
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../test_data/', help='Path to data')
    args = parser.parse_args()

    states, actions, action_set = get_data_from_logs(args.data, one_hot_actions = True)
    print("samples: ", states.shape[0])
    model = CBOW_state(states.shape[1])
    train_size = int(0.85 * states.shape[0])
    train_states = torch.Tensor(states[:train_size])
    test_states = torch.Tensor(states[train_size:])

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    losses = list()
    val_losses = list()

    i = 0
    while i + 1000 < train_size:
        if i % 50000 ==  0:
            print(i)
            state_action = model.forward(test_states)
            loss = loss_fn(state_action, test_states)
            print(loss.mean())
            val_losses.append(loss.mean())
            del state_action
        optimizer.zero_grad()
        #state, action = torch.unsqueeze(train_states[i], 0), torch.unsqueeze(train_actions[i], 0)
        states = train_states[i:i + 1000]
        state_pred = model.forward(states)
        loss = loss_fn(state_pred, states)
        loss.backward()
        optimizer.step()
        losses.append(loss.mean())
        del states
        del state_pred
        i += 1000
    
    plt.title("Reconstruction Loss (Validation)")
    plt.xlabel("Epoch")
    plt.plot(val_losses)
    plt.show()

    model.save()

if __name__ == "__main__":
    main()
