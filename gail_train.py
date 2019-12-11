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
from envs.dataset import *
from envs.melee_env import MeleeEnv
import argparse

# hyperparameters
learning_rate = 3e-4

# Constants
gamma = 0.99
num_steps = 300
max_episodes = 3000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=time.asctime(), help='Name of this run')
    parser.add_argument('--log', default=False, action='store_true')
    parser.add_argument('--data', default='../test_data/', help='Path to data')
    parser.add_argument(
            '--render', default=False, action='store_true', help='Display Dolphin GUI')
    parser.add_argument(
            '--eval', default=False, action='store_true', help='Run evaluation games')
    parser.add_argument(
            '--warm_start', default=None, help='Directory of human replay data')
    parser.add_argument('--self_play', default=False, action='store_true')
    parser.add_argument(
            '--iso_path', default='../smash.iso', help='Path to MELEE iso')
    parser.add_argument(
            '--model_path', default='weights/', help='Path to store weights')
    parser.add_argument('--load_model', default=None, help='Load model from file')
    parser.add_argument(
            '--num_episodes', type=int, default=10, help='# of games to play')
    args = parser.parse_args()
    lr = 0.0002                 # learing rate
    betas = (0.5, 0.999)        # betas for adam optimizer
    states, actions, action_set = get_data_from_logs(args.data, one_hot_actions = True)
    model = GAIL(states, actions, action_set, lr, betas)
    env = MeleeEnv(action_set,
                   log=args.log,
                   render=args.render,
                   iso_path=args.iso_path)
    gen_losses, discrim_losses = do_warm_start(model, env, states, actions)
    fig, ax = plt.subplots(2)
    ax[0].plot(gen_losses)
    ax[1].plot(discrim_losses)
    plt.show()

def do_warm_start(model, env, states, actions):
    gen_losses = list()
    discrim_losses = list()
    for i in range(1):
        print(i)
        gen_loss, discrim_loss = model.update(1000)
        gen_losses.append(gen_loss)
        discrim_losses.append(discrim_loss)
    validate_on_cpu(model, env)
    return gen_losses, discrim_losses

def validate_on_cpu(model, env):
    eval_done = False
    eval_score = 0
    eval_state = env.reset()
    while not eval_done:
        eval_action = model.select_action(torch.FloatTensor(eval_state))
        eval_state, eval_reward, eval_done = env.step(eval_action)
        eval_score += eval_reward
    while (env.gamestate.menu_state in [
           melee.enums.Menu.IN_GAME, melee.enums.Menu.SUDDEN_DEATH] and (
            env.gamestate.ai_state.stock == 0 or
            env.gamestate.opponent_state.stock == 0)):
            env.gamestate.step()
    return eval_score

if __name__ == "__main__":
    main()