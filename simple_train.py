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

    states, actions, action_set = get_data_from_logs(args.data, one_hot_actions = True)
    model = ActorCriticPolicy(495, 345, 200)
    ac_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    env = MeleeEnv(action_set,
                   log=args.log,
                   render=args.render,
                   iso_path=args.iso_path)
    losses = do_warm_start(model, ac_optimizer, env, states, actions)
    plt.plot(losses)
    plt.show()

def do_warm_start(model, optimizer, env, states, actions):
    mean_losses = list()
    mean_actors, mean_critics = list(), list()
    for i in range(states.shape[0]):
        if i % 50000 == 0:
            print("iter: ", i)
            #val_loss, val_act, val_crit = validate_on_dataset(model, states, actions)
            val_loss = validate_on_cpu(model, env)
            mean_losses.append(val_loss)
        state = states[i]
        if state[5] == 0 or state[252] == 0:
                continue
        action = actions[i]
        next_state = states[i + 1]
        done = False
        p1_score = (1000 * (next_state[5] - state[5]) -
                    (next_state[4] - state[4]))
        p2_score = (1000 * (next_state[252] - state[252]) -
                    (next_state[251] - state[251]))
        if next_state[5] == 0 or next_state[252] == 0:
            done = True
        reward = p1_score - p2_score

        value, policy_dist = model.forward(state)
        dist = policy_dist.detach().numpy()
        #value = value.detach().numpy()[0, 0]

        # TODO is this correct?
        Q = torch.Tensor([[reward]])
        if not done:
            next_value, _ = model.forward(next_state)
            next_value = next_value.detach().numpy()[0, 0]
            Q += gamma * next_value
        #advantage = torch.FloatTensor([Q - value])
        advantage = (Q - value).squeeze(0)
        true_action = np.where(action == 1)[0]
        log_prob = torch.log(policy_dist.squeeze(0)[true_action])
        actor_loss = -log_prob * advantage
        critic_loss = 0.5 * torch.pow(advantage, 2)
        ac_loss = actor_loss + critic_loss

        optimizer.zero_grad()
        ac_loss.backward()
        optimizer.step()
    return mean_losses

def validate_on_data(model, states, actions):
    '''next_states = np.roll(states, -1, axis = 0)
    p1_scores = (1000 * (next_states[:,5] - states[:, 5]) -
        next_states[:,4] - states[:, 4])
    p2_scores = (1000 * (next_states[:,252] - states[:,252]) -
        next_states[:,251] - states[:,251])
    rewards = p1_scores - p2_scores
    values, policy_dists = model.forward(states)
    next_values = model.forward(next_states)
    dones = not (next_state[:,5] == 0 or next_states[:,252] == 0)
    dones = dones.float()
    Q = reward + dones * next_values
    advantages = Q - values'''
    ac_losses = list()
    actor_losses, critic_losses = list(), list()
    for i in range(states.shape[0]):
        state = states[i]
        if state[5] == 0 or state[252] == 0:
                continue
        action = actions[i]
        next_state = states[i + 1]
        done = False
        p1_score = (1000 * (next_state[5] - state[5]) -
                    (next_state[4] - state[4]))
        p2_score = (1000 * (next_state[252] - state[252]) -
                    (next_state[251] - state[251]))
        if next_state[5] == 0 or next_state[252] == 0:
            done = True
        reward = p1_score - p2_score

        value, policy_dist = model.forward(state)
        dist = policy_dist.detach().numpy()
        value = value.detach().numpy()[0, 0]

        # TODO is this correct?
        Q = reward
        if not done:
            next_value, _ = model.forward(next_state)
            next_value = next_value.detach().numpy()[0, 0]
            Q += gamma * next_value
        advantage = torch.FloatTensor([Q - value])
        true_action = np.where(action == 1)[0]
        log_prob = torch.log(policy_dist.squeeze(0)[true_action])
        actor_loss = -log_prob * advantage
        critic_loss = 0.5 * torch.pow(advantage, 2)
        ac_loss = actor_loss + critic_loss
        actor_losses.append(actor_loss.detach().numpy()[0])
        critic_losses.append(critic_loss.detach().numpy()[0])
        ac_losses.append(ac_loss.detach().numpy()[0])
    mean_loss = sum(ac_losses)/len(ac_losses)
    mean_actor = sum(actor_losses)/len(actor_losses)
    mean_critic = sum(critic_losses)/len(critic_losses)
    return  mean_loss, mean_actor, mean_critic

def validate_on_cpu(model, env):
    eval_done = False
    eval_score = 0
    eval_state = env.reset()
    while not eval_done:
        out = model.forward(eval_state)[1]
        action_dist = torch.distributions.Categorical(out.squeeze(0))
        action_idx = action_dist.sample()
        eval_action = torch.zeros((env.action_set.shape[0]))
        eval_action[action_idx] = 1
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
