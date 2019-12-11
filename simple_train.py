import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from models.ActorCriticPolicy import ActorCriticPolicy
from envs.dataset import *

# hyperparameters
learning_rate = 3e-4

# Constants
gamma = 0.99
num_steps = 300
max_episodes = 3000

def main():
    states, actions, action_set = get_data_from_logs("/home/sc/school/cs8803/libmelee/logs/", one_hot_actions = True)
    model = ActorCriticPolicy(495, 345, 200)
    ac_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    '''
    env = MeleeEnv(action_set,
                   log=args.log,
                   render=args.render,
                   self_play=args.self_play,
                   iso_path=args.iso_path)
    '''
    env = None
    losses = do_warm_start(model, ac_optimizer, env, states, actions)
    plt.plot(losses)
    plt.show()

def do_warm_start(model, optimizer, env, states, actions):
    mean_losses = list()
    mean_actors, mean_critics = list(), list()
    for i in range(states.shape[0]):
        if i % 10000 == 0:
            print(i)
            val_loss, val_act, val_crit = validate_on_data(model, states, actions)
            mean_losses.append(val_loss)
            mean_actors.append(val_act)
            mean_critics.append(val_crit)
        state = states[i]
        if state[5] == 0 or state[20] == 0:
                continue
        action = actions[i]
        next_state = states[i + 1]
        done = False
        p1_score = (1000 * (next_state[5] - state[5]) -
                    (next_state[4] - state[4]))
        p2_score = (1000 * (next_state[20] - state[20]) -
                    (next_state[19] - state[19]))
        if next_state[5] == 0 or next_state[20] == 0:
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
    p2_scores = (1000 * (next_states[:,20] - states[:,20]) -
        next_states[:,19] - states[:,19])
    rewards = p1_scores - p2_scores
    values, policy_dists = model.forward(states)
    next_values = model.forward(next_states)
    dones = not (next_state[:,5] == 0 or next_states[:,20] == 0)
    dones = dones.float()
    Q = reward + dones * next_values
    advantages = Q - values'''
    ac_losses = list()
    actor_losses, critic_losses = list(), list()
    for i in range(states.shape[0]):
        state = states[i]
        if state[5] == 0 or state[20] == 0:
                continue
        action = actions[i]
        next_state = states[i + 1]
        done = False
        p1_score = (1000 * (next_state[5] - state[5]) -
                    (next_state[4] - state[4]))
        p2_score = (1000 * (next_state[20] - state[20]) -
                    (next_state[19] - state[19]))
        if next_state[5] == 0 or next_state[20] == 0:
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
        eval_action = model.forward(eval_state)
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
