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

# this is just here as a direct copy-paste from:
# https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
def _copy_pasted_train_func():
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    actor_critic = ActorCriticPolicy(num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        for steps in range(num_steps):
            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy()

            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state

            if done or steps == num_steps-1:
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0,0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                break

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        #update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()



    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothend_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()

def main():
    model = ActorCriticPolicy(495, 345, 200)
    ac_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mean_losses, mean_actors, mean_critics = do_warm_start(model, ac_optimizer)
    fig, ax = plt.subplots(2)
    ax[0].plot(mean_actors)
    ax[1].plot(mean_critics)
    plt.show()

def do_warm_start(model, optimizer):
    states, actions, action_set = get_data_from_logs("/home/sc/school/cs8803/libmelee/logs/", one_hot_actions = True)
    mean_losses = list()
    mean_actors, mean_critics = list(), list()
    for i in range(states.shape[0]):
        if i % 1000 == 0:
            print(i)
            val_loss, val_act, val_crit = validate(model, states, actions)
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
    return mean_losses, mean_actors, mean_critics

def validate(model, states, actions):
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

if __name__ == "__main__":
    main()
