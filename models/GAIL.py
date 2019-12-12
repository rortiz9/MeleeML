import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.cbow import CBOW, CBOW_state
from collections import deque
from envs.dataset import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_window_size, encoder = None):
        super(Actor, self).__init__()
        self.encoder = encoder
        self.max_window_size = max_window_size
        self.hidden_dim = 100
        self.n_layers = 3
        self.lstm = nn.LSTM(state_dim + action_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.flatten = nn.Flatten()
        if self.encoder == None:
            self.l1 = nn.Linear(state_dim, 150)
        else:
            self.l1 = nn.Linear(150, 150)
        self.l2 = nn.Linear(150, 150)
        self.l3 = nn.Linear((self.hidden_dim * (self.max_window_size - 1)) + 150, action_dim)

    def forward(self, prev_state, prev_action, current_state):
        prev_state_action = torch.cat([prev_state, prev_action], 2)
        batch_size = prev_state_action.shape[0]

        # Lstm on Previous State Action pair
        self.hidden = self.init_hidden(batch_size)
        prev_state_action, self.hidden = self.lstm(prev_state_action, self.hidden)

        # Embedding New State
        if self.encoder is not None:
            current_state = self.encoder.encode(current_state)
        current_state = F.relu(self.l1(current_state))
        current_state = F.relu(self.l2(current_state))

        x = torch.cat([self.flatten(prev_state_action), current_state], 1)
        x = F.relu(self.l3(x))
        return nn.Softmax()(x)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, max_window_size, encoder = None):
        super(Discriminator, self).__init__()
        self.encoder = None
        #self.encoder = encoder
        self.max_window_size = max_window_size
        self.hidden_dim = 300
        self.n_layers = 1
        if self.encoder == None:
            self.lstm = nn.LSTM(state_dim+action_dim, self.hidden_dim, self.n_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(200, self.hidden_dim, self.n_layers, batch_first=True)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(self.hidden_dim * self.max_window_size, 300)
        self.l2 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 2)
        if self.encoder is not None:
            state_action = self.encoder.encode(state, action)
        batch_size = state_action.shape[0]

        # Lstm
        self.hidden = self.init_hidden(batch_size)
        output, self.hidden = self.lstm(state_action, self.hidden)

        output = self.flatten(output)
        output = torch.sigmoid(self.l1(output))
        output = torch.sigmoid(self.l2(output))
        return output

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

class GAIL:
    def __init__(self, expert_states, expert_actions, action_set, lr, betas):
        state_dim = expert_states.shape[1]
        action_dim = expert_actions.shape[1]
        self.action_set = action_set
        self.max_window_size = 4

        self.expert_states = expert_states
        self.expert_actions = expert_actions
        self.state_action_enc = CBOW(state_dim,  action_dim)
        self.state_action_enc.load()

        self.state_enc = CBOW_state(state_dim)
        self.state_enc.load()

        self.actor = Actor(state_dim, action_dim, self.max_window_size, encoder = self.state_enc).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr*5, betas=betas)

        self.discriminator = Discriminator(state_dim, action_dim, self.max_window_size, encoder = self.state_action_enc).to(device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr/200., betas=betas)

        self.loss_fn = nn.BCELoss()

        self.state_quque = deque()
        self.action_quque = deque()

    def select_action(self, state):
        if self.max_window_size == len(self.state_quque):
            # Update Quque
            self.state_quque.popleft()
            self.action_quque.popleft()

            # Get Previous States and Actions
            prev_states = torch.cat(list(self.state_quque), 0)
            prev_actions = torch.cat(list(self.action_quque), 0)

            # Get Action
            out = self.actor.forward(prev_states.view(1, self.max_window_size - 1, -1), prev_actions.view(1, self.max_window_size - 1, -1), state)
            action_dist = torch.distributions.Categorical(out.squeeze(0))
            action_idx = action_dist.sample()
            eval_action = torch.zeros((self.action_set.shape[0]))
            eval_action[action_idx] = 1

        else:

            # Return Nothing
            eval_action = torch.zeros((self.action_set.shape[0]))
            eval_action[0] = 1

        # Add New Pair
        self.state_quque.append(state)
        self.action_quque.append(eval_action.view(1, -1))

        return eval_action

    def update(self, n_iter, batch_size=100, entropy_penalty = True, pg_penalty = True):
        gen_losses = list()
        discrim_losses = list()
        for ii in range(n_iter):
            # sample expert transitions
            indexes = list()
            while len(indexes) < batch_size:
                idx = np.random.randint(self.expert_states.shape[0])

                # Check if states are inbetween games: 5 and 252 is the feature for stock hopefully
                if idx + self.max_window_size < self.expert_states.shape[0] and self.expert_states[idx][5] >= self.expert_states[idx + self.max_window_size][5] and self.expert_states[idx][252] >= self.expert_states[idx + self.max_window_size][252]:
                    indexes.append(np.arange(idx, idx + self.max_window_size))
            indexes = [indexes]

            exp_states = torch.FloatTensor(self.expert_states[indexes]).to(device)
            exp_actions = torch.FloatTensor(self.expert_actions[indexes]).to(device)

            # sample expert states for actor
            indexes = list()
            while len(indexes) < batch_size:
                idx = np.random.randint(self.expert_states.shape[0])

                # Check if states are inbetween games
                if idx + self.max_window_size < self.expert_states.shape[0] and self.expert_states[idx][5] >= self.expert_states[idx + self.max_window_size][5] and self.expert_states[idx][252] >= self.expert_states[idx + self.max_window_size][252]:
                    indexes.append(np.arange(idx, idx + self.max_window_size))
            indexes = [indexes]
            states = torch.FloatTensor(self.expert_states[indexes]).to(device)
            actions = torch.FloatTensor(self.expert_actions[indexes]).to(device)

            # Run Actor
            action = self.actor(states[:, :-1], actions[:, :-1], states[:, -1])

            #######################
            # update discriminator
            #######################
            self.optim_discriminator.zero_grad()

            # label tensors
            exp_label= torch.full((batch_size,1), 1, device=device)
            policy_label = torch.full((batch_size,1), 0, device=device)

            # with expert transitions
            prob_exp = self.discriminator(exp_states, exp_actions)
            loss = self.loss_fn(prob_exp, exp_label)

            # with policy transitions
            actions[:, -1, :] = action.detach()
            prob_policy = self.discriminator(states, actions)
            loss += self.loss_fn(prob_policy, policy_label)

            # take gradient step
            loss.backward()
            self.optim_discriminator.step()

            ################
            # update policy
            ################
            self.optim_actor.zero_grad()

            loss_actor = self.loss_fn(self.discriminator(states, actions), exp_label)

            if entropy_penalty:
                entropy = -torch.sum(torch.mean(action) * torch.log(action))
                new_loss = loss_actor + 0.01 * entropy
                new_loss.mean().backward()

            elif pg_penalty:
                 #pg loss
                reward = torch.Tensor(get_rewards(states[:, -2:, :])).to(device)
                indexes = np.array(indexes)[:, :, -1]
                correct_actions_onehot = self.expert_actions[indexes]
                action_indices = torch.Tensor(np.where(correct_actions_onehot == 1)[0]).long().to(device)
                action_indices = action_indices.unsqueeze(0).T
                log_prob = action.gather(1, action_indices)
                pg_loss = -log_prob * reward

                new_loss = loss_actor + 0.01 * pg_loss
                new_loss.mean().backward()

            else:
                loss_actor.mean().backward()

            self.optim_actor.step()
            gen_losses.append(loss_actor.mean())
            discrim_losses.append(loss)
        avg_gen_loss = sum(gen_losses)/len(gen_losses)
        avg_discrim_loss = sum(discrim_losses)/len(gen_losses)
        return avg_gen_loss, avg_discrim_loss


    def save(self, directory='./preTrained', name='GAIL'):
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory,name))
        torch.save(self.discriminator.state_dict(), '{}/{}_discriminator.pth'.format(directory,name))

    def load(self, directory='./preTrained', name='GAIL'):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory,name), map_location=torch.device('cpu')))
        self.discriminator.load_state_dict(torch.load('{}/{}_discriminator.pth'.format(directory,name), map_location=torch.device('cpu')))

