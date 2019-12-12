import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_window_size):
        super(Actor, self).__init__()

        self.max_window_size = max_window_size
        self.hidden_dim = 50
        self.n_layers = 3
        self.lstm = nn.LSTM(state_dim + action_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 200)
        self.l3 = nn.Linear((self.hidden_dim * (self.max_window_size - 1)) + 200, action_dim)

    def forward(self, prev_state, prev_action, current_state):
        prev_state_action = torch.cat([prev_state, prev_action], 2)
        batch_size = prev_state_action.shape[0]

        # Lstm on Previous State Action pair
        self.hidden = self.init_hidden(batch_size)
        prev_state_action, self.hidden = self.lstm(prev_state_action, self.hidden)

        # Embedding New State
        current_state = F.relu(self.l1(current_state))
        current_state = F.relu(self.l2(current_state))

        x = torch.cat([self.flatten(prev_state_action), current_state], 1)
        x = F.relu(self.l3(x))
        return nn.Softmax()(x)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device), 
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, max_window_size):
        super(Discriminator, self).__init__()

        self.max_window_size = max_window_size
        self.hidden_dim = 50
        self.n_layers = 3
        self.lstm = nn.LSTM(state_dim+action_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(self.hidden_dim * self.max_window_size, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 2)
        batch_size = state_action.shape[0]

        # Lstm
        self.hidden = self.init_hidden(batch_size)
        output, self.hidden = self.lstm(state_action, self.hidden)

        output = self.flatten(output)
        output = torch.sigmoid(self.l1(output))
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

        self.actor = Actor(state_dim, action_dim, self.max_window_size).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=betas)

        self.discriminator = Discriminator(state_dim, action_dim, self.max_window_size).to(device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        self.loss_fn = nn.BCELoss()

    def select_action(self, state):
        out = self.actor.forward(state)
        action_dist = torch.distributions.Categorical(out.squeeze(0))
        action_idx = action_dist.sample()
        eval_action = torch.zeros((self.action_set.shape[0]))
        eval_action[action_idx] = 1
        return eval_action

    def update(self, n_iter, batch_size=100, entropy_penalty = True):
        gen_losses = list()
        discrim_losses = list()
        for ii in range(n_iter):
            # sample expert transitions
            indexes = list()
            while len(indexes) < batch_size:
                idx = np.random.randint(self.expert_states.shape[0])

                # Check if states are inbetween games
                if idx + self.max_window_size < self.expert_states.shape[0] and self.expert_states[idx][4] >= self.expert_states[idx + self.max_window_size][4]:
                    indexes.append(np.arange(idx, idx + self.max_window_size))
            indexes = [indexes]

            exp_states = torch.FloatTensor(self.expert_states[indexes]).to(device)
            exp_actions = torch.FloatTensor(self.expert_actions[indexes]).to(device)

            # sample expert states for actor
            indexes = list()
            while len(indexes) < batch_size:
                idx = np.random.randint(self.expert_states.shape[0])

                # Check if states are inbetween games
                if idx + self.max_window_size < self.expert_states.shape[0] and self.expert_states[idx][4] >= self.expert_states[idx + self.max_window_size][4]:
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
            entropy = -torch.sum(torch.mean(action) * torch.log(action))
            new_loss = loss_actor + 0.01 * entropy
            #loss_actor += 0.0000 * entropy
            #loss_actor.mean().backward()
            if entropy_penalty:
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

