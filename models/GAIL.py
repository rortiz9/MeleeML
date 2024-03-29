import torch
import torch.nn as nn
import torch.nn.functional as F
from envs.dataset import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 200)
        self.l3 = nn.Linear(200, action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = nn.Softmax()(self.l3(x))
        #x = torch.sigmoid(self.l3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        self.l1 = nn.Linear(state_dim+action_dim, 500)
        self.l2 = nn.Linear(500, 300)
        self.l3 = nn.Linear(300, 300)
        self.l4 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = torch.tanh(self.l1(state_action))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = torch.sigmoid(self.l4(x))
        return x

class GAIL:
    def __init__(self, expert_states, expert_actions, action_set, lr, betas):
        state_dim = expert_states.shape[1]
        action_dim = expert_actions.shape[1]
        self.action_set = action_set

        self.expert_states = expert_states
        self.expert_actions = expert_actions

        self.actor = Actor(state_dim, action_dim).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=betas)

        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        self.loss_fn = nn.BCELoss()

    def select_action(self, state):
        out = self.actor.forward(state)
        action_dist = torch.distributions.Categorical(out.squeeze(0))
        action_idx = action_dist.sample()
        eval_action = torch.zeros((self.action_set.shape[0]))
        eval_action[action_idx] = 1
        return eval_action

    def update(self, n_iter, batch_size=100, entropy_penalty = True, pg_penalty = True):
        gen_losses = list()
        discrim_losses = list()
        for i in range(n_iter):
            # sample expert transitions
            expert_samples = torch.randint(self.expert_states.shape[0], (batch_size,))
            exp_state = torch.FloatTensor(self.expert_states[expert_samples]).to(device)
            exp_action = torch.FloatTensor(self.expert_actions[expert_samples]).to(device)

            # sample expert states for actor
            actor_samples = torch.randint(self.expert_states.shape[0], (batch_size,))
            state = torch.FloatTensor(self.expert_states[actor_samples]).to(device)
            action = self.actor(state)

            #######################
            # update discriminator
            #######################
            self.optim_discriminator.zero_grad()

            # label tensors
            exp_label= torch.full((batch_size,1), 1, device=device)
            policy_label = torch.full((batch_size,1), 0, device=device)

            # with expert transitions
            prob_exp = self.discriminator(exp_state, exp_action)
            loss = self.loss_fn(prob_exp, exp_label)

            # with policy transitions
            prob_policy = self.discriminator(state, action.detach())
            loss += self.loss_fn(prob_policy, policy_label)

            # take gradient step
            loss.backward()
            self.optim_discriminator.step()

            ################
            # update policy
            ################
            self.optim_actor.zero_grad()

            #loss_actor = -self.discriminator(state, action)
            loss_actor = self.loss_fn(self.discriminator(state, action), exp_label)
            entropy = -torch.sum(torch.mean(action) * torch.log(action))

            #pg loss
            reward = torch.Tensor(get_rewards(state)).to(device)
            correct_actions_onehot = self.expert_actions[actor_samples]
            action_indices = torch.Tensor(np.where(correct_actions_onehot == 1)[0]).long().to(device)
            action_indices = action_indices.unsqueeze(0).T
            #log_prob = torch.log(action.squeeze(0)[action_indices])
            #log_prob = torch.log(action[action_indices])
            log_prob = action.gather(1, action_indices)
            pg_loss = -log_prob * reward

            #loss_actor += 0.0000 * entropy
            #loss_actor.mean().backward()
            if entropy_penalty:
                new_loss = loss_actor + 0.01 * entropy
                new_loss.mean().backward()
            elif pg_penalty:
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

