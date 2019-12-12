import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CBOW(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CBOW, self).__init__()
        self.l1 = nn.Linear(state_dim+action_dim, 500)
        self.l2 = nn.Linear(500, 200)
        self.l3 = nn.Linear(200, state_dim+action_dim)

    def forward(self, state, action):
        x = self.encode(state, action)
        x = torch.tanh(self.l3(x))
        return x

    def encode(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = torch.tanh(self.l1(state_action))
        x = torch.tanh(self.l2(x))
        return x

    def save(self, directory='./preTrained', name='CBOW'):
        torch.save(self.state_dict(), '{}/{}_actor.pth'.format(directory,name))

    def load(self, directory='./preTrained', name='CBOW'):
        self.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory,name), map_location=torch.device('cpu')))

