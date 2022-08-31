# Adapted from https://github.com/blanyal/alpha-zero

import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from config import CFG


class ActorCritic(nn.Module):
    def __init__(self, game):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 1, stride=1)
        self.conv2 = nn.Conv2d(8, 16, 1, stride=1)
        self.conv3 = nn.Conv2d(16, 32, 1, stride=1)

        self.fc1 = nn.Linear(32 * 3 * 3, 120)

        self.actor = nn.Linear(120, game.action_size)  # number of actions
        self.critic = nn.Linear(120, 1) # linear output for value

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 32 * 3 * 3)

        x = F.relu(self.fc1(x))

        policy = self.actor(x)
        value = self.critic(x)

        return policy, value


class NeuralNetworkWrapper(object):
    def __init__(self, game):
        self.game = game
        self.model = ActorCritic(self.game)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)

        self.policy_criterion = nn.BCEWithLogitsLoss()
        self.value_criterion = nn.MSELoss()

    def predict(self, state):
        state = torch.tensor(state.reshape(1, 1, 3, 3)).float()
        pi, v = self.model(state)
        
        return F.softmax(pi[0]), v[0][0]

    def train(self, training_data):
        print("\nTraining the network.\n")
        for epoch in range(30):
            print("Epoch", epoch + 1)

            examples_num = len(training_data)
            running_loss = 0.0

            # Divide epoch into batches.
            loss = 0
            for i in range(0, examples_num, CFG.batch_size):
                states, pis, vs = map(list, zip(*training_data[i:i + CFG.batch_size]))

                # prepare the data
                _states = torch.tensor(states).view(-1, 1, 3, 3).float()
                _pis = torch.tensor([list(pi) for pi in pis])
                _vs = torch.tensor(vs).view(-1, 1).float()

                # forward + backward + optimize
                policy, value = self.model(_states)
                policy_loss = self.policy_criterion(policy, _pis)
                value_loss = self.value_criterion(value, _vs)
                loss += (policy_loss + value_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    def load_best_model(self):
        self.model.load_state_dict(torch.load("tic_tac_toe_best_model.pth"))
