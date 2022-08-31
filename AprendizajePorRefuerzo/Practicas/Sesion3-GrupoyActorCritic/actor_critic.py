#!/usr/bin/env python
# coding: utf-8

### Import libraries
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable


# ### Global variables
MODEL_PATH = 'cartpole_actor_critic.pth'
ENV_NAME = 'CartPole-v0'
SEED = 22
EPISODES = 5000
STEPS = 64
VALUE_LOSS_COEF = 0.5
GAMMA = 0.99


# ### Define model architecture
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 36)
        
        self.actor = nn.Linear(36, 2)
        self.critic = nn.Linear(36, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy = self.actor(x)
        value = self.critic(x)

        return policy, value


def get_action_vector(action):
    return action


def calculate_reward(reward):
    reward = np.clip(1, -1, reward)

    return reward


def train():
    env = gym.make(ENV_NAME)
    model = ActorCritic()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    obs = env.reset()
    state = torch.from_numpy(obs).float()

    total_values = []
    
    for _ in range(int(EPISODES)):
        done = False
        values = []
        log_probs = []
        rewards = []

        while not done:
            logits, value = model(state)

            prob = F.softmax(logits, -1)
            action = prob.multinomial(num_samples=1)
            log_prob = F.log_softmax(logits, -1)
            log_prob = log_prob.gather(0, action)

            obs, reward, done, info = env.step(get_action_vector(action.item()))
            reward = calculate_reward(reward)

            if done:
                obs = env.reset()

            state = torch.from_numpy(obs).float()

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            total_values.append(value.item())

            if done:
                break

        print(np.sum(rewards), np.mean(total_values))

        ###################################
        ### Prepare for update the policy
        ###################################

        R = 0
        if not done:
            _, value = model(state)
            R = value.data

        values.append(R)
        policy_loss = 0
        value_loss = 0
        for i in reversed(range(len(rewards))):
            R = GAMMA * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            policy_loss = policy_loss - (log_probs[i] * Variable(advantage))

        optimizer.zero_grad()
        loss_fn = (policy_loss + VALUE_LOSS_COEF * value_loss)
        loss_fn.backward(retain_graph=True)
        optimizer.step()
        torch.save(model.state_dict(), MODEL_PATH)


def test():
    env = gym.make(ENV_NAME)
    model = ActorCritic()
    model.eval()
    model.load_state_dict(torch.load(MODEL_PATH))

    obs = env.reset()
    state = torch.from_numpy(obs).float()
    
    while True:
        done = False
        total_values = []
        total_rewards = []

        while not done:
            logits, value = model(state)

            prob = F.softmax(logits, -1)
            action = prob.argmax()

            obs, reward, done, info = env.step(get_action_vector(action.item()))
            env.render()
            reward = calculate_reward(reward)

            if done:
                obs = env.reset()

            state = torch.from_numpy(obs).float()

            total_rewards.append(reward)
            total_values.append(value.item())

            if done:
                break

        print(np.sum(total_rewards), np.mean(total_values))


if __name__ == '__main__':
    torch.manual_seed(SEED)

    train()