#!/usr/bin/env python
# coding: utf-8

### Import libraries
import math
import os
import sys
import time

import numpy as np
import gym
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from torch.autograd import Variable


# ### Global variables
MODEL_PATH = 'breakout_a3c.pth'
ENV_NAME = os.getenv('ENV_NAME', 'BreakoutDeterministic-v4')


SEED = 22
NUM_PROCESSES = 4
EPISODES_TRAINING = 5000
EPISODES_TESTING = 5000
HEIGHT = 84
WIDTH = 84
N_FRAMES = 4
VALUE_LOSS_COEF = 0.5
GAMMA = 0.99

TRAINING_PARAMETERS = {
    'frames': N_FRAMES,
    'trajectory_steps': 64,
    'num_processes': NUM_PROCESSES
}


# ### Define model architecture
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc1 = nn.Linear(7 * 7 * 64, 512)

        self.actor = nn.Linear(512, 4)  # number of actions
        self.critic = nn.Linear(512, 1) # linear output for value

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 7 * 7 * 64)

        x = F.relu(self.fc1(x))

        policy = self.actor(x)
        value = self.critic(x)

        return policy, value


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def get_action_vector(action):
    return action


def rgb2gray_and_resize(screen):
    screen_grey = np.array(np.dot(screen[..., :3], [0.299, 0.587, 0.114]), dtype=np.uint8)
    img_from_arr = Image.fromarray(screen_grey)
    img_from_arr = img_from_arr.resize((84, 84))

    return np.array(img_from_arr)


def update_frame_sequence(state, obs, n_frames=4, width=84, height=84):
    obs = np.ascontiguousarray(rgb2gray_and_resize(obs), dtype=np.float32) / 255
    obs = torch.FloatTensor(obs)

    if state is None:
        _state = obs.repeat(n_frames, 1).view(n_frames, width, height)
    else:
        _state = state.view(n_frames, width, height)
        _state = torch.cat((_state[1:], obs.view((1, width, height))))

    return _state


def calculate_reward(reward):
    reward = np.clip(1, -1, reward)

    return reward


def train(rank, episodes, training_params, shared_model, counter, lock, optimizer=None, models_path='models'):
    torch.manual_seed(SEED + rank)

    env = gym.make(ENV_NAME)
    model = ActorCritic()
    model.train()

    optimizer = optim.Adam(shared_model.parameters(), lr=0.00025)

    env.reset()
    obs = env.render(mode='rgb_array')
    state = None
    state = update_frame_sequence(state, obs, n_frames=training_params['frames'])

    for _ in range(int(episodes)):
        done = False

        while not done:
            values = []
            log_probs = []
            rewards = []
            # Sync with the shared model
            model.load_state_dict(shared_model.state_dict())

            for step in range(training_params['trajectory_steps']):
                logits, value = model(state.unsqueeze(0))

                prob = F.softmax(logits, -1)
                action = prob.multinomial(num_samples=1)
                
                log_prob = F.log_softmax(logits, -1)
                log_prob = log_prob.gather(1, Variable(action))
                
                obs, reward, done, info = env.step(get_action_vector(action.item()))
                reward = calculate_reward(reward)

                with lock:
                    counter.value += 1

                state = update_frame_sequence(state, obs, n_frames=training_params['frames'])
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    env.reset()
                    obs = env.render(mode='rgb_array')
                    state = None
                    state = update_frame_sequence(state, obs, n_frames=training_params['frames'])
                    break

            ###################################
            ### Prepare for update the policy
            ###################################

            R = torch.zeros(1, 1)
            if not done:
                _, value = model(state.unsqueeze(0))
                R = value.data

            values.append(Variable(R))
            policy_loss = 0
            value_loss = 0
            R = Variable(R)
            for i in reversed(range(len(rewards))):
                R = GAMMA * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)
                policy_loss = policy_loss - (log_probs[i] * Variable(advantage))

            optimizer.zero_grad()
            loss_fn = (policy_loss + VALUE_LOSS_COEF * value_loss)
            loss_fn.backward(retain_graph=True)
            ensure_shared_grads(model, shared_model)
            optimizer.step()
            torch.save(shared_model.state_dict(), MODEL_PATH)


def test(rank, episodes, training_params, shared_model, counter, render=True, models_path='models'):
    torch.manual_seed(SEED + rank)

    env = gym.make(ENV_NAME)
    model = ActorCritic()
    model.eval()

    env.reset()
    obs = env.render(mode='rgb_array')
    state = None
    state = update_frame_sequence(state, obs, n_frames=training_params['frames'])

    rewards = []
    reward_sum = 0
    value_acum = []
    value_avg_best = -1000.0

    start_time = time.time()
    training_time = 0
    episode = 0
    episode_steps = 0

    for _ in range(int(episodes)):
        model.load_state_dict(shared_model.state_dict())

        done = False
        while not done:
            logits, value = model(state.unsqueeze(0))
            prob = F.softmax(logits, -1)
            action = prob.multinomial(num_samples=1)

            obs, reward, done, info = env.step(get_action_vector(action.item()))
            env.render()
            reward = calculate_reward(reward)

            episode_steps += 1
            rewards.append(reward)
            reward_sum += reward
            value_acum.append(value.data[0, 0])

            state = update_frame_sequence(state, obs, n_frames=training_params['frames'])

            if done:
                episode += 1
                episode_time = time.time() - start_time
                training_time += episode_time
                value_avg = np.mean(value_acum)

                if value_avg > value_avg_best:
                    torch.save(model.state_dict(), 'breakout_a3c_best.pth')
                    value_avg_best = value_avg

                dict_info = {}
                dict_info["episode"] = episode
                dict_info["episode_time_secs"] = (str(episode_time))
                dict_info["episode_steps"] = (str(episode_steps))
                dict_info["episode_reward"] = reward_sum
                dict_info["reward_min"] = np.min(rewards)
                dict_info["reward_max"] = np.max(rewards)
                dict_info["reward_avg_by_step"] = np.mean(rewards)
                dict_info["value_avg"] = value_avg
                dict_info["training_time"] = training_time
                dict_info["training_steps"] = (str(counter.value))

                print(dict_info)

                rewards = []
                reward_sum = 0
                episode_steps = 0
                start_time = time.time()

                env.reset()
                obs = env.render(mode='rgb_array')
                state = None
                state = update_frame_sequence(state, obs, n_frames=training_params['frames'])

                break


if __name__ == '__main__':
    torch.manual_seed(22)

    shared_model = ActorCritic()
    shared_model.share_memory()

    num_processes = int(TRAINING_PARAMETERS['num_processes'])
    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    print("Launching testing process")
    p = mp.Process(
        target=test,
        args=(num_processes, EPISODES_TESTING, TRAINING_PARAMETERS,
            shared_model, counter, lock, None))
    p.start()
    processes.append(p)

    print("Launching {} training processes".format(NUM_PROCESSES - 1))
    for rank in range(0, num_processes - 1):
        p = mp.Process(
            target=train,
            args=(rank, EPISODES_TRAINING, TRAINING_PARAMETERS, shared_model, counter, lock, None, None))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
