import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

DEVICE = 'cpu'

#Deep Q Network (DQN) estimates future rewards of possible actions given state
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        #Convolutional layers
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        #witdh of the final convolutional layer
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        #height of the final convolutional layer
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32

        #Final layer (linear)
        self.head = nn.Linear(linear_input_size, outputs)


    def forward(self, x): #forward pass
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)
 

class Agent():
    def __init__(self, strategy, num_actions = 4):
        self.current_step = 0
        self.strategy = strategy #Epsilon greedy strategy in this case
        self.num_actions = num_actions


    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step).to(DEVICE)
        self.current_step += 1

        if rate > random.random():
            return random.randrange(self.num_actions) # explore      
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).item().to(DEVICE) # exploit  