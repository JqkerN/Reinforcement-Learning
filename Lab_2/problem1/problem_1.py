import numpy as np
import gym
import torch
from tqdm import trange


# Load model
try:
    model = torch.load('neural-network-avg_141.pth')
    print('Network model: {}'.format(model))
except:
    print('File neural-network-1.pth not found!')
    exit(-1)