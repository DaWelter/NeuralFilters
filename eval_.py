import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import tqdm
import copy
from functools import partial
import pandas as pd
from scipy.signal import butter, sosfilt
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from typing import Union, Optional, List
from torch import Tensor

import data
import util

def make_testset():
    gen = np.random.default_rng(123456+5)
    gt, inputs, targets = data.make_synthetic_sequence(1024, gen)
    return [ (gt, inputs, targets) ]


def callmodel(model, x : Tensor):
    assert len(x.shape) == 2 # Length x Image Width
    if hasattr(model, 'create_initial_state'):
        # Sequencemodel
        inputs_tensor = util.to_tensor(x, model.device)[:,None, None,:] # Add batch and channel dimensions
        pred, _ = model(inputs_tensor, model.create_initial_state(1))
    else:
        # Single frame model
        inputs_tensor = util.to_tensor(x, model.device)[:,None,:] # Add channel dimension
        pred = model(inputs_tensor)
    mean = pred.y.detach().cpu().view(-1).numpy()
    std = np.sqrt(pred.r.detach().cpu().view(-1).numpy()) if pred.r is not None else None
    return mean, std


def evaluate(model, testset):
    model.eval()
    
    def plot_samples_err_bounds(mean, std):
        plt.fill_between(np.arange(mean.shape[0]), mean-std, mean+std, color='b', alpha=0.25)

    for gt, inputs, targets in testset:
        mean, std = callmodel(model, inputs)
        _ = plt.figure(figsize=(35,5))
        plt.plot(gt, color='r')
        plt.plot(mean, color='b', label='pred')
        if std is not None:
            plot_samples_err_bounds(mean, std)
        plt.legend()
        plt.show()