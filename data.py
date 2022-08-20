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

__all__ = [ 'make_synthetic_sequence' ]

def imageify(ys, size):
    # Makes a heatmap with the peak location corresponding to the supplied y coords.
    # Dimension: Sequence Length X 1d Size.
    x = np.linspace(-2., 2., size)
    z = np.exp(-np.power(ys[:,None]-x[None,:],2.)/0.25)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    return z


def sawtooth(n : int, y0 : float, period : float, start_increasing : bool):
    x = np.arange(n).astype(np.float64)
    if not start_increasing:
        y0 = -y0
    z = np.fmod(x + period * 2. + period * 0.5 + y0 * period * 0.5, period*2.) / period
    mask = z > 1.
    z[mask] = 2. - z[mask]
    z = 2.*z - 1.
    if not start_increasing:
        z = -z
    return z


def wave(n : int, y0 : float, period : float, start_increasing : bool):
    x = np.arange(n).astype(np.float64)
    if not start_increasing:
        y0 = -y0
    cs = np.cos(x  / period * np.pi * 2.)
    sn = np.sin(x / period * np.pi * 2.)
    z =  y0 * cs + np.sqrt(1. - y0**2) * sn
    if not start_increasing:
        z = -z
    return z


def constant(n : int, y0 : float):
    return np.full((n,), y0)


def added_corruption(x : np.ndarray, p : float, minsize=3):
    x = x.copy()
    npoints = int(p * len(x))
    indices = np.random.choice(len(x), size=npoints)
    ends = indices + np.random.randint(minsize, minsize*2, size=len(indices))
    ends = np.minimum(ends, len(x))
    for a,b in zip(indices, ends):
        x[a:b] -= 0.25
    return x


def make_synthetic_sequence(seq_len = 128, gen : Optional[np.random.Generator]=None):
    # Creates a batch. Consists of multiple sequences of random length.
    # Sequences are concatenated, so the tensors look like regular non-sequential batches.
    if gen is None:
        gen = np.random.default_rng()
    gts = []
    noises = []
    total_n = seq_len
    y0 = gen.uniform(-0.9,0.9)
    start_increasing = gen.integers(1)==0
    while total_n > 0:
        sel = gen.integers(3)
        n = gen.integers(16, 64)
        if sel == 0:
            y = sawtooth(n, y0, gen.uniform(16, 64), start_increasing)
            start_increasing = y[-1]>y[-2]
        elif sel == 1:
            y = wave(n, y0, gen.uniform(8, 64), start_increasing)
            start_increasing = y[-1]>y[-2]
        else:
            y = constant(n, y0)
            start_increasing = not start_increasing
        y0 = y[-1]
        gts.append(y)
        noises.append(0.02*gen.normal(0, 1., size=n))
        total_n -= n
    gt = np.concatenate(gts)[:seq_len]
    noises = np.concatenate(noises)[:seq_len]
    sos = butter(2, 1./(16.), fs=1., output='sos')
    gt = sosfilt(sos, gt)
    targets = added_corruption(gt, p=0.01)
    inputs = imageify(targets, 7) + gen.normal(0, 0.1, size=(len(gt), 7))
    targets = targets + noises
    gt = gt.astype(np.float32)
    inputs = inputs.astype(np.float32)
    targets = targets.astype(np.float32)
    return gt, inputs, targets


def plot_sequence(gt, inputs, targets):
    fig, axes = plt.subplots(2,1, figsize=(20,5))
    axes[0].set(title = 'Target Signals')
    axes[0].plot(gt, color='r', label='ground truth')
    axes[0].plot(targets, marker='x', lw=0, label='noisy labels', color='gray')
    axes[0].legend()
    axes[1].set(title = 'Input Image Sequence')
    axes[1].imshow(inputs.T)
    return fig


if __name__ == '__main__':
    gen = np.random.default_rng(123456)
    gt_example, inputs_example, targets_example = make_synthetic_sequence(1024, gen)