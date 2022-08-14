import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import dataclasses
from functools import partial
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from typing import Union, Optional, List, NamedTuple, Any
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

#nll_loss = torch.nn.GaussianNLLLoss(reduction='mean')

import util
import data



class SecondDerivative(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_dxx = nn.Conv1d(1,1,3,bias=False)
        self.conv_dxx.weight.data[...] = torch.tensor([1.,-2., 1.])
        self.conv_dxx.weight.requires_grad = False

    def forward(self, x : Tensor):
        '''
        x must have shape (L x B x C)
        '''
        L, B, C = x.shape
        # Switch to B x C x L
        # Apply conv, and back to L B C
        return self.conv_dxx(x.permute(1,2,0)).permute(2,0,1)


class Smooth2ndDerivLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dxx_op = SecondDerivative()
    def forward(self, x : Tensor):
        vals = self.dxx_op(x).square()
        return vals.mean()


class RobustSmooth2ndDerivLoss(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        self.dxx_op = SecondDerivative()
    def forward(self, x : Tensor):
        vals = F.relu(self.dxx_op(x).abs() - self.eps).square()
        return vals.mean()


def _test_smoothness():
    l = RobustSmooth2ndDerivLoss(eps=0.2)
    x = torch.linspace(0., 1., 10)[:,None,None].expand(-1,3,-1)
    val = l(x)
    assert torch.allclose(val, torch.tensor(0.))

    x[5,:,:] = 1.
    val = l(x)
    assert torch.all(val > torch.tensor(0.))




class SequenceDataset(IterableDataset):
    def __init__(self, seqsize):
        super().__init__()
        self.seqsize = seqsize
    def __iter__(self):
        while True:
            _, inputs, targets = data.make_synthetic_sequence(self.seqsize)
            yield inputs, targets


@dataclasses.dataclass
class TrainState():
    bar : tqdm.tqdm
    model_states : Any = None # state for sequence models


def compute_gradients_single_frame(model, inputs, targets, state : TrainState):
    device =  model.device
    inputs = util.to_tensor(inputs, device)[:,None,:]
    targets = util.to_tensor(targets, device)[:,None]
    pred = model(inputs)
    loss = F.mse_loss(pred.y, targets)        
    loss.backward()
    state.bar.set_description("Loss: %1.5f" % (loss.item()))


def compute_gradients_sequence_model(model, inputs, targets, state : TrainState, smoothness_weight, smoothness_func):
    device =  model.device
    seqsize = 32
    batchsize = inputs.shape[0] // seqsize
    assert inputs.shape[0] == batchsize*seqsize
    inputs = util.split_and_stack(util.to_tensor(inputs, device)[:,None,:], seqsize)
    targets = util.split_and_stack(util.to_tensor(targets, device)[:,None], seqsize)

    if state.model_states is None:
        state.model_states = model.create_initial_state(batchsize)

    measurement_pred, state_pred = model(inputs, state.model_states)
    
    pred_y = measurement_pred.y
    pred_var = measurement_pred.r
    
    assert pred_y.shape == targets.shape, f"Bad shapes {pred_y.shape} vs {targets.shape}"

    mseloss = F.mse_loss(pred_y, targets)
    loss = mseloss.clone()

    if smoothness_weight is not None:
        smoothness_value = smoothness_func(pred_y)
        loss += smoothness_weight * smoothness_value
    else:
        smoothness_value = torch.tensor(torch.nan)
    
    loss.backward()
    
    state.bar.set_description("Loss: %1.5f, Smoothness %1.5f" % (mseloss.item(), smoothness_value.item()))



def _train(model, num_samples, learning_rate, gradient_udpate_func):
    num_workers = 4
    full_length_sequence = 1024*16
    num_batches_train = num_samples // (num_workers * full_length_sequence)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=1, epochs=num_batches_train)
    
    # The loader is tricked into created a batch with as many elements as there are worker processes. This is so that they
    # can work nicely independently. But they return a sequence which is much longer than the sequence I want to train on.
    # Therefore the batch is later cut and rearranged so I have pieces of 'seqsize' and `batchsize` (the true training batch size)
    loader = DataLoader(SequenceDataset(full_length_sequence), batch_size=num_workers, num_workers=num_workers)

    model.train()

    bar = tqdm.trange(num_batches_train)

    train_state = TrainState(bar)

    for epoch, batch in (zip(bar, loader)):
        inputs, targets = batch
        inputs = np.concatenate(tuple(inputs), axis=0)
        targets = np.concatenate(tuple(targets), axis=0)

        optimizer.zero_grad()

        gradient_udpate_func(model, inputs, targets, train_state)

        optimizer.step()
        scheduler.step()


def train_sequence_model(model, num_samples, learning_rate, smoothness_weight, smoothness_func):
    _train(
        model, 
        num_samples, 
        learning_rate, 
        partial(compute_gradients_sequence_model, smoothness_weight=smoothness_weight, smoothness_func=smoothness_func))


def train_singleframe_model(model, num_samples, learning_rate):
    _train(
        model, 
        num_samples, 
        learning_rate, 
        compute_gradients_single_frame)



if __name__ == '__main__':
    _test_smoothness()