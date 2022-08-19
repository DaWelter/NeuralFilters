from ast import YieldFrom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import dataclasses
from functools import partial
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from typing import Union, Optional, List, NamedTuple, Any, Tuple
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


class SequenceDataset(IterableDataset):
    def __init__(self):
        super().__init__()
        self.super_sequence_length = 4096
    def __iter__(self):
        while True:
            _, inputs, targets = data.make_synthetic_sequence(self.super_sequence_length)
            yield torch.from_numpy(inputs), torch.from_numpy(targets)

def iterate_batches(batch_size : int, sequence_length : Optional[int]):
    num_workers = 4
    loader = DataLoader(SequenceDataset(), batch_size=num_workers, num_workers=num_workers)
    inputlist = []
    targetlist = []
    for inputs, targets in loader:
        # num_workers x super sequence length -> flatten
        inputlist += list(inputs)
        targetlist += list(targets)
        # When enough samples are accumulated to fill a batch ...
        if len(inputlist)*inputs.size(1) > batch_size*(1 if sequence_length is None else sequence_length):
            inputs = torch.cat(inputlist, dim=0)
            targets = torch.cat(targetlist, dim=0)
            if sequence_length is None:
                inputs = torch.split(inputs, batch_size, dim=0)
                targets = torch.split(targets, batch_size, dim=0)
                yield from zip(inputs, targets)
            else:
                sz = batch_size*sequence_length
                # Just cut off the superfluous data
                inputs = util.split_and_stack(inputs[:sz,...], sequence_length)
                targets = util.split_and_stack(targets[:sz,...], sequence_length)
                yield inputs, targets
            inputlist = []
            targetlist = []


@dataclasses.dataclass
class TrainState():
    bar : tqdm.tqdm
    model_states : Any = None # state for sequence models


def compute_gradients_single_frame(model, inputs : Tensor, targets : Tensor, state : TrainState):
    device =  model.device
    #  Adds the channel dimension
    inputs = inputs.to(device)[:,None,:] 
    targets = targets.to(device)[:,None]
    pred = model(inputs)
    loss = F.mse_loss(pred.y, targets)        
    loss.backward()
    state.bar.set_description("Loss: %1.5f" % (loss.item()))


def compute_gradients_sequence_model(model, inputs : Tensor, targets : Tensor, state : TrainState, smoothness_weight, smoothness_func):
    device =  model.device
    L, B = inputs.shape[:2]
     # Add channel dimension
    inputs = inputs.to(device)[:,:,None,:]
    targets = targets.to(device)[:,:,None]

    # Or use the previous state?
    # It wouldn't be the one that came previously in the sequence.
    # Regardless it might be more representative of what the network inputs will be.
    state.model_states = model.create_initial_state(B)

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



def _train(model, batch_size, sequence_length, num_samples, learning_rate, gradient_udpate_func):
    batchsamples = batch_size * (sequence_length if sequence_length is not None else 1)
    num_batches_train = (num_samples + batchsamples - 1) // batchsamples

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=1, epochs=num_batches_train)

    model.train()

    bar = tqdm.trange(num_batches_train)

    train_state = TrainState(bar)

    for _, (inputs, targets) in (zip(bar, iterate_batches(batch_size, sequence_length))):
        optimizer.zero_grad()

        gradient_udpate_func(model, inputs, targets, train_state)

        optimizer.step()
        scheduler.step()


def train_sequence_model(model, batch_size, sequence_length, num_samples, learning_rate, smoothness_weight, smoothness_func):
    _train(
        model,
        batch_size, 
        sequence_length,
        num_samples, 
        learning_rate, 
        partial(compute_gradients_sequence_model, smoothness_weight=smoothness_weight, smoothness_func=smoothness_func))


def train_singleframe_model(model, batch_size, num_samples, learning_rate):
    _train(
        model,
        batch_size,
        None,
        num_samples, 
        learning_rate, 
        compute_gradients_single_frame)



def _test_smoothness():
    l = RobustSmooth2ndDerivLoss(eps=0.2)
    x = torch.linspace(0., 1., 10)[:,None,None].expand(-1,3,-1)
    val = l(x)
    assert torch.allclose(val, torch.tensor(0.))

    x[5,:,:] = 1.
    val = l(x)
    assert torch.all(val > torch.tensor(0.))


def _test_ds1():
    B, L = 16, 5
    for i, (inputs, targets) in enumerate(iterate_batches(B,L)):
        assert inputs.shape[:2] == (L,B) and len(inputs.shape)==3, f"Bad shape {inputs.shape}"
        assert targets.shape[:2] == (L,B) and len(targets.shape)==2, f"Bad shape {targets.shape}"
        if i>10:
            break

def _test_ds2():
    B, L = 16, None
    for i, (inputs, targets) in enumerate(iterate_batches(B,L)):
        assert inputs.shape[:1] == (B,) and len(inputs.shape)==2
        assert targets.shape[:1] == (B,) and len(targets.shape)==1
        if i>10:
            break



if __name__ == '__main__':
    _test_smoothness()
    _test_ds1()
    _test_ds2()