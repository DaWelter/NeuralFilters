from typing import Union
import torch
from torch.nn.utils.rnn import PackedSequence
from torch import Tensor


def to_tensor(x, device = 'cpu'):
    return torch.from_numpy(x).to(torch.float32).to(device)


def split_and_stack(x, splitsize):
    return torch.stack(torch.split(x, splitsize), dim=1)


def seq_as_batch(x : Union[Tensor,PackedSequence]):
    if isinstance(x, PackedSequence):
        return x.data
    else:
        return x.view(-1, *x.shape[2:])

def batch_as_seq_like(x : Tensor, z : Union[Tensor,PackedSequence]):
    if isinstance(z, PackedSequence):
        return torch.nn.utils.rnn.PackedSequence(
            x,
            z.batch_sizes,
            z.sorted_indices,
            z.unsorted_indices
        )
    else:
        return x.view(*z.shape[:2],*x.shape[1:])


def weighted_logsumexp(logprobs, weightlogits, dim):
    '''
    Computes
    log( sum_i w_i exp(logprobs_i) ), where
    w_i = z_i / (sum_j z_j), and
    z_i = exp(weightlogits_i)

    Numerically stable alternative to mixing probabilities like
    sum_i (w_i pi),
    w_i = softmax(weightlogits)_i
    due to working with log probabilities.
    '''
    normalizer = torch.logsumexp(weightlogits, dim=dim, keepdim=True)
    weighted_logits = weightlogits - normalizer + logprobs
    return torch.logsumexp(weighted_logits, dim=dim)



def gather_particles(x : Tensor, indices):
    # For resampling
    assert indices.size(1) == x.size(1), f"bad shape {indices.shape} vs {x.shape}"
    P, B = x.shape[:2]
    xview = x.view(P, B, -1)
    indices = indices[:,:,None].expand(-1, -1, xview.size(-1))
    out = torch.gather(xview, 0, indices)
    return out.view(*indices.shape[:2],*x.shape[2:])


def test_weighted_logsum_exp():
    logprobs = torch.tensor([1., 2., 3.])
    
    weightlogits = torch.tensor([-100., -100., 100.])
    logprob = weighted_logsumexp(logprobs, weightlogits, dim=-1)
    assert (torch.allclose(logprob, logprobs[2]))

    weightlogits = torch.tensor([ 100., -100., -100.])
    logprob = weighted_logsumexp(logprobs, weightlogits, dim=-1)
    assert (torch.allclose(logprob, logprobs[0]))


def test_gather_particles1():
    P, B = 5, 3
    x = torch.arange(P*B).view(P,B)
    #print (x)
    # Two particles per batch as result. Different ones for each batch.
    i = torch.tensor([
        [ 1, 2, 2 ],
        [ 3, 3, 4 ]
    ])
    y = gather_particles(x, i)
    #print (y)
    yexpect = torch.tensor([
        [ 3, 7,  8 ],
        [ 9, 10, 14 ]
    ])
    #print (yexpect)
    assert torch.all(y == yexpect)

def test_gather_particles2():
    # Again with bigger tensors
    P, B, C = 5, 3, 3
    x = torch.rand(P,B,C)
    # Two particles per batch as result. Different ones for each batch.
    i = torch.tensor([
        [ 1, 2, 2 ],
        [ 3, 3, 4 ]
    ])
    y = gather_particles(x, i)
    z = x.view(-1,C)
    yexpect = torch.stack([
        torch.stack([ z[3], z[7],  z[8] ]),
        torch.stack([ z[9], z[10], z[14] ])
    ])
    assert torch.all(y == yexpect)

test_weighted_logsum_exp()
test_gather_particles1()
test_gather_particles2()


