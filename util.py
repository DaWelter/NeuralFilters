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


def test_weighted_logsum_exp():
    logprobs = torch.tensor([1., 2., 3.])
    
    weightlogits = torch.tensor([-100., -100., 100.])
    logprob = weighted_logsumexp(logprobs, weightlogits, dim=-1)
    assert (torch.allclose(logprob, logprobs[2]))

    weightlogits = torch.tensor([ 100., -100., -100.])
    logprob = weighted_logsumexp(logprobs, weightlogits, dim=-1)
    assert (torch.allclose(logprob, logprobs[0]))

test_weighted_logsum_exp()