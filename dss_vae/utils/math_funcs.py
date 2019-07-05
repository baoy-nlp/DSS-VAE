from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def kl_divergence(dis1, dis2):
    val = F.kl_div(dis1.log(), dis2, reduction='none')
    return val


def js_divergence(dis1, dis2):
    sym_dis = (dis1 + dis2) / 2
    return 0.5 * kl_divergence(dis1, sym_dis) + 0.5 * kl_divergence(dis2, sym_dis)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


if __name__ == "__main__":
    # p1 = torch.Tensor([0.1, 0.2, 0.1, 0.2, 0.4])
    # p2 = torch.Tensor([0.1, 0.2, 0.2, 0.3, 0.3])
    # print(kl_divergence(p1, p2))
    # print(js_divergence(p1, p2))
    import math

    print(gumbel_softmax(
        Variable(torch.cuda.FloatTensor([[math.log(0.1), math.log(0.4), math.log(0.3), math.log(0.2)]] * 20000)),
        0.8).sum(dim=0))
