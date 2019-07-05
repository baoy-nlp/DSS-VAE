import torch
import torch.nn as nn
from torch.autograd import Variable

from dss_vae.utils.tensor_ops import get_float_tensor, get_long_tensor


def positional_matrix(seq_len, hidden, use_cuda=True):
    positions = torch.arange(0, seq_len)
    if use_cuda:
        positions = positions.cuda()
    positions = Variable(positions.float())
    channels = torch.arange(0, hidden, 2).float() / hidden
    if use_cuda:
        channels = channels.cuda()
    channels = 1 / (10000 ** Variable(channels).float())
    encodings = positions.unsqueeze(-1) @ channels.unsqueeze(0)  # batch x target_len x 256
    encodings = torch.cat([torch.sin(encodings).unsqueeze(-1), torch.cos(encodings).unsqueeze(-1)], -1)
    encodings = encodings.contiguous().view(*encodings.size()[:-2], -1)
    return encodings


def init_position_embedding(max_len, hidden_size, is_matrix=False):
    embed_matrix = positional_matrix(seq_len=max_len, hidden=hidden_size)
    if is_matrix:
        embed_matrix.requires_grad = False
        return embed_matrix
    embed_weight = nn.Parameter(embed_matrix)
    embed_weight.requires_grad = False
    embed_table = nn.Embedding(
        max_len,
        hidden_size,
        _weight=embed_weight
    )
    return embed_table


def to_index(st_gumbel_out, **kwargs):
    """

    Args:
        st_gumbel_out: [batch_size,seq_len,seq_len] or [batch_size,seq_len]
        **kwargs:

    Returns: index select [batch_size,seq_len]

    """
    n_class = st_gumbel_out.size(-1)
    label_vec = get_float_tensor(val=range(n_class)).view(-1, 1)
    ret = torch.matmul(st_gumbel_out, label_vec)
    if ret.dim() > 2:
        ret = ret.squeeze(-1)
    return ret.long()


def sequential_index(batch_size, n_class):
    long_tensor = get_long_tensor(val=range(n_class)).view(-1)
    return long_tensor.expand(batch_size, *long_tensor.size())


def get_position_encoding_expectation(pos_prob, embed_matrix):
    """

    Args:
        pos_prob: (FloatTensor) [batch_size,seq_len,max_len]
        embed_matrix: (FloatTensor) [max_len.hidden_size] w/o requires_grad

    Returns: expectation position encoding

    """
    return torch.matmul(pos_prob, embed_matrix)


def positional_encodings_like(x, t=None, use_cuda=True):  # hope to be differentiable
    """
    Args:
        x: [batch_size,length,hidden]
        t:
        use_cuda:
    """
    # if batch_first:
    #     to_e = x.contiguous().transpose(1, 0).contiguous()
    to_e = x
    if t is None:
        positions = torch.arange(0, to_e.size(-2))  # .expand(*x.size()[:2])
        if use_cuda:
            positions = positions.cuda()
        positions = Variable(positions.float())
    else:
        positions = t

    # channels
    channels = torch.arange(0, to_e.size(-1), 2).float() / to_e.size(-1)  # 0 2 4 6 ... (256)
    if use_cuda:
        channels = channels.cuda()
    channels = 1 / (10000 ** Variable(channels).float())

    # get the positional encoding: batch x target_len
    encodings = positions.unsqueeze(-1) @ channels.unsqueeze(0)  # batch x target_len x 256
    encodings = torch.cat([torch.sin(encodings).unsqueeze(-1), torch.cos(encodings).unsqueeze(-1)], -1)
    encodings = encodings.contiguous().view(*encodings.size()[:-2], -1)  # batch x target_len x 512

    if encodings.ndimension() == 2:
        encodings = encodings.unsqueeze(0).expand_as(to_e)

    # if batch_first:
    #     encodings = encodings.contiguous().transpose(1, 0).contiguous()

    return encodings
