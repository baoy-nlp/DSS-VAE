import torch


def get_tensor(val):
    x = torch.Tensor(val)
    if torch.cuda.is_available():
        return x.cuda()
    return x


def get_long_tensor(val):
    return get_tensor(val).long()


def get_byte_tensor(val):
    return get_tensor(val).byte()


def get_float_tensor(val):
    return get_tensor(val).float()


def tile_batch(x, multiplier, batch_dim=0):
    x_size = x.size()
    out_size = x_size[:batch_dim] + (x_size[batch_dim] * multiplier,) + x_size[batch_dim + 1:]

    x_tiled = torch.unsqueeze(x, dim=batch_dim + 1)
    x_tiled = x_tiled.repeat(*[1 if d != batch_dim + 1 else multiplier for d in range(len(x_size) + 1)])
    x_tiled = x_tiled.view(*out_size)

    return x_tiled


def tensor_gather_helper(gather_indices,
                         gather_from,
                         batch_size,
                         beam_size,
                         gather_shape,
                         use_gpu=torch.cuda.is_available()):
    range_ = (torch.arange(0, batch_size) * beam_size).long()

    if use_gpu:
        range_ = range_.cuda()

    gather_indices_ = (gather_indices + torch.unsqueeze(range_, 1)).view(-1)

    output = torch.index_select(gather_from.view(*gather_shape), 0, gather_indices_)

    out_size = gather_from.size()[:1 + len(gather_shape)]

    return output.view(*out_size)


def inflate(tensor, times, dim):
    """
    Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times (in-place)

    Args:
        tensor: A :class:`Tensor` to inflate
        times: number of repetitions
        dim: axis for inflation (default=0)

    Returns:
        A :class:`Tensor`

    Examples::
        >> a = torch.LongTensor([[1, 2], [3, 4]])
        >> a
        1   2
        3   4
        [torch.LongTensor of size 2x2]
        >> b = ._inflate(a, 2, dim=1)
        >> b
        1   2   1   2
        3   4   3   4
        [torch.LongTensor of size 2x4]
        >> c = _inflate(a, 2, dim=0)
        >> c
        1   2
        3   4
        1   2
        3   4
        [torch.LongTensor of size 4x2]

    """
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)


def sequence_mask(sequence_length, max_len: int):
    batch_size = sequence_length.size(0)
    sequence_length = sequence_length.view(-1, 1)
    seq_range = get_tensor(range(0, max_len)).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = (sequence_length.expand_as(seq_range_expand))
    return seq_range_expand.le(seq_length_expand).long()


def batch_slice_set(input, dim, index, to_set):
    index = index.squeeze()
    _input = input * 1
    if dim == 1:
        _input[range(index.size()[0]), index, :] = to_set
    else:
        _input[range(index.size()[0]), :, index] = to_set
    return _input


def batch_slice_select(input, dim, index):
    """
    def batch_slice_select(input,dim,index):
        index = index.squeeze()
        if dim == 1:
            return input[range(index.size()[0]), index, :]
        else:
            return input[range(index.size()[0]), :, index]
    Returns a new tensor which indexes the input 3-D tensor along with the batch-dimension and the dim using index which is a LongTensor
        :param input: (Tensor) the input 3-D Tensor
        :param dim: (int) the dimension in which we index
        :param index: (LongTensor) the 1-D tensor containing the indices to index
        :return: torch.Tensor


    Examples::

        >>>import torch
        >>>x=torch.Tensor(range(36)).view(2,3,6)
        tensor([[[  0.,   1.,   2.,   3.,   4.,   5.],
                 [  6.,   7.,   8.,   9.,  10.,  11.],
                 [ 12.,  13.,  14.,  15.,  16.,  17.]],

                [[ 18.,  19.,  20.,  21.,  22.,  23.],
                 [ 24.,  25.,  26.,  27.,  28.,  29.],
                 [ 30.,  31.,  32.,  33.,  34.,  35.]]])
        >>>ind=torch.Tensor([1,0]).long()
        >>>batch_slice_select(input,0,ind)
        tensor([[  1.,   7.,  13.],
                [ 18.,  24.,  30.]])
        >>>batch_slice_select(input,1,ind)
        tensor([[  6.,   7.,   8.,   9.,  10.,  11.],
                [ 18.,  19.,  20.,  21.,  22.,  23.]])
    """
    index = index.squeeze()
    if dim == 1:
        return input[range(index.size()[0]), index, :]
    else:
        return input[range(index.size()[0]), :, index]


def batch_elements_select(inputs, index):
    index = index.squeeze()
    return inputs[range(index.size()[0]), index]


def shuffle_2d(inputs, dim=-1):
    """

    Args:
        inputs: 2-D Tensor, [batch_size, range_size]
        dim:

    Returns:

    """
    from .np_ops import batch_shuffle_indices
    if dim == 0:
        inputs = inputs.transpose(1, 0)
    batch_size, range_size = inputs.size()
    shuffle_ids = batch_shuffle_indices(batch_size, range_size)
    for b_ids in range(batch_size):
        inputs[b_ids] = inputs[b_ids][shuffle_ids[b_ids]]
    shuffle_ids = get_long_tensor(shuffle_ids)
    if dim == 0:
        inputs = inputs.transpose(1, 0), shuffle_ids.transpose(1, 0)
    return inputs, shuffle_ids


def zero_initialize(layers, batch_size, hidden_dims, rnn_cell='lstm'):
    initial_val = [0.0] * (layers * batch_size * hidden_dims)
    if rnn_cell == 'lstm':
        return [
            get_tensor(initial_val).view(layers, batch_size, hidden_dims),
            get_tensor(initial_val).view(layers, batch_size, hidden_dims),
        ]
    else:
        get_tensor(initial_val).view(layers, batch_size, hidden_dims)


def rnn_initialize(encoder_hidden, bidirectional_encoder):
    def _cat_directions(h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    if encoder_hidden is None:
        return None
    if isinstance(encoder_hidden, tuple):
        encoder_hidden = tuple([_cat_directions(h) for h in encoder_hidden])
    else:
        encoder_hidden = _cat_directions(encoder_hidden)
    return encoder_hidden


def find_val(inputs, val, axis=1):
    """
    Args:
        inputs: batch,max_len
        val: eos id
        axis: dim

    Return:
        is_find: byteTensor [batch,1]
        indices: longTensor [batch,1]
    """
    val_match = (inputs == val)
    return ((val_match.cumsum(axis) == 1) & val_match).max(axis)


def tensor_resort(inputs, tensor_order):
    """
    rearrange the last dim tensor with tensor_order.
    Args:
        inputs: 3-D Tensor <batch,seq_len,hidden>
        tensor_order: <batch,seq_len>

    Returns:

    """
    pass


def super_concatenate(tensor1, tensor2):
    """

    Args:
        tensor1: [batch_size,seq_len,hidden1]
        tensor2: [batch_size,seq_len,hidden2]

    Returns: [batch_size,seq_len,seq_len,hidden1+hidden2]

    """
    # batch_size,seq_len,hidden1 -> batch_size,seq_len,seq_len,hidden1
    seq_len = tensor1.size(1)
    new_tensor1 = tensor1.expand(seq_len, *tensor1.size()).transpose(1, 0).transpose(1, 2)
    new_tensor2 = tensor2.expand(seq_len, *tensor2.size()).transpose(1, 0)
    return torch.cat([new_tensor1, new_tensor2], dim=-1)


def self_concatenate(tensor1):
    return super_concatenate(tensor1, tensor1)


def self_add(tensor):
    seq_len = tensor.size(1)
    new_tensor1 = tensor.expand(seq_len, *tensor.size()).transpose(1, 0).transpose(1, 2)
    new_tensor2 = tensor.expand(seq_len, *tensor.size()).transpose(1, 0)
    return new_tensor1 + new_tensor2
