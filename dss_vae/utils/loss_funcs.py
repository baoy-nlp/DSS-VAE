import torch
import torch.nn.functional as torch_func


def _assert_no_grad(variable):
    assert not variable.requires_grad, "nn criterion don't compute the gradient w.r.t. targets - please " \
                                       "mark these variables as not requiring gradients"


def scale_rank_loss(inputs, target, mask, epsilon):
    """
    scaled, single-sided L1 loss epsilon: parameter for scaling

    """
    _assert_no_grad(target)

    diff = inputs[:, :, None] - inputs[:, None, :]
    target_diff_positive = ((target[:, :, None] - target[:, None, :]) > 0).float()
    target_diff_negative = - ((target[:, :, None] - target[:, None, :]) < 0).float()

    target_diff = target_diff_positive + target_diff_negative
    tgt_diff_zero = 1 - (target_diff_positive + (- target_diff_negative))

    mask = mask[:, :, None] * mask[:, None, :]

    exp_epsilon = torch.exp(epsilon)
    loss = torch_func.relu(
        exp_epsilon - target_diff * diff) + tgt_diff_zero * diff * diff / exp_epsilon ** 2 + 1 / exp_epsilon
    loss = (loss * mask).sum() / (mask.sum() + 1e-9)
    return loss


def rank_loss(inputs, target, mask, exp=False):
    if len(inputs.size()):
        inputs = inputs.squeeze()
    diff = inputs[:, :, None] - inputs[:, None, :]
    target_diff = ((target[:, :, None] - target[:, None, :]) > 0).float()
    mask = (mask[:, :, None] * mask[:, None, :]).float() * target_diff

    if exp:
        loss = torch.exp(torch_func.relu(target_diff - diff)) - 1
    else:
        loss = torch_func.relu(target_diff - diff)
    # [batch_size,seq_len,seq_len]
    loss = (loss * mask).sum() / (mask.sum() + 1e-9)

    return loss


mse = torch.nn.MSELoss(reduction='none')


def mse_loss(inputs, target, mask):
    mask = mask.float()
    loss = mse(inputs, target.float())  # [batch_size,seq_len]
    return (loss * mask).sum() / (mask.sum() + 1e-9)


def pos_loss(inputs, target, mask, exp=False, use_rank=True, use_mse=False, use_dst=False, with_clip=True):
    """

    Args:
        inputs: [batch_size, seq_len]
        target: [batch_size, seq_len]
        mask: [batch_size, seq_len]
        exp:
        use_rank:
        use_mse:
        use_dst:

    Returns:
        mse loss: mse(inputs,target) instance average with mask.sum(dim=-1)
        rank loss:

    """
    if len(inputs.size()):
        inputs = inputs.squeeze()
    _mse_mask = mask.float()  # [batch_size,seq_len]
    if use_mse:
        _mse_loss = (mse(inputs, target.float()) * _mse_mask).sum(dim=-1) / (_mse_mask.sum(dim=-1) + 1e-9)
    else:
        _mse_loss = 0.0

    diff = inputs.unsqueeze(-1) - inputs.unsqueeze(-2)
    target_diff = (target.unsqueeze(-1) - target.unsqueeze(-2)).float()
    _dst_mask = (_mse_mask.unsqueeze(-1) * _mse_mask.unsqueeze(-2))

    if use_rank:
        target_diff = (target_diff > 0).float()
        rank_mask = _dst_mask * target_diff
        if exp:
            _rank_loss = torch.exp(torch_func.relu(target_diff - diff)) - 1
        else:
            _rank_loss = torch_func.relu(target_diff - diff)
        _rank_loss = (_rank_loss * rank_mask).sum(dim=-1).sum(dim=-1) / (rank_mask.sum(dim=-1).sum(dim=-1) + 1e-9)
    else:
        _rank_loss = 0.0

    if use_dst:
        _dst_loss = mse(diff, target_diff) * _dst_mask
        _dst_loss = _dst_loss.sum(dim=-1).sum(dim=-1) / (_dst_mask.sum(dim=-1).sum(dim=-1) + 1e-9)
    else:
        _dst_loss = 0.0
    return _mse_loss + _rank_loss + _dst_loss


arc_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
tag_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
bce = torch.nn.BCELoss(reduction='none')


def label_loss(inputs, target, mask):
    loss = bce(inputs * mask, target * mask)
    return loss / (mask.sum() + 1e-9)


def bag_of_word_loss(input_log_softmax, input_var, criterion=None):
    """
    :param input_log_softmax: [batch_size,1/max_step,vocab_size]
    :param input_var: [batch_size,max_step]
    :param criterion
    :return:
    """
    seq_length = input_var.size(1)
    batch_size = input_var.size(0)
    vocab_size = input_log_softmax.size(-1)
    origin_score = input_log_softmax.squeeze(1).unsqueeze(1).expand((batch_size, seq_length, vocab_size))
    expand_log_score = origin_score.contiguous().view(-1, vocab_size)
    return criterion(expand_log_score, input_var.view(-1))


def cross_entropy_loss(scores, targets, criterion, pad=0):
    loss = criterion(scores, targets.view(-1))
    pred = scores.max(1)[1]
    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(pad).data).sum()
    num_total = targets.ne(pad).data.sum()
    loss = loss / num_total

    return loss, num_total, num_correct


def advance_loss(logits, tgt_var, criterion=None, log_prob=True, pad=-1):
    """
    Used for Training
    Args:
        logits: 'FloatTensor' [batch_size,seq_len,vocab_size]
        tgt_var: 'LongTensor' [batch_size,seq_len]
        criterion:  'nn.NLLLoss or nn.CrossEntropyLoss' orient to Input Tensor: [batch_size,logits] Target Tensor: [batch_size]
        log_prob: bool
        pad: default is -1

    Returns:
        Float Loss with [batch]
    """
    if criterion is None:
        criterion = torch.nn.NLLLoss(ignore_index=pad, reduction='none') if log_prob else torch.nn.CrossEntropyLoss(
            ignore_index=pad, reduction='none')

    batch_size, tgt_len = tgt_var.size()
    vocab_size = logits.size(-1)
    # origin_score = logits.squeeze(1)
    if logits.dim() < 3:
        # for bag-of-word loss
        origin_score = logits.unsqueeze(1).expand(batch_size, tgt_len, vocab_size)
    else:
        # for norm cross entropy loss
        logits_len = logits.size(1)
        common_len = min(logits_len, tgt_len)
        tgt_var = tgt_var.contiguous()[:, 0:common_len]
        origin_score = logits.contiguous()[:, 0:common_len, :]
    flatten_tgt = tgt_var.contiguous().view(-1)
    flatten_score = origin_score.contiguous().view(-1, vocab_size)  # [batch*common_len,vocab_size]
    _ret_loss = criterion(flatten_score, flatten_tgt)
    # [batch*seq_length]
    return _ret_loss.view(batch_size, -1).sum(dim=-1)


def mt_loss(logits, tgt_var, critic, pad=-1, log_prob=True, norm_by_word=True,
            reduce=False,
            normalization=1.0
            ):
    if not log_prob:
        logits = logits.log_softmax(dim=-1)
    if tgt_var.size(1) != logits.size(1):
        logits = logits[:, 0:tgt_var.size(1), :].contiguous()
    words_norm = tgt_var.ne(pad).float().sum(1)
    loss = critic(inputs=logits, labels=tgt_var, reduce=reduce, normalization=normalization)
    if norm_by_word:
        loss = loss.div(words_norm)
    return loss


def batch_bow_loss(logits, tgt_var, criterion=None, log_prob=True, pad=-1):
    """
    
    Args:
        logits: [batch_size,pred_len,vocab_size] 
        tgt_var: [batch_size, tgt_len]
        criterion:
        log_prob:
        pad: 

    Returns:

    """

    def batch_sum(log_logits, _tgt_var, _pad):
        batch_size, pred_len, vocab_size = log_logits.size()
        log_logits = log_logits.contiguous().view(-1, vocab_size)
        _tgt_var = _tgt_var.unsqueeze(1).expand(batch_size, pred_len, -1).contiguous().view(batch_size * pred_len, -1)
        _tgt_var_mask = (1.0 - _tgt_var.eq(_pad).float())
        _score = (torch.gather(log_logits, dim=1, index=_tgt_var) * _tgt_var_mask).sum(dim=-1)
        return _score.contiguous().view(batch_size, pred_len).sum(dim=-1)

    if not log_prob:
        logits = logits.log_softmax(dim=-1)

    # if criterion is None:
    #     criterion = torch.nn.NLLLoss(ignore_index=pad, reduction='none')

    # criterion = torch.nn.CrossEntropyLoss(ignore_index=pad, reduction='none')
    # batch_size, tgt_len = tgt_var.size()
    # _, pred_len, vocab_size = logits.size()
    #
    # logits = logits.contiguous().view(-1, vocab_size)
    # origin_score = logits.unsqueeze(1).expand(-1, tgt_len, vocab_size)
    # flatten_score = origin_score.contiguous().view(-1, vocab_size)  # [batch*pred_len*tgt_len,vocab_size]
    #
    # flatten_tgt = tgt_var.unsqueeze(1).expand(-1, pred_len, tgt_len).contiguous().view(-1)
    #
    # _ret_loss = criterion(flatten_score, flatten_tgt)
    # [batch*seq_length]
    # return _ret_loss.view(batch_size, -1).sum(dim=-1)
    return -batch_sum(log_logits=logits, _tgt_var=tgt_var, _pad=pad)
