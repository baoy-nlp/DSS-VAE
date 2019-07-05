# coding=utf-8

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

TINY = 1e-9

_FLOAT32_INF = np.float32(np.finfo('float32').max / 10)


def unk_replace(input_sequence, dropoutr, vocab):
    if dropoutr > 0.:
        prob = torch.rand(input_sequence.size())
        if torch.cuda.is_available():
            prob = prob.cuda()
        prob[(input_sequence.data - vocab.sos_id) * (input_sequence.data - vocab.pad_id) * (
                input_sequence.data - vocab.eos_id) == 0] = 1
        decoder_input_sequence = input_sequence.clone()
        decoder_input_sequence[prob < dropoutr] = vocab.unk_id
        return decoder_input_sequence
    return input_sequence


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == "fixed":
        return 1.0
    elif anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'sigmoid':
        return float(1 / (1 + np.exp(0.001 * (x0 - step))))
    elif anneal_function == 'negative-sigmoid':
        return float(1 / (1 + np.exp(-0.001 * (x0 - step))))
    elif anneal_function == 'linear':
        return min(1, step / x0)


def wd_anneal_function(unk_max, anneal_function, step, k, x0):
    return unk_max * kl_anneal_function(anneal_function, step, k, x0)


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def id2word(sents, vocab):
    if type(sents[0]) == list:
        return [robust_id2word(s, vocab) for s in sents]
    else:
        return robust_id2word(sents, vocab)


def robust_id2word(sents, vocab):
    res = []
    for w in sents:
        if w == vocab.sos_id or w == vocab.pad_id:
            pass
        elif w == vocab.eos_id:
            break
        else:
            res.append(vocab.id2word[w])
    return res


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def isnan(data):
    data = data.cpu().numpy()
    return np.isnan(data).any() or np.isinf(data).any()


def mask_scores(scores, beam_mask, EOS):
    """
    Mask scores of next step according to beam mask.
    Args:
        scores (torch.Tensor): Scores of next tokens with shape [batch_size, beam_size, vocab_size].
            Smaller should be better (usually negative log-probs).
        beam_mask (torch.Tensor): Mask of beam. 1.0 means not closed and vice verse. The shape is
            [batch_size, beam_size]

    Returns:
        Masked scores of next tokens.
    """
    vocab_size = scores.size(-1)

    finished_row = beam_mask.new(vocab_size, ).zero_() + float(_FLOAT32_INF)

    # If beam finished, only PAD could be generated afterwards.
    finished_row[EOS] = 0.0

    scores = scores * beam_mask.unsqueeze(2) + torch.matmul((1.0 - beam_mask).unsqueeze(2), finished_row.unsqueeze(0))

    return scores


def gumbel_softmax(inputs, beta=0.5, tau=1.0):
    noise = inputs.data.new(*inputs.size()).uniform_()
    noise.add_(TINY).log_().neg_().add_(TINY).log_().neg_()
    return F.softmax((inputs + beta * Variable(noise)) / tau, dim=-1)


def input_padding(sents, pad_token, max_len=-1):
    if max_len == -1:
        max_len = max(len(s) for s in sents)
    batch_size = len(sents)
    seqs_t = []
    masks = []
    for i in range(max_len):
        seqs_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in range(batch_size)])
    return seqs_t, masks


def padding_input(sents, pad_token, max_len=-1):
    if max_len == -1:
        max_len = max(len(s) for s in sents)
    batch_size = len(sents)
    seqs = []
    for i in range(batch_size):
        seqs.append(sents[i] + [pad_token] * (max_len - len(sents[i])))
    return seqs


def input_scaling(sent, max_tgt_len=-1, extra_token="</s>", scale=0.0):
    if scale < 0:
        return sent
    else:
        pad_len = int(len(sent) * scale)
        if max_tgt_len != -1:
            cur_diff = max_tgt_len - len(sent)
            pad_len = pad_len if pad_len < cur_diff else cur_diff
        return sent + [extra_token] * pad_len


def to_input_variable(sequences, vocab, max_len=-1, cuda=False, training=True, append_boundary_sym=False,
                      batch_first=False, shuffle=False):
    """
    given a list of sequences,
    return a tensor of shape (max_sent_len, batch_size)
    """
    from .tensor_ops import get_long_tensor
    if not isinstance(sequences[0], list):
        sequences = [sequences]
    if append_boundary_sym:
        sequences = [['<s>'] + seq + ['</s>'] for seq in sequences]
    pad_sents = padding_input(sequences, pad_token="<pad>", max_len=max_len)
    seqs = word2id(pad_sents, vocab)
    # seqs_t, masks = input_padding(word_ids, vocab['<pad>'], max_len)
    if not training:
        with torch.no_grad():
            seqs_var = Variable(get_long_tensor(seqs), requires_grad=False)
    else:
        seqs_var = Variable(get_long_tensor(seqs), requires_grad=False)
    # if cuda:
    #     seqs_var = seqs_var.cuda()
    if not batch_first:
        seqs_var = seqs_var.transpose(1, 0).contiguous()
        shuffle_dim = -1
    else:
        shuffle_dim = 0

    if shuffle:
        from .tensor_ops import shuffle_2d
        return shuffle_2d(inputs=seqs_var, dim=shuffle_dim)

    return seqs_var


def to_input_dict(examples,
                  vocab,
                  max_tgt_len=-1,
                  cuda=False,
                  training=True,
                  src_append=True,
                  tgt_append=True,
                  use_tgt=False,
                  use_tag=False,
                  use_dst=False,
                  shuffle_tgt=False,
                  scale_to_tgt=0.0
                  ):
    from .tensor_ops import get_float_tensor
    sources = [e.src for e in examples]
    if not use_tgt and scale_to_tgt > 0:
        sources = [input_scaling(sent=c, max_tgt_len=max_tgt_len, scale=scale_to_tgt) for c in sources]

    sources_length = [len(c) for c in sources] if not src_append else [len(c) + 2 for c in sources]
    batch_sources = to_input_variable(
        sequences=sources, vocab=vocab.src, cuda=cuda,
        training=training, append_boundary_sym=src_append, batch_first=True
    )
    if not use_tgt:
        return {
            "src": batch_sources,
            "src_len": sources_length,
        }

    targets = [e.tgt for e in examples]

    if use_tgt and scale_to_tgt > 0:
        sources = []
        targets = []
        for e in examples:
            if len(e.src) < len(e.tgt) and scale_to_tgt > 0:
                src = e.src + ['</s>'] * (len(e.tgt) - len(e.src))
                tgt = e.tgt
            else:
                src = e.src
                tgt = e.tgt + ["</s>"] * (len(e.src) - len(e.tgt))
            sources.append(src)
            targets.append(tgt)

    sources_length = [len(c) for c in sources] if not src_append else [len(c) + 2 for c in sources]
    batch_sources = to_input_variable(
        sequences=sources, vocab=vocab.src, cuda=cuda,
        training=training, append_boundary_sym=src_append, batch_first=True
    )
    batch_targets = to_input_variable(
        sequences=targets, vocab=vocab.tgt, max_len=max_tgt_len, cuda=cuda,
        training=training, append_boundary_sym=tgt_append, batch_first=True
    )
    longest_len = batch_targets.size(1)
    if use_dst:
        distances = [e.dst for e in examples]
        batch_distances = []
        for dst in distances:
            padded_dst = dst + [0] * (longest_len - 1 - len(dst))
            batch_distances.append(padded_dst)
        batch_distances = get_float_tensor(batch_distances)
    else:
        batch_distances = None

    if use_tag:
        postags = [e.tag for e in examples]
        syntags = [e.arc for e in examples]
        batch_postags = to_input_variable(
            sequences=postags, vocab=vocab.arc, max_len=longest_len, cuda=cuda,
            training=training, append_boundary_sym=True, batch_first=True
        )
        batch_syntags = to_input_variable(
            sequences=syntags, vocab=vocab.arc, max_len=longest_len, cuda=cuda,
            training=training, append_boundary_sym=True, batch_first=True
        )
    else:
        batch_postags = None
        batch_syntags = None
    if not shuffle_tgt:
        return {
            "src": batch_sources,
            "src_len": sources_length,
            "tgt": batch_targets,
            "dst": batch_distances,
            "tag": batch_postags,
            "arc": batch_syntags,
        }
    else:
        targets = [e.tgt for e in examples]
        shuffle_targets, shuffle_positions = to_input_variable(sequences=targets, vocab=vocab.tgt, max_len=max_tgt_len,
                                                               cuda=cuda,
                                                               training=training, append_boundary_sym=tgt_append,
                                                               batch_first=True,
                                                               shuffle=True)

        return {
            "src": batch_sources,
            "src_len": sources_length,
            "tgt": batch_targets,
            "s_tgt": shuffle_targets,
            "s_pos": shuffle_positions,
            "dst": batch_distances,
            "tag": batch_postags,
            "arc": batch_syntags,
        }


def to_target_word(log_prob, vocab):
    _, word_ids = log_prob.sort(dim=-1, descending=True)
    word_ids = word_ids[:, :, 0].data.tolist()
    return [[[id2word(sents, vocab)], [-1]] for sents in word_ids]


def data_to_word(tensor, vocab):
    word_ids = tensor.squeeze(1).data.tolist()
    return [[[id2word(sents, vocab)], [-1]] for sents in word_ids]


def positional_encodings_like(x, t=None, use_cuda=True):  # hope to be differentiable
    """
    Args:
        x: [batch_size,length,hidden]
        t:
        use_cuda:
    """
    if t is None:
        positions = torch.arange(0, x.size(-2))  # .expand(*x.size()[:2])
        if use_cuda:
            positions = positions.cuda()
        positions = Variable(positions.float())
    else:
        positions = t

    # channels
    channels = torch.arange(0, x.size(-1), 2).float() / x.size(-1)  # 0 2 4 6 ... (256)
    if use_cuda:
        channels = channels.cuda()
    channels = 1 / (10000 ** Variable(channels).float())

    # get the positional encoding: batch x target_len
    encodings = positions.unsqueeze(-1) @ channels.unsqueeze(0)  # batch x target_len x 256
    encodings = torch.cat([torch.sin(encodings).unsqueeze(-1), torch.cos(encodings).unsqueeze(-1)], -1)
    encodings = encodings.contiguous().view(*encodings.size()[:-2], -1)  # batch x target_len x 512

    if encodings.ndimension() == 2:
        encodings = encodings.unsqueeze(0).expand_as(x)

    return encodings


def positional_encodings_from_range(batch_size, seq_len, hidden, use_cuda=True):
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
    encodings = encodings.expand(batch_size, seq_len, hidden)

    return encodings
