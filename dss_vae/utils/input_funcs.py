# coding=utf-8

import torch
from torch.autograd import Variable


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


def padding_input(sents, pad_token="<pad>", tgt_len=-1):
    """
    padding the token sequence to max length
    Args:
        sents:
        pad_token:
        tgt_len:

    Returns:

    """
    if tgt_len == -1:
        tgt_len = max(len(s) for s in sents)
    batch_size = len(sents)
    seqs = []
    for i in range(batch_size):
        seqs.append(sents[i][0:tgt_len] + [pad_token] * (tgt_len - len(sents[i])))
    return seqs


def scaling_input(sent, tgt_len=-1, eos_token="</s>", scale=0.0):
    """
    scaling the token sequence to min{ max_tgt_len, (1+scale) * len(sent) }
    Args:
        sent:
        tgt_len:
        eos_token:
        scale:

    Returns:

    """
    if scale <= 0:
        return sent
    else:
        pad_len = int(len(sent) * scale)
        if tgt_len != -1:
            # cur_dif = tgt_len - len(sent)
            # pad_len = pad_len if pad_len < cur_diff else cur_diff
            pad_len = tgt_len - len(sent)
        return sent + [eos_token] * pad_len


def to_input_variable(sequences, vocab, tgt_len=-1, training=True, append_boundary_sym=False,
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

    pad_sents = padding_input(sequences, tgt_len=tgt_len)
    seqs = word2id(pad_sents, vocab)

    if not training:
        with torch.no_grad():
            seqs_var = Variable(get_long_tensor(seqs), requires_grad=False)
    else:
        seqs_var = Variable(get_long_tensor(seqs), requires_grad=False)
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
    sources_length = [len(c) for c in sources] if not src_append else [len(c) + 2 for c in sources]
    max_src_length = int(max(sources_length) * (1 + scale_to_tgt))
    batch_sources = to_input_variable(
        sequences=sources, vocab=vocab.src, tgt_len=max_src_length,
        training=training, append_boundary_sym=src_append, batch_first=True
    )
    if not use_tgt:
        return {
            "src": batch_sources,
            "src_len": sources_length,
        }

    targets = [e.tgt for e in examples]
    if max_tgt_len == -1:
        targets_length = [len(c) for c in targets] if not tgt_append else [len(c) + 2 for c in targets]
        max_tgt_length = max(targets_length)
        common_length = max(max_src_length, max_tgt_length)
    else:
        common_length = max_tgt_len
    if scale_to_tgt > 0.0:
        batch_sources = to_input_variable(
            sequences=sources, vocab=vocab.src, tgt_len=common_length,
            training=training, append_boundary_sym=src_append, batch_first=True
        )
    batch_targets = to_input_variable(
        sequences=targets, vocab=vocab.tgt, tgt_len=common_length,
        training=training, append_boundary_sym=tgt_append, batch_first=True
    )
    if use_dst:
        distances = [e.dst for e in examples]
        batch_distances = []
        for dst in distances:
            padded_dst = dst + [0] * (common_length - 1 - len(dst))
            batch_distances.append(padded_dst)
        batch_distances = get_float_tensor(batch_distances)
    else:
        batch_distances = None

    if use_tag:
        postags = [e.tag for e in examples]
        syntags = [e.arc for e in examples]
        batch_postags = to_input_variable(
            sequences=postags, vocab=vocab.arc, tgt_len=common_length,
            training=training, append_boundary_sym=True, batch_first=True
        )
        batch_syntags = to_input_variable(
            sequences=syntags, vocab=vocab.arc, tgt_len=common_length,
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
        shuffle_targets, shuffle_positions = to_input_variable(sequences=targets, vocab=vocab.tgt,
                                                               tgt_len=common_length,
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


def reverse_to_word(inputs, vocab, batch_first=True, use_bpe=True):
    def trim(s, t):
        sentence = []
        for w in s:
            if w == t:
                break
            sentence.append(w)
        return sentence

    def filter_special(tok):
        return tok not in ("<s>", "<pad>")

    if not batch_first:
        inputs.t_()

    with torch.cuda.device_of(inputs):
        input_list = inputs.tolist()

    process_ret = [id2word(ex, vocab) for ex in input_list]  # denumericalize
    process_ret = [trim(ex, "</s>") for ex in process_ret]  # trim past frst eos
    if use_bpe:
        process_ret = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in process_ret]
    else:
        process_ret = [" ".join(filter(filter_special, ex)) for ex in process_ret]
    ret = [[[r], [-1]] for r in process_ret]
    return ret
