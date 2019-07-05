import os

import torch
import torch.nn as nn

# from source.models.base_gen import BaseGenerator
from dss_vae.utils.nn_funcs import id2word
from dss_vae.utils.nn_funcs import to_input_variable
from dss_vae.utils.schedule_funs import kl_anneal_function
from dss_vae.utils.schedule_funs import unk_replace


class BaseVAE(nn.Module):
    """
    contains process:
    1. encoder sentence to latent variable [ret]
    2. decoder to the sample sentence
    """

    # def get_loss(self, **kwargs):
    #     return self.score(**kwargs)

    def score(self, **kwargs):
        raise NotImplementedError

    def __init__(self, args, vocab, name="Sentence VAE"):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.name = name
        self.step_kl_weight = args.init_step_kl_weight

    def base_information(self):
        return "layers:{}\n" \
               "types:{}\n" \
               "embed:{}\n" \
               "hidden:{}\n" \
               "latent:{}\n" \
               "kl_factor:{}\n" \
               "unk_rate:{}\n" \
               "x0:{}\n".format(str(self.args.num_layers),
                                str(self.args.rnn_type),
                                str(self.args.embed_size),
                                str(self.args.hidden_size),
                                str(self.args.latent_size),
                                str(self.args.kl_factor),
                                str(self.args.unk_rate),
                                str(self.args.x0))

    def forward(self, **kwargs):
        raise NotImplementedError

    def get_loss(self, **kwargs):
        raise NotImplementedError

    def predict(self, examples, to_word=True):
        """
        Args:
            examples: raw format with <src,tgt>
            to_word: whether or not convert the output to word token
        """
        if not isinstance(examples, list):
            examples = [examples]
        ret = self.encode_to_hidden(examples)
        ret = self.hidden_to_latent(ret=ret, is_sampling=self.training)
        ret = self.latent_for_init(ret=ret)
        return self.decode_to_sentence(ret=ret)

    def generating(self, sample_num, batch_size=50):
        """
        sampling sample_num latent variable, then generating to a sentence
        Args:
            sample_num: sum number of generating
            batch_size:
        """

        batch_size = batch_size if sample_num >= batch_size else sample_num
        batch_num = int((sample_num - 1) / batch_size)
        final_res = []
        for _ in range(batch_num + 1):
            ret = self.sample_latent(batch_size)
            ret = self.latent_for_init(ret)
            res = self.decode_to_sentence(ret)
            final_res.extend(res)
        return final_res

    def unsupervised_generating(self, sample_num, batch_size=50):
        batch_size = batch_size if sample_num >= batch_size else sample_num
        batch_num = int((sample_num - 1) / batch_size)
        final_res = []
        for _ in range(batch_num + 1):
            ret = self.sample_latent(batch_size)
            ret = self.latent_for_init(ret)
            res = self.decode_to_sentence(ret)
            final_res.extend(res)
        return final_res

    def conditional_generating(self, condition=None, **kwargs):
        raise NotImplementedError

    def batch_beam_decode(self, **kwargs):
        pass

    def encode(self, input_var, length):
        raise NotImplementedError

    def decode(self, inputs, encoder_outputs, encoder_hidden):
        raise NotImplementedError

    def encode_to_hidden(self, examples, need_sort=False, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        if not need_sort:
            sent_words = [e.src for e in examples]
            length = [len(e.src) for e in examples]
            src_var = to_input_variable(sent_words, self.vocab.src, training=False, cuda=self.args.cuda,
                                        batch_first=True)

            encoder_output, encoder_hidden = self.encode(input_var=src_var, length=length)

            return {
                "outputs": encoder_output,
                "hidden": encoder_hidden,
                'length': length,
                'batch_size': len(examples)
            }
        sent_words = [e.src for e in examples]
        batch_size = len(sent_words)
        sent_lengths = [len(sent_word) for sent_word in sent_words]
        sorted_example_ids = sorted(range(batch_size), key=lambda x: -sent_lengths[x])

        example_old_pos_map = [-1] * batch_size
        for new_pos, old_pos in enumerate(sorted_example_ids):
            example_old_pos_map[old_pos] = new_pos

        sorted_sent_words = [sent_words[i] for i in sorted_example_ids]
        sorted_sent_var = to_input_variable(sorted_sent_words, self.vocab.src, cuda=self.args.cuda, batch_first=True)

        if self.training and self.args.src_wd:
            sorted_sent_var = unk_replace(sorted_sent_var, self.step_unk_rate, self.vocab.src)

        sorted_sent_lengths = [len(sent_word) for sent_word in sorted_sent_words]

        _, sent_hidden = self.encode(sorted_sent_var, sorted_sent_lengths)

        if sent_hidden.dim() > 2:
            hidden = sent_hidden[:, example_old_pos_map, :]
        else:
            hidden = sent_hidden[example_old_pos_map, :]

        return {
            "outputs": None,
            "hidden": hidden,
            'length': sent_lengths,
            'batch_size': batch_size
        }

    def hidden_to_latent(self, ret, is_sampling):
        """return encoder hidden"""
        raise NotImplementedError

    def sample_latent(self, batch_size):
        """return sample from latent"""
        raise NotImplementedError

    def latent_for_init(self, ret):
        raise NotImplementedError

    def decode_to_sentence(self, ret):
        sentence_decode_init = ret['decode_init']
        sentence_decode_init = self.bridger.forward(input_tensor=sentence_decode_init)

        decoder_outputs, decoder_hidden, ret_dict, enc_states = self.decode(
            inputs=None,
            encoder_outputs=None,
            encoder_hidden=sentence_decode_init,
        )

        result = torch.stack(ret_dict['sequence']).squeeze()
        final_result = []
        if result.dim() < 2:
            result = result.unsqueeze(1)
        example_nums = result.size(1)
        for i in range(example_nums):
            hyp = result[:, i].data.tolist()
            res = id2word(hyp, self.vocab.src)
            seems = [[res], [len(res)]]
            final_result.append(seems)

        return final_result

    def get_kl_weight(self, step):
        if self.step_kl_weight is None:
            return kl_anneal_function(self.args.anneal_function, step, self.args.k, self.args.x0)
        else:
            return self.step_kl_weight

    def compute_kl_loss(self, mean, logv, step):
        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        kl_weight = self.get_kl_weight(step)
        return kl_loss, kl_weight

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict)

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if not os.path.exists(path + ".config"):
            with open(path + ".config", "w") as f:
                f.write("{}".format(self.base_information()))

        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict(),
        }

        torch.save(params, path)
