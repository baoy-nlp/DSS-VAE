import torch
import torch.nn as nn

from dss_vae.decoder.rnn_decoder import RNNDecoder
from dss_vae.encoder import RNNEncoder
from dss_vae.vae.base_vae import BaseVAE
from dss_vae.networks.bridger import MLPBridger
from dss_vae.utils.nn_funcs import id2word
from dss_vae.utils.nn_funcs import to_input_variable
from dss_vae.utils.nn_funcs import to_var
from dss_vae.utils.nn_funcs import unk_replace
from dss_vae.utils.nn_funcs import wd_anneal_function


class SyntaxGuideVAE(BaseVAE):

    def encode(self, input_var, length):
        pass

    def conditional_generating(self, condition=None, **kwargs):
        pass

    def generating(self, sample_num, batch_size=50):
        return super().generating(sample_num, batch_size)

    def unsupervised_generating(self, sample_num, batch_size=50):
        return super().unsupervised_generating(sample_num, batch_size)

    def predict(self, examples, to_word=True):
        return super().predict(examples, to_word)

    def base_information(self):
        return super().base_information()

    def __init__(self, args, vocab, src_embed=None, tgt_embed=None):
        super(SyntaxGuideVAE, self).__init__(args, vocab, name='SyntaxGuideVAE')

        self.latent_size = args.latent_size
        self.rnn_type = args.rnn_type
        self.bidirectional = args.bidirectional
        self.num_layers = args.num_layers
        self.unk_rate = args.unk_rate
        self.step_unk_rate = 0.0

        self.encoder = RNNEncoder(
            vocab_size=len(vocab.src),
            max_len=args.src_max_time_step,
            input_size=args.enc_embed_dim,
            hidden_size=args.enc_hidden_dim,
            embed_droprate=args.enc_ed,
            rnn_droprate=args.enc_rd,
            n_layers=args.enc_num_layers,
            bidirectional=args.bidirectional,
            rnn_cell=args.rnn_type,
            variable_lengths=True,
            embedding=src_embed
        )

        self.syntax_encoder = RNNEncoder(
            vocab_size=len(vocab.tgt),
            max_len=args.tgt_max_time_step,
            input_size=args.enc_embed_dim,
            hidden_size=args.enc_hidden_dim,
            embed_droprate=args.enc_ed,
            rnn_droprate=args.enc_rd,
            n_layers=args.enc_num_layers,
            bidirectional=args.bidirectional,
            rnn_cell=args.rnn_type,
            variable_lengths=True,
            embedding=tgt_embed
        )

        self.hidden_size = args.enc_hidden_dim
        self.hidden_factor = (2 if args.bidirectional else 1) * args.enc_num_layers
        self.hidden2mean = nn.Linear(self.hidden_size * self.hidden_factor, self.latent_size)
        self.hidden2logv = nn.Linear(self.hidden_size * self.hidden_factor, self.latent_size)
        self.latent2hidden = nn.Linear(self.latent_size, self.hidden_size * self.hidden_factor)

        self.enc_dim = args.enc_hidden_dim * (2 if args.bidirectional else 1)
        if args.mapper_type == "link":
            self.dec_hidden = self.enc_dim
        elif args.use_attention:
            self.dec_hidden = self.enc_dim
        else:
            self.dec_hidden = args.dec_hidden_dim

        self.bridger = MLPBridger(
            rnn_type=args.rnn_type,
            mapper_type=args.mapper_type,
            encoder_dim=self.enc_dim,
            encoder_layer=args.enc_num_layers,
            decoder_dim=self.dec_hidden,
            decoder_layer=args.dec_num_layers,
        )

        self.decoder = RNNDecoder(
            vocab=len(vocab.src),
            max_len=args.src_max_time_step,
            input_size=args.dec_embed_dim,
            hidden_size=self.dec_hidden,
            embed_droprate=args.dec_ed,
            rnn_droprate=args.dec_rd,
            n_layers=args.dec_num_layers,
            rnn_cell=args.rnn_type,
            use_attention=args.use_attention,
            embedding=src_embed,
            eos_id=vocab.src.eos_id,
            sos_id=vocab.src.sos_id,
        )

    def syntax_encode(self, syntax_var, length):
        syntax_outputs, syntax_hidden = self.syntax_encoder.forward(syntax_var, length)
        return syntax_outputs, syntax_hidden

    def sentence_encode(self, sent_words):
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

        _, sent_hidden = self.encoder.forward(sorted_sent_var, sorted_sent_lengths)

        hidden = sent_hidden[:, example_old_pos_map, :]

        return hidden

    def forward(self, examples):
        if not isinstance(examples, list):
            examples = [examples]
        batch_size = len(examples)
        ret = self.encode_to_hidden(examples)
        ret = self.hidden_to_latent(ret=ret, is_sampling=self.training)
        ret = self.latent_for_init(ret=ret)
        decode_init = ret['decode_init']
        tgt_var = ret['tgt_var']
        syntax_output = ret['syn_output']
        decode_init = self.bridger.forward(decode_init)
        if self.training and self.args.tgt_wd:
            input_var = unk_replace(tgt_var, self.step_unk_rate, self.vocab.src)
            tgt_token_scores = self.decoder.generate(
                con_inputs=input_var,
                encoder_outputs=syntax_output,
                encoder_hidden=decode_init,
                teacher_forcing_ratio=1.0,
            )
            reconstruct_loss = -self.decoder.score_decoding_results(tgt_token_scores, tgt_var)
        else:
            reconstruct_loss = -self.decoder.score(
                inputs=tgt_var,
                encoder_outputs=syntax_output,
                encoder_hidden=decode_init,
            )

        return {
            "mean": ret['mean'],
            "logv": ret['logv'],
            "z": ret['latent'],
            'nll_loss': reconstruct_loss,
            'batch_size': batch_size
        }

    def get_loss(self, examples, step):
        self.step_unk_rate = wd_anneal_function(unk_max=self.unk_rate, anneal_function=self.args.unk_schedule,
                                                step=step, x0=self.args.x0,
                                                k=self.args.k)
        explore = self.forward(examples)
        kl_loss, kl_weight = self.compute_kl_loss(explore['mean'], explore['logv'], step)
        kl_weight *= self.args.kl_factor
        nll_loss = torch.sum(explore['nll_loss']) / explore['batch_size']
        kl_loss = kl_loss / explore['batch_size']
        kl_item = kl_loss * kl_weight
        loss = kl_item + nll_loss
        return {
            'KL Loss': kl_loss,
            'NLL Loss': nll_loss,
            'Model Score': nll_loss + kl_loss,
            'Loss': loss,
            'ELBO': loss,
            'KL Weight': kl_weight,
            'WD Drop': self.step_unk_rate,
            'KL Item': kl_item,
        }

    def batch_beam_decode(self, **kwargs):
        pass

    def latent_for_init(self, ret):
        z = ret['latent']
        batch_size = z.size(0)
        hidden = self.latent2hidden(z)

        if self.hidden_factor > 1:
            hidden = hidden.view(batch_size, self.hidden_factor, self.args.enc_hidden_dim)
            hidden = hidden.permute(1, 0, 2)
        else:
            hidden = hidden.unsqueeze(0)
        ret['decode_init'] = (hidden + ret['syn_hidden']) / 2
        return ret

    def sample_latent(self, batch_size):
        z = to_var(torch.randn([batch_size, self.latent_size]))
        return {
            "latent": z
        }

    def encode_to_hidden(self, examples):
        if not isinstance(examples, list):
            examples = [examples]
        batch_size = len(examples)
        sorted_example_ids = sorted(range(batch_size), key=lambda x: -len(examples[x].tgt))
        example_old_pos_map = [-1] * batch_size

        sorted_examples = [examples[i] for i in sorted_example_ids]

        syntax_word = [e.tgt for e in sorted_examples]
        syntax_var = to_input_variable(syntax_word, self.vocab.tgt, training=False, cuda=self.args.cuda,
                                       batch_first=True)
        length = [len(e.tgt) for e in sorted_examples]
        syntax_output, syntax_hidden = self.syntax_encode(syntax_var, length)

        sent_words = [e.src for e in sorted_examples]
        sentence_hidden = self.sentence_encode(sent_words)
        tgt_var = to_input_variable(sent_words, self.vocab.src, training=False, cuda=self.args.cuda,
                                    append_boundary_sym=True, batch_first=True)

        for new_pos, old_pos in enumerate(sorted_example_ids):
            example_old_pos_map[old_pos] = new_pos

        return {
            'hidden': sentence_hidden,
            "syn_output": syntax_output,
            "syn_hidden": syntax_hidden,
            'tgt_var': tgt_var,
            'old_pos': example_old_pos_map
        }

    def hidden_to_latent(self, ret, is_sampling):
        hidden = ret['hidden']
        batch_size = hidden.size(1)
        hidden = hidden.permute(1, 0, 2).contiguous()
        if self.hidden_factor > 1:
            hidden = hidden.view(batch_size, self.args.enc_hidden_dim * self.hidden_factor)
        else:
            hidden = hidden.squeeze()
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        if is_sampling:
            std = torch.exp(0.5 * logv)
            z = to_var(torch.randn([batch_size, self.latent_size]))
            z = z * std + mean
        else:
            z = mean
        ret["latent"] = z
        ret["mean"] = mean
        ret['logv'] = logv
        return ret

    def decode_to_sentence(self, ret):
        sentence_decode_init = ret['decode_init']
        sentence_decode_init = self.bridger.forward(input_tensor=sentence_decode_init)

        decoder_outputs, decoder_hidden, ret_dict, enc_states = self.decoder.forward(
            inputs=None,
            encoder_outputs=None,
            encoder_hidden=sentence_decode_init,
        )

        result = torch.stack(ret_dict['sequence']).squeeze()
        temp_result = []
        if result.dim() < 2:
            result = result.unsqueeze(1)
        example_nums = result.size(1)
        for i in range(example_nums):
            hyp = result[:, i].data.tolist()
            res = id2word(hyp, self.vocab.src)
            seems = [[res], [len(res)]]
            temp_result.append(seems)

        final_result = [temp_result[i] for i in ret['old_pos']]
        return final_result
