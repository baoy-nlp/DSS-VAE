import torch
import torch.nn as nn

from dss_vae.decoder.rnn_decoder import RNNDecoder
from dss_vae.encoder import RNNEncoder
from dss_vae.vae.base_vae import BaseVAE
from dss_vae.networks.bridger import MLPBridger
from dss_vae.utils.nn_funcs import to_input_variable
from dss_vae.utils.nn_funcs import to_var
from dss_vae.utils.nn_funcs import unk_replace
from dss_vae.utils.nn_funcs import wd_anneal_function


class VanillaVAE(BaseVAE):
    def __init__(self, args, vocab, word_embed=None):
        super(VanillaVAE, self).__init__(args, vocab, name='MySentVAE')
        print("This is {} with parameter\n{}".format(self.name, self.base_information()))
        if word_embed is None:
            src_embed = nn.Embedding(len(vocab.src), args.embed_size)
        else:
            src_embed = word_embed

        if args.share_embed:
            tgt_embed = src_embed
            args.dec_embed_dim = args.enc_embed_dim
        else:
            tgt_embed = None

        self.latent_size = args.latent_size
        self.unk_rate = args.unk_rate
        self.step_unk_rate = 0.0
        self.hidden_size = args.enc_hidden_dim
        self.hidden_factor = (2 if args.bidirectional else 1) * args.enc_num_layers
        args.use_attention = False

        # layer size setting
        self.enc_dim = args.enc_hidden_dim * (2 if args.bidirectional else 1)  # single layer unit size
        if args.mapper_type == "link":
            self.dec_hidden = self.enc_dim
        else:
            self.dec_hidden = args.dec_hidden_dim

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
            use_attention=False,
            embedding=tgt_embed,
            eos_id=vocab.src.eos_id,
            sos_id=vocab.src.sos_id,
        )

        self.hidden2mean = nn.Linear(args.hidden_size * self.hidden_factor, args.latent_size)
        self.hidden2logv = nn.Linear(args.hidden_size * self.hidden_factor, args.latent_size)
        self.latent2hidden = nn.Linear(args.latent_size, args.hidden_size * self.hidden_factor)

    def encode(self, input_var, length):
        if self.training and self.args.src_wd:
            input_var = unk_replace(input_var, self.step_unk_rate, self.vocab.src)
        encoder_output, encoder_hidden = self.encoder.forward(input_var, length)
        return encoder_output, encoder_hidden

    def decode(self, inputs, encoder_outputs, encoder_hidden):
        return self.decoder.forward(
            inputs=inputs,
            encoder_outputs=encoder_outputs,
            encoder_hidden=encoder_hidden
        )

    def forward(self, examples):
        if not isinstance(examples, list):
            examples = [examples]
        batch_size = len(examples)
        sent_words = [e.src for e in examples]
        ret = self.encode_to_hidden(examples)
        ret = self.hidden_to_latent(ret=ret, is_sampling=self.training)
        ret = self.latent_for_init(ret=ret)
        decode_init = ret['decode_init']
        tgt_var = to_input_variable(sent_words, self.vocab.src, training=False, cuda=self.args.cuda,
                                    append_boundary_sym=True, batch_first=True)
        decode_init = self.bridger.forward(decode_init)
        if self.training and self.args.tgt_wd > 0.:
            input_var = unk_replace(tgt_var, self.step_unk_rate, self.vocab.src)
            tgt_token_scores = self.decoder.generate(
                con_inputs=input_var,
                encoder_hidden=decode_init,
                encoder_outputs=None,
                teacher_forcing_ratio=1.0,
            )
            reconstruct_loss = -torch.sum(self.decoder.score_decoding_results(tgt_token_scores, tgt_var))
        else:
            reconstruct_loss = -torch.sum(self.decoder.score(
                inputs=tgt_var,
                encoder_outputs=None,
                encoder_hidden=decode_init,
            ))

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
        batch_size = explore['batch_size']
        kl_loss, kl_weight = self.compute_kl_loss(explore['mean'], explore['logv'], step)
        kl_weight *= self.args.kl_factor
        nll_loss = explore['nll_loss'] / batch_size
        kl_loss = kl_loss / batch_size
        kl_item = kl_loss * kl_weight
        return {
            'KL Loss': kl_loss,
            'NLL Loss': nll_loss,
            'KL Weight': kl_weight,
            'Model Score': kl_loss + nll_loss,
            'ELBO': kl_item + nll_loss,
            'Loss': kl_item + nll_loss,
            'KL Item': kl_item,
        }

    def sample_latent(self, batch_size):
        z = to_var(torch.randn([batch_size, self.latent_size]))
        return {
            "latent": z
        }

    def latent_for_init(self, ret):
        z = ret['latent']
        batch_size = z.size(0)
        hidden = self.latent2hidden(z)

        if self.hidden_factor > 1:
            hidden = hidden.view(batch_size, self.hidden_factor, self.hidden_size)
            hidden = hidden.permute(1, 0, 2)
        else:
            hidden = hidden.unsqueeze(0)
        ret['decode_init'] = hidden
        return ret

    def batch_beam_decode(self, examples):
        raise NotImplementedError

    def hidden_to_latent(self, ret, is_sampling=True):
        hidden = ret['hidden']
        batch_size = hidden.size(1)
        hidden = hidden.permute(1, 0, 2).contiguous()
        if self.hidden_factor > 1:
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
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

    def conditional_generating(self, condition='sem', examples=None):
        if not isinstance(examples, list):
            examples = [examples]
        if condition.startswith("sem"):
            ret = self.encode_to_hidden(examples)
            ret = self.hidden_to_latent(ret=ret, is_sampling=True)
            ret = self.latent_for_init(ret=ret)
            return {
                'res': self.decode_to_sentence(ret=ret)
            }
        if condition is None:
            return {
                "res": self.unsupervised_generating(sample_num=len(examples))
            }

    def eval_adv(self, sem_in, syn_ref):
        sem_ret = self.encode_to_hidden(sem_in)
        sem_ret = self.hidden_to_latent(sem_ret, is_sampling=self.training)
        syn_ret = self.encode_to_hidden(syn_ref, need_sort=True)
        syn_ret = self.hidden_to_latent(syn_ret, is_sampling=self.training)
        sem_ret = self.latent_for_init(ret=sem_ret)
        syn_ret = self.latent_for_init(ret=syn_ret)
        ret = dict()
        ret["latent"] = (sem_ret['latent'] + syn_ret['latent']) * 0.5
        ret = self.latent_for_init(ret)
        ret['res'] = self.decode_to_sentence(ret=ret)
        return ret
