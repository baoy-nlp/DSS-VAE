from __future__ import division

import torch
import torch.nn as nn

from dss_vae.decoder.rnn_decoder import RNNDecoder
from dss_vae.encoder import RNNEncoder
from dss_vae.networks.bridger import MLPBridger
from dss_vae.utils.loss_funcs import bag_of_word_loss
from dss_vae.utils.nn_funcs import id2word
from dss_vae.utils.nn_funcs import to_input_variable
from dss_vae.utils.nn_funcs import to_var
from dss_vae.utils.nn_funcs import unk_replace
from dss_vae.utils.nn_funcs import wd_anneal_function
from dss_vae.vae.base_vae import BaseVAE


class DisentangleVAE(BaseVAE):
    """
    Encoder the sentence, predict the parser,
    """

    def score(self, **kwargs):
        pass

    def decode(self, inputs, encoder_outputs, encoder_hidden):
        return self.decoder.forward(
            inputs=inputs,
            encoder_outputs=encoder_outputs,
            encoder_hidden=encoder_hidden
        )

    def __init__(self, args, vocab, src_embed=None, tgt_embed=None):
        super(DisentangleVAE, self).__init__(args, vocab, name="Disentangle VAE with deep encoder")
        print("This is {} with parameter\n{}".format(self.name, self.base_information()))
        if src_embed is None:
            self.src_embed = nn.Embedding(len(vocab.src), args.embed_size)
        else:
            self.src_embed = src_embed
        if tgt_embed is None:
            self.tgt_embed = nn.Embedding(len(vocab.tgt), args.embed_size)
        else:
            self.tgt_embed = tgt_embed

        self.pad_idx = vocab.src.sos_id

        self.latent_size = int(args.latent_size)
        self.rnn_type = args.rnn_type
        self.unk_rate = args.unk_rate
        self.step_unk_rate = 0.0
        self.direction_num = 2 if args.bidirectional else 1

        self.enc_hidden_dim = args.enc_hidden_dim
        self.enc_layer_dim = args.enc_hidden_dim * self.direction_num
        self.enc_hidden_factor = self.direction_num * args.enc_num_layers
        self.dec_hidden_factor = args.dec_num_layers
        args.use_attention = False
        if args.mapper_type == "link":
            self.dec_layer_dim = self.enc_layer_dim
        elif args.use_attention:
            self.dec_layer_dim = self.enc_layer_dim
        else:
            self.dec_layer_dim = args.dec_hidden_dim

        syn_var_dim = int(self.enc_hidden_dim * self.enc_hidden_factor / 2)
        sem_var_dim = int(self.enc_hidden_dim * self.enc_hidden_factor / 2)

        task_enc_dim = int(self.enc_layer_dim / 2)
        task_dec_dim = int(self.dec_layer_dim / 2)

        self.encoder = RNNEncoder(
            vocab_size=len(vocab.src),
            max_len=args.src_max_time_step,
            input_size=args.enc_embed_dim,
            hidden_size=self.enc_hidden_dim,
            embed_droprate=args.enc_ed,
            rnn_droprate=args.enc_rd,
            n_layers=args.enc_num_layers,
            bidirectional=args.bidirectional,
            rnn_cell=args.rnn_type,
            variable_lengths=True,
            embedding=self.src_embed
        )
        # output: [layer*direction ,batch_size, enc_hidden_dim]

        pack_decoder = BridgeRNN(
            args,
            vocab,
            enc_hidden_dim=self.enc_layer_dim,
            dec_hidden_dim=self.dec_layer_dim,
            embed=self.src_embed if args.share_embed else None,
            mode='src'
        )
        self.bridger = pack_decoder.bridger
        self.decoder = pack_decoder.decoder

        if "report" in self.args:
            syn_common = nn.Sequential(
                nn.Linear(syn_var_dim, self.latent_size * 2, True),
                nn.ReLU()
            )
            self.syn_mean = nn.Sequential(
                syn_common,
                nn.Linear(self.latent_size * 2, self.latent_size)
            )
            self.syn_logv = nn.Sequential(
                syn_common,
                nn.Linear(self.latent_size * 2, self.latent_size)
            )

            sem_common = nn.Sequential(
                nn.Linear(sem_var_dim, self.latent_size * 2, True),
                nn.ReLU()
            )
            self.sem_mean = nn.Sequential(
                sem_common,
                nn.Linear(self.latent_size * 2, self.latent_size)
            )
            self.sem_logv = nn.Sequential(
                sem_common,
                nn.Linear(self.latent_size * 2, self.latent_size))
        else:
            self.syn_mean = nn.Linear(syn_var_dim, self.latent_size)
            self.syn_logv = nn.Linear(syn_var_dim, self.latent_size)
            self.sem_mean = nn.Linear(sem_var_dim, self.latent_size)
            self.sem_logv = nn.Linear(sem_var_dim, self.latent_size)

        self.syn_to_h = nn.Linear(self.latent_size, syn_var_dim)
        self.sem_to_h = nn.Linear(self.latent_size, sem_var_dim)

        self.sup_syn = BridgeRNN(
            args,
            vocab,
            enc_hidden_dim=task_enc_dim,
            dec_hidden_dim=task_dec_dim,
            embed=tgt_embed,
            mode='tgt'
        )

        self.sup_sem = BridgeMLP(
            args=args,
            vocab=vocab,
            enc_dim=task_enc_dim,
            dec_hidden=task_dec_dim,
        )

        self.syn_adv = BridgeRNN(
            args,
            vocab,
            enc_hidden_dim=task_enc_dim,
            dec_hidden_dim=task_dec_dim,
            embed=self.tgt_embed if args.share_embed else None,
            mode='tgt'
        )

        self.syn_infer = BridgeRNN(
            args,
            vocab,
            enc_hidden_dim=task_enc_dim,
            dec_hidden_dim=task_dec_dim,
            embed=self.src_embed if args.share_embed else None,
            mode='src'
        )

        self.sem_adv = BridgeMLP(
            args=args,
            vocab=vocab,
            enc_dim=task_enc_dim,
            dec_hidden=task_dec_dim,
        )

        self.sem_infer = BridgeRNN(
            args,
            vocab,
            enc_hidden_dim=task_enc_dim,
            dec_hidden_dim=task_dec_dim,
            embed=self.src_embed if args.share_embed else None,
            mode='src'
        )

    def base_information(self):
        origin = super().base_information()
        return origin + "mul_syn:{}\n" \
                        "mul_sen:{}\n" \
                        "adv_syn:{}\n" \
                        "adv_sem:{}\n" \
                        "inf_syn:{}\n" \
                        "inf_sem:{}\n" \
                        "kl_syn:{}\n" \
                        "kl_sem:{}\n".format(str(self.args.mul_syn),
                                             str(self.args.mul_sem),
                                             str(self.args.adv_syn),
                                             str(self.args.adv_sem),
                                             str(self.args.inf_syn * self.args.infer_weight),
                                             str(self.args.inf_sem * self.args.infer_weight),
                                             str(self.args.syn_weight),
                                             str(self.args.sem_weight)
                                             )

    def encode(self, input_var, length):
        if self.training and self.args.src_wd > 0.:
            input_var = unk_replace(input_var, self.step_unk_rate, self.vocab.src)

        encoder_output, encoder_hidden = self.encoder.forward(input_var, length)
        return encoder_output, encoder_hidden

    def forward(self, examples, is_dis=False):
        if not isinstance(examples, list):
            examples = [examples]
        batch_size = len(examples)

        words = [e.src for e in examples]
        tgt_var = to_input_variable(words, self.vocab.src, training=False, cuda=self.args.cuda,
                                    append_boundary_sym=True, batch_first=True)
        syn_seqs = [e.tgt for e in examples]
        syn_var = to_input_variable(syn_seqs, self.vocab.tgt, training=False, cuda=self.args.cuda,
                                    append_boundary_sym=True, batch_first=True)

        ret = self.encode_to_hidden(examples)
        ret = self.hidden_to_latent(ret=ret, is_sampling=self.training)
        ret = self.latent_for_init(ret=ret)
        syn_hidden = ret['syn_hidden']
        sem_hidden = ret['sem_hidden']

        if is_dis:
            dis_syn_loss, dis_sem_loss = self.get_dis_loss(
                syntax_hidden=syn_hidden,
                semantic_hidden=sem_hidden,
                syn_tgt=syn_var,
                sem_tgt=tgt_var
            )
            ret['dis syn'] = dis_syn_loss
            ret['dis sem'] = dis_sem_loss
            return ret

        decode_init = ret['decode_init']

        sentence_decode_init = self.bridger.forward(decode_init)
        if self.training and self.args.tgt_wd:
            input_var = unk_replace(tgt_var, self.step_unk_rate, self.vocab.src)
            tgt_log_score = self.decoder.generate(
                con_inputs=input_var,
                encoder_hidden=sentence_decode_init,
                encoder_outputs=None,
                teacher_forcing_ratio=1.0)
            reconstruct_loss = -torch.sum(self.decoder.score_decoding_results(tgt_log_score, tgt_var))
        else:
            reconstruct_loss = -torch.sum(self.decoder.score(
                inputs=tgt_var,
                encoder_outputs=None,
                encoder_hidden=sentence_decode_init))

        mul_syn_loss, mul_sem_loss = self.get_mul_loss(
            syntax_hidden=syn_hidden,
            semantic_hidden=sem_hidden,
            syn_tgt=syn_var,
            sem_tgt=tgt_var
        )

        adv_syn_loss, adv_sem_loss = self.get_adv_loss(
            syntax_hidden=syn_hidden,
            semantic_hidden=sem_hidden,
            syn_tgt=syn_var,
            sem_tgt=tgt_var
        )
        ret['adv'] = adv_syn_loss + adv_sem_loss
        ret['mul'] = mul_syn_loss + mul_sem_loss

        ret['nll_loss'] = reconstruct_loss
        ret['sem_loss'] = mul_sem_loss
        ret['syn_loss'] = mul_syn_loss
        ret['batch_size'] = batch_size
        return ret

    def get_loss(self, examples, step, is_dis=False):
        self.step_unk_rate = wd_anneal_function(unk_max=self.unk_rate, anneal_function=self.args.unk_schedule,
                                                step=step, x0=self.args.x0,
                                                k=self.args.k)
        explore = self.forward(examples, is_dis)

        if is_dis:
            return explore

        sem_kl, kl_weight = self.compute_kl_loss(
            mean=explore['sem_mean'],
            logv=explore['sem_logv'],
            step=step,
        )
        syn_kl, _ = self.compute_kl_loss(
            mean=explore['syn_mean'],
            logv=explore['syn_logv'],
            step=step,
        )

        batch_size = explore['batch_size']
        kl_weight *= self.args.kl_factor
        kl_loss = (self.args.sem_weight * sem_kl + self.args.syn_weight * syn_kl) / (
                self.args.sem_weight + self.args.syn_weight)
        kl_loss /= batch_size
        mul_loss = explore['mul'] / batch_size
        adv_loss = explore['adv'] / batch_size
        nll_loss = explore['nll_loss'] / batch_size
        kl_item = kl_loss * kl_weight

        return {
            'KL Loss': kl_loss,
            'NLL Loss': nll_loss,
            'MUL Loss': mul_loss,
            'ADV Loss': adv_loss,
            'KL Weight': kl_weight,
            'KL Item': kl_item,
            'Model Score': kl_loss + nll_loss,
            'ELBO': kl_item + nll_loss,
            'Loss': kl_item + nll_loss + mul_loss - adv_loss,
            'SYN KL Loss': syn_kl / explore['batch_size'],
            'SEM KL Loss': sem_kl / explore['batch_size'],
        }

    def get_adv_loss(self, syntax_hidden, semantic_hidden, syn_tgt, sem_tgt):
        if self.training:
            with torch.no_grad():
                loss_dict = self._dis_loss(syntax_hidden, semantic_hidden, syn_tgt, sem_tgt)
            if self.args.infer_weight > 0.:
                adv_syn = self.args.adv_syn * loss_dict['adv_syn_sup'] + self.args.infer_weight * self.args.inf_sem * \
                          loss_dict['adv_sem_inf']
                adv_sem = self.args.adv_sem * loss_dict['adv_sem_sup'] + self.args.infer_weight * self.args.inf_syn * \
                          loss_dict['adv_syn_inf']
            else:
                adv_syn = self.args.adv_syn * loss_dict['adv_syn_sup']
                adv_sem = self.args.adv_sem * loss_dict['adv_sem_sup']
            return adv_syn, adv_sem
        else:
            loss_dict = self._dis_loss(syntax_hidden, semantic_hidden, syn_tgt, sem_tgt)
            if self.args.infer_weight > 0.:
                adv_syn = self.args.adv_syn * loss_dict['adv_syn_sup'] + self.args.infer_weight * self.args.inf_sem * \
                          loss_dict['adv_sem_inf']
                adv_sem = self.args.adv_sem * loss_dict['adv_sem_sup'] + self.args.infer_weight * self.args.inf_syn * \
                          loss_dict['adv_syn_inf']
            else:
                adv_syn = self.args.adv_syn * loss_dict['adv_syn_sup']
                adv_sem = self.args.adv_sem * loss_dict['adv_sem_sup']

            return adv_syn, adv_sem

    def get_dis_loss(self, syntax_hidden, semantic_hidden, syn_tgt, sem_tgt):
        syntax_hid = syntax_hidden.detach()
        semantic_hid = semantic_hidden.detach()

        loss_dict = self._dis_loss(syntax_hid, semantic_hid, syn_tgt, sem_tgt)
        if self.args.infer_weight > 0.:
            return loss_dict['adv_syn_sup'] + loss_dict['adv_sem_inf'], loss_dict['adv_sem_sup'] + loss_dict[
                'adv_syn_inf']
        else:
            return loss_dict['adv_syn_sup'], loss_dict['adv_sem_sup']

    def _dis_loss(self, syntax_hidden, semantic_hidden, syn_tgt, sem_tgt):
        dis_syn_sup = self.syn_adv.forward(hidden=semantic_hidden, tgt_var=syn_tgt)
        dis_sem_sup = self.sem_adv.forward(hidden=syntax_hidden, tgt_var=sem_tgt)
        if self.args.infer_weight > 0.:

            dis_syn_inf = self.syn_infer.forward(hidden=syntax_hidden, tgt_var=sem_tgt)
            dis_sem_inf = self.sem_infer.forward(hidden=semantic_hidden, tgt_var=sem_tgt)
            return {
                'adv_syn_sup': dis_syn_sup if self.args.adv_syn > 0. else 0.,
                'adv_sem_sup': dis_sem_sup if self.args.adv_sem > 0. else 0.,
                'adv_syn_inf': dis_syn_inf if self.args.inf_syn > 0. else 0.,
                "adv_sem_inf": dis_sem_inf if self.args.inf_sem > 0. else 0.
            }
        else:
            return {
                'adv_syn_sup': dis_syn_sup,
                'adv_sem_sup': dis_sem_sup,
            }

    def get_mul_loss(self, syntax_hidden, semantic_hidden, syn_tgt, sem_tgt):
        syn_loss = self.sup_syn.forward(hidden=syntax_hidden, tgt_var=syn_tgt)
        sem_loss = self.sup_sem.forward(hidden=semantic_hidden, tgt_var=sem_tgt)
        return self.args.mul_syn * syn_loss, self.args.mul_sem * sem_loss

    def sample_latent(self, batch_size):
        syntax_latent = to_var(torch.randn([batch_size, self.latent_size]))
        semantic_latent = to_var(torch.randn([batch_size, self.latent_size]))
        return {
            "syn_z": syntax_latent,
            "sem_z": semantic_latent,
        }

    def hidden_to_latent(self, ret, is_sampling=True):
        hidden = ret['hidden']

        def sampling(mean, logv):
            if is_sampling:
                std = torch.exp(0.5 * logv)
                z = to_var(torch.randn([batch_size, self.latent_size]))
                z = z * std + mean
            else:
                z = mean
            return z

        def split_hidden(encode_hidden):
            bs = encode_hidden.size(1)
            factor = encode_hidden.size(0)
            hid = encode_hidden.permute(1, 0, 2).contiguous().view(bs, factor, 2, -1)
            return hid[:, :, 0, :].contiguous().view(bs, -1), hid[:, :, 1, :].contiguous().view(bs, -1)

        batch_size = hidden.size(1)
        sem_hid, syn_hid = split_hidden(hidden)

        semantic_mean = self.sem_mean(sem_hid)
        semantic_logv = self.sem_logv(sem_hid)
        syntax_mean = self.syn_mean(syn_hid)
        syntax_logv = self.syn_logv(syn_hid)
        syntax_latent = sampling(syntax_mean, syntax_logv)
        semantic_latent = sampling(semantic_mean, semantic_logv)

        ret['syn_mean'] = syntax_mean
        ret['syn_logv'] = syntax_logv
        ret['sem_mean'] = semantic_mean
        ret['sem_logv'] = semantic_logv
        ret['syn_z'] = syntax_latent
        ret['sem_z'] = semantic_latent

        return ret

    def latent_for_init(self, ret):
        def reshape(xx_hidden):
            xx_hidden = xx_hidden.view(batch_size, self.enc_hidden_factor, self.enc_hidden_dim / 2)
            xx_hidden = xx_hidden.permute(1, 0, 2)
            return xx_hidden

        syntax_latent = ret['syn_z']
        semantic_latent = ret['sem_z']
        batch_size = semantic_latent.size(0)
        syntax_hidden = reshape(self.syn_to_h(syntax_latent))
        semantic_hidden = reshape(self.sem_to_h(semantic_latent))

        ret['syn_hidden'] = syntax_hidden
        ret['sem_hidden'] = semantic_hidden
        ret['decode_init'] = torch.cat([syntax_hidden, semantic_hidden], dim=-1)
        return ret

    def evaluate_(self, examples, beam_size=5):
        if not isinstance(examples, list):
            examples = [examples]
        ret = self.encode_to_hidden(examples)
        ret = self.hidden_to_latent(ret=ret, is_sampling=self.training)
        ret = self.latent_for_init(ret=ret)
        ret['res'] = self.decode_to_sentence(ret=ret)
        return ret

    def predict_syntax(self, hidden, predictor):
        result = predictor.predict(hidden)
        numbers = result.size(1)
        final_result = []
        for i in range(numbers):
            hyp = result[:, i].data.tolist()
            res = id2word(hyp, self.vocab.tgt)
            seems = [[res], [len(res)]]
            final_result.append(seems)
        return final_result

    def extract_variable(self, examples):
        pass

    def eval_syntax(self, examples):
        ret = self.encode_to_hidden(examples, need_sort=True)
        ret = self.hidden_to_latent(ret, is_sampling=False)
        ret = self.latent_for_init(ret)
        return self.predict_syntax(hidden=ret['syn_hidden'], predictor=self.sup_syn)

    def eval_adv(self, sem_in, syn_ref):
        sem_ret = self.encode_to_hidden(sem_in)
        sem_ret = self.hidden_to_latent(sem_ret, is_sampling=self.training)
        syn_ret = self.encode_to_hidden(syn_ref, need_sort=True)
        syn_ret = self.hidden_to_latent(syn_ret, is_sampling=self.training)
        sem_ret = self.latent_for_init(ret=sem_ret)
        syn_ret = self.latent_for_init(ret=syn_ret)
        ret = dict(sem_z=sem_ret['sem_z'], syn_z=syn_ret['syn_z'])
        ret = self.latent_for_init(ret)
        ret['res'] = self.decode_to_sentence(ret=ret)
        ret['ori syn'] = self.predict_syntax(hidden=sem_ret['syn_hidden'], predictor=self.sup_syn)
        ret['ref syn'] = self.predict_syntax(hidden=syn_ret['syn_hidden'], predictor=self.sup_syn)
        return ret

    def conditional_generating(self, condition="sem", examples=None):
        ref_ret = self.encode_to_hidden(examples)
        ref_ret = self.hidden_to_latent(ref_ret, is_sampling=True)
        if condition.startswith("sem"):
            ref_ret['sem_z'] = ref_ret['sem_mean']
        else:
            ref_ret['syn_z'] = ref_ret['syn_mean']

        if condition == "sem-only":
            sam_ref = self.sample_latent(batch_size=ref_ret['batch_size'])
            ref_ret['syn_z'] = sam_ref['syn_z']

        ret = self.latent_for_init(ret=ref_ret)

        ret['res'] = self.decode_to_sentence(ret=ret)
        return ret


class BridgeRNN(nn.Module):
    def __init__(self, args, vocab, enc_hidden_dim, dec_hidden_dim, embed, mode='src'):
        super().__init__()

        self.bridger = MLPBridger(
            rnn_type=args.rnn_type,
            mapper_type=args.mapper_type,
            encoder_dim=enc_hidden_dim,
            encoder_layer=args.enc_num_layers,
            decoder_dim=dec_hidden_dim,
            decoder_layer=args.dec_num_layers,
        )
        if mode == 'src':
            self.decoder = RNNDecoder(
                vocab=len(vocab.src),
                max_len=args.src_max_time_step,
                input_size=args.dec_embed_dim,
                hidden_size=dec_hidden_dim,
                embed_droprate=args.dec_ed,
                rnn_droprate=args.dec_rd,
                n_layers=args.dec_num_layers,
                rnn_cell=args.rnn_type,
                use_attention=args.use_attention,
                embedding=embed,
                eos_id=vocab.src.eos_id,
                sos_id=vocab.src.sos_id,
            )
        else:
            self.decoder = RNNDecoder(
                vocab=len(vocab.tgt),
                max_len=args.tgt_max_time_step,
                input_size=args.dec_embed_dim,
                hidden_size=dec_hidden_dim,
                embed_droprate=args.dec_ed,
                rnn_droprate=args.dec_rd,
                n_layers=args.dec_num_layers,
                rnn_cell=args.rnn_type,
                use_attention=args.use_attention,
                embedding=embed,
                eos_id=vocab.tgt.eos_id,
                sos_id=vocab.tgt.sos_id,
            )

    def forward(self, hidden, tgt_var):
        decode_init = self.bridger.forward(input_tensor=hidden)

        _loss = -torch.sum(self.decoder.score(
            inputs=tgt_var,
            encoder_outputs=None,
            encoder_hidden=decode_init,
        ))
        return _loss

    def predict(self, hidden):
        decode_init = self.bridger.forward(input_tensor=hidden)

        decoder_outputs, decoder_hidden, ret_dict, enc_states = self.decoder.forward(
            inputs=None,
            encoder_outputs=None,
            encoder_hidden=decode_init,
        )
        result = torch.stack(ret_dict['sequence']).squeeze()
        if result.dim() < 2:
            result = result.unsqueeze(1)
        return result


class BridgeMLP(nn.Module):
    def __init__(self, args, vocab, enc_dim, dec_hidden):
        super().__init__()
        self.bridger = MLPBridger(
            rnn_type=args.rnn_type,
            mapper_type=args.mapper_type,
            encoder_dim=enc_dim,
            encoder_layer=args.enc_num_layers,
            decoder_dim=dec_hidden,
            decoder_layer=args.dec_num_layers,
        )
        if "stack_mlp" not in args:
            self.scorer = nn.Sequential(
                nn.Dropout(args.dec_rd),
                nn.Linear(
                    in_features=dec_hidden * args.dec_num_layers,
                    out_features=len(vocab.src),
                    bias=True
                ),
                nn.LogSoftmax(dim=-1)
            )
        else:
            self.scorer = nn.Sequential(
                nn.Dropout(args.dec_rd),
                nn.Linear(
                    in_features=dec_hidden * args.dec_num_layers,
                    out_features=dec_hidden,
                    bias=True
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=dec_hidden,
                    out_features=len(vocab.src),
                    bias=True
                ),
                nn.LogSoftmax(dim=-1)
            )
        self.semantic_nll = nn.NLLLoss(ignore_index=vocab.src.pad_id)

    def forward(self, hidden, tgt_var):
        batch_size = tgt_var.size(0)
        semantic_decode_init = self.bridger.forward(input_tensor=hidden)
        xx_hidden = semantic_decode_init.permute(1, 0, 2).contiguous().view(batch_size, -1)
        score = self.scorer.forward(input=xx_hidden)
        sem_loss = bag_of_word_loss(score, tgt_var, self.semantic_nll)
        return sem_loss
