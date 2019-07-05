import torch
import torch.nn as nn
import torch.nn.functional as F

from dss_vae.networks.sublayers import ConditionAttBlock
from dss_vae.networks.sublayers import MultiHeadedAttention
from dss_vae.networks.sublayers import PositionwiseFeedForward
from dss_vae.utils.loss_funcs import pos_loss
from dss_vae.utils.position_tools import get_position_encoding_expectation
from dss_vae.utils.position_tools import init_position_embedding
from dss_vae.utils.position_tools import positional_encodings_like
from dss_vae.utils.position_tools import sequential_index
from dss_vae.utils.position_tools import to_index
from dss_vae.utils.tensor_ops import self_concatenate


class PositionFeatExtractor(nn.Module):
    """
    FEAT Selection:
    Feat 1: rnn -> mlp -> tgt-enc-att ->
    Feat 2: self-att -> tgt-enc-att ->
    Feat 3: rnn -> mlp -> self-att -> tgt-enc-att ->
    """

    @property
    def desc(self):
        if self.feat_type == 1:
            return "FEAT 1: TGT -> RNN -> MLP -> TGT-ENC-ATT"
        elif self.feat_type == 2:
            return "FEAT 2: TGT -> Self-ATT -> TGT-ENC-ATT"
        elif self.feat_type == 3:
            return "FEAT 3: TGT -> RNN -> MLP ->  Self-ATT -> TGT-ENC-ATT"
        else:
            raise RuntimeError("illegal selection")

    def __init__(self, model_dim, inner_dim, dropout=0.1, head_num=8, att_num_layer=1, rnn_num_layer=2, feat_type=1):
        super().__init__()
        self.feat_type = feat_type
        self.att_num_layer = att_num_layer
        if self.feat_type == 1:
            self.feat_rnn = nn.LSTM(model_dim, model_dim, batch_first=True, bidirectional=True,
                                    num_layers=rnn_num_layer, dropout=dropout)
            self.feat_mlp = nn.Sequential(
                nn.Linear(2 * model_dim, model_dim),
                nn.Dropout(0.1),
                nn.LayerNorm(model_dim)
            )
            self.att_to_enc = MultiHeadedAttention(
                model_dim=model_dim,
                head_count=head_num,
            )
            self.dropout = nn.Dropout(dropout)
            self.pos_ffn_feature = PositionwiseFeedForward(model_dim, inner_dim)
        if self.feat_type == 2:
            self.feature_extractor = nn.ModuleList([ConditionAttBlock(
                d_model=model_dim,
                d_inner_hid=inner_dim,
                n_head=head_num,
                dropout=dropout,
                dim_per_head=None
            ) for _ in range(att_num_layer)])
            self.out_layer_norm = nn.LayerNorm(model_dim)

        if self.feat_type == 3:
            self.feat_rnn = nn.LSTM(model_dim, model_dim, batch_first=True, bidirectional=True,
                                    num_layers=rnn_num_layer, dropout=dropout)
            self.feat_mlp = nn.Sequential(
                nn.Linear(2 * model_dim, model_dim),
                nn.Dropout(0.1),
                nn.LayerNorm(model_dim)
            )
            self.feature_extractor = nn.ModuleList([ConditionAttBlock(
                d_model=model_dim,
                d_inner_hid=inner_dim,
                n_head=head_num,
                dropout=dropout,
                dim_per_head=None
            ) for _ in range(att_num_layer)])
            self.out_layer_norm = nn.LayerNorm(model_dim)

    def forward(self, dec_inputs, enc_outputs=None, enc_mask=None):
        """

        Args:
            dec_inputs: [batch,out,hid]
            enc_outputs: [batch,seq,hid]
            enc_mask: [batch,seq]

        Returns:

        """
        if enc_mask is not None:
            batch_size, seq_len, _ = enc_outputs.size()
            out_len = dec_inputs.size(1)
            dec_enc_mask = enc_mask.unsqueeze(1).expand(batch_size, out_len, seq_len)
        else:
            dec_enc_mask = None

        if self.feat_type == 1:
            dec_query, _ = self.feat_rnn(dec_inputs)
            dec_query = self.feat_mlp(dec_query)
            mid, _, _ = self.att_to_enc(
                key=enc_outputs,
                value=enc_outputs,
                query=dec_query,
                mask=dec_enc_mask
            )
            position_feature = self.pos_ffn_feature(self.dropout(mid) + dec_query)
            return position_feature
        if self.feat_type == 2:
            output = dec_inputs
            for i in range(self.att_num_layer):
                output, attn, self_attn_cache, enc_attn_cache = self.feature_extractor[i](
                    dec_input=output,
                    enc_output=enc_outputs,
                    dec_enc_attn_mask=dec_enc_mask,
                )
            if self.att_num_layer > 1:
                output = self.out_layer_norm(output)
            return output
        if self.feat_type == 3:
            dec_query, _ = self.feat_rnn(dec_inputs)
            dec_query = self.feat_mlp(dec_query)
            output = dec_query
            for i in range(self.att_num_layer):
                output, attn, self_attn_cache, enc_attn_cache = self.feature_extractor[i](
                    dec_input=output,
                    enc_output=enc_outputs,
                    dec_enc_attn_mask=dec_enc_mask,
                )
            if self.att_num_layer > 1:
                output = self.out_layer_norm(output)
            return output


class PositionPredictor(nn.Module):
    """
    Position Predictor:
    Pred 1: absolute position classifier [0,...,M]
    Pred 2: relative position regression [0,M]
    """

    @property
    def desc(self):
        if self.pred_type == 1:
            return "PRED 1: Absolute Position Classifier"
        elif self.pred_type == 2:
            return "PRED 2: Relative Position Value Regression"
        else:
            raise RuntimeError("illegal selection")

    def __init__(self, model_dim, max_len, dropout=0.1, pred_type=1, use_rank=True, use_mse=False, use_dst=False):
        super().__init__()
        self.pred_type = pred_type
        self.use_rank = use_rank
        self.use_mse = use_mse
        self.use_dst = use_dst

        if self.pred_type == 1:
            self.predictor = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model_dim, model_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, max_len, bias=True)
            )
            self.loss_info = "Classifier Loss"
        if self.pred_type == 2:
            self.predictor = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model_dim, model_dim // 2, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim // 2, 1, bias=True)
            )
            self.loss_info = "Rank: {}\tMSE: {}\tDST: {}".format(
                self.use_rank,
                self.use_mse,
                self.use_dst
            )

    def forward(self, position_feature, pos_ref=None, pos_ref_mask=None):
        pos_logits = self.predictor.forward(input=position_feature)
        batch_size, seq_len, vocab_size = pos_logits.size()
        if self.pred_type == 1:
            pos_logits = pos_logits.view(-1, vocab_size)
            gumbel_out = F.gumbel_softmax(logits=pos_logits, hard=True).view(batch_size, seq_len, vocab_size)
            position_select = to_index(st_gumbel_out=gumbel_out).view(-1, seq_len)
            pos_logits = pos_logits.view(batch_size, -1, vocab_size)
        else:
            relative_pos = pos_logits.view(batch_size, -1)
            position_select = torch.gt(relative_pos.unsqueeze(-1) - relative_pos.unsqueeze(-2), 0).sum(dim=-1)
            pos_logits = relative_pos
        if pos_ref is not None:
            if self.pred_type == 1:
                pos_logits = pos_logits.contiguous().transpose(1, 0).contiguous()
                pos_ref = pos_ref.contiguous().transpose(1, 0)
                flattened_mask = pos_ref_mask.transpose(1, 0).contiguous().view(-1)
                log_probs = F.log_softmax(pos_logits.view(-1, vocab_size).contiguous(), dim=-1)
                flattened_tgt_pos = pos_ref.contiguous().view(-1)
                pos_log_probs = torch.gather(log_probs, 1, flattened_tgt_pos.unsqueeze(1)).squeeze(1)
                pos_log_probs = pos_log_probs * flattened_mask.float()
                ret_loss = pos_log_probs.view(-1, batch_size).sum(dim=0)
            else:
                ret_loss = -pos_loss(inputs=pos_logits, target=pos_ref, mask=pos_ref_mask, use_rank=self.use_rank,
                                     use_mse=self.use_mse, use_dst=self.use_dst)
        else:
            ret_loss = 0.0
        return {
            'pos': position_select,
            "loss": ret_loss,
            'logits': pos_logits,
        }


class PositionModel(nn.Module):
    """
    Relative Position Prediction
    """

    def __init__(self, model_dim, head_num, max_len, use_pos_exp, use_pos_pred, use_st_gumbel,
                 pos_feat=1, pos_pred=1,
                 use_rank=True,
                 use_mse=False,
                 use_dst=False,
                 att_num_layer=1,
                 rnn_num_layer=2,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        self.use_pos_pred = use_pos_pred
        self.embed_table = init_position_embedding(
            max_len=max_len,
            hidden_size=model_dim,
            is_matrix=use_pos_exp
        )
        if self.use_pos_pred:
            self.feature_extractor = PositionFeatExtractor(
                model_dim=model_dim,
                inner_dim=2 * model_dim,
                dropout=dropout,
                head_num=head_num,
                att_num_layer=att_num_layer,
                rnn_num_layer=rnn_num_layer,
                feat_type=pos_feat,
            )
            self.position_predictor = PositionPredictor(
                model_dim=model_dim,
                max_len=max_len,
                dropout=dropout,
                pred_type=pos_pred,
                use_rank=use_rank,
                use_mse=use_mse,
                use_dst=use_dst,
            )
            print('Feature Info: {}\nPrediction Info: {}\nLoss Info: {}'.format(
                self.feature_extractor.desc,
                self.position_predictor.desc,
                self.position_predictor.loss_info))

    def forward(self, dec_inputs, enc_outputs=None, enc_mask=None, return_pos_prob=False, pos_ref=None, pos_mask=None):
        ret = {
            'out': None,
            'logits': None,
            'loss': 0.0,
            'mask': None
        }
        if not self.use_pos_pred or enc_outputs is None:
            bsize, out_len, _ = dec_inputs.size()
            position_select = sequential_index(batch_size=bsize, n_class=out_len)
            # position_encoding = self.embed_table(position_select)
            position_encoding = positional_encodings_like(x=dec_inputs, use_cuda=torch.cuda.is_available())
        else:
            position_feature = self.feature_extractor.forward(
                dec_inputs=dec_inputs,
                enc_outputs=enc_outputs,
                enc_mask=enc_mask,
            )
            pred_ret = self.position_predictor.forward(
                position_feature=position_feature,
                pos_ref=pos_ref,
                pos_ref_mask=pos_mask,
            )
            position_select = pred_ret['pos']
            position_encoding = self.embed_table(position_select)
            ret['logits'] = pred_ret['logits']
            ret['loss'] = pred_ret['loss']
        ret['pos'] = position_select
        ret['mask'] = torch.gt(position_select.unsqueeze(-2) - position_select.unsqueeze(-1), 0)
        if pos_ref is not None:
            position_encoding = self.embed_table(pos_ref)
            ret['mask'] = torch.gt(pos_ref.unsqueeze(-2) - pos_ref.unsqueeze(-1), 0)
        ret['out'] = dec_inputs + position_encoding
        return ret


class AbsolutelyPosition(nn.Module):
    def __init__(self, model_dim, head_num, max_len, use_pos_exp, use_pos_pred, use_st_gumbel, **kwargs):
        super().__init__()
        print("Using Absolutely Position")

        self.model_dim = model_dim
        self.max_len = max_len
        self.use_pos_exp = use_pos_exp
        self.use_pos_pred = use_pos_pred
        self.use_st_gumbel = use_st_gumbel

        self.embed_table = init_position_embedding(
            max_len=max_len,
            hidden_size=model_dim,
            is_matrix=use_pos_exp
        )

        # embedding or position matrix

        if self.use_pos_pred:
            self.feature_extractor = ConditionAttBlock(
                d_model=model_dim,
                d_inner_hid=2 * model_dim,
                n_head=head_num,
                dim_per_head=None
            )

            self.predictor = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(model_dim, model_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(model_dim, max_len, bias=True)
            )

    def forward(self, dec_inputs, enc_outputs=None, enc_mask=None, return_pos_prob=False, true_position=None):
        ret = {
            'out': None,
            'logits': None,
            'mask': None
        }
        if not self.use_pos_pred or enc_outputs is None:
            bsize, seq_len, _ = dec_inputs.size()
            position_select = sequential_index(batch_size=bsize, n_class=seq_len)
            position_encoding = self.embed_table(position_select)
        else:
            position_feature, _, _, _ = self.feature_extractor.forward(
                dec_input=dec_inputs,
                enc_output=enc_outputs,
                dec_enc_attn_mask=enc_mask
            )

            logits = self.predictor.forward(input=position_feature)
            bsize, seq_len, v_size = logits.size()
            if self.use_pos_exp:
                prob = F.softmax(logits, dim=-1)
                position_encoding = get_position_encoding_expectation(
                    pos_prob=prob,
                    embed_matrix=self.embed_table
                )
                position_select = to_index(st_gumbel_out=logits)
            else:
                logits = logits.view(-1, v_size)
                gumbel_out = F.gumbel_softmax(logits=logits, hard=self.use_st_gumbel).view(bsize, seq_len, v_size)
                position_select = to_index(st_gumbel_out=gumbel_out)
                position_encoding = self.embed_table(position_select)

            ret['logits'] = logits.view(bsize, -1, v_size)
        ret['pos'] = position_select
        ret['mask'] = torch.gt(position_select.unsqueeze(-2) - position_select.unsqueeze(-1), 0)
        if true_position is not None:
            position_encoding = self.embed_table(true_position)
            ret['mask'] = torch.gt(true_position.unsqueeze(-2) - true_position.unsqueeze(-1), 0)
        ret['out'] = dec_inputs + position_encoding
        return ret


class RelativePosition(nn.Module):
    def __init__(self, model_dim, head_num, max_len, use_pos_exp, use_pos_pred, use_st_gumbel, **kwargs):
        super().__init__()
        print("Using Relative Position")

        self.model_dim = model_dim
        self.max_len = max_len
        self.use_pos_exp = False
        self.use_pos_pred = use_pos_pred
        self.use_st_gumbel = use_st_gumbel

        self.embed_table = init_position_embedding(
            max_len=max_len,
            hidden_size=model_dim,
            is_matrix=use_pos_exp
        )

        self.use_rnn = True
        # embedding or position matrix

        if self.use_pos_pred:
            if self.use_rnn:
                self.feat_rnn = nn.LSTM(model_dim, model_dim, batch_first=True, bidirectional=True,
                                        num_layers=2, dropout=0.1)
                self.feat_mlp = nn.Sequential(
                    nn.Linear(2 * model_dim, model_dim),
                    nn.Dropout(0.1),
                    nn.LayerNorm(model_dim)
                )

            else:
                self.gen_query = nn.Sequential(
                    PositionwiseFeedForward(size=2 * model_dim, hidden_size=4 * model_dim),
                    nn.Linear(2 * model_dim, model_dim),
                    nn.Dropout(0.1),
                    nn.LayerNorm(model_dim),
                )

            self.att_to_enc = MultiHeadedAttention(
                model_dim=model_dim,
                head_count=head_num,
            )
            self.dropout = nn.Dropout(0.1)
            self.pos_ffn_feature = PositionwiseFeedForward(model_dim, 2 * model_dim)

            self.predictor = nn.Sequential(
                # nn.Dropout(0.1),
                # nn.Linear(model_dim, model_dim, bias=True),
                # nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(model_dim, 1, bias=True)
            )

    def forward(self, dec_inputs, enc_outputs=None, enc_mask=None, return_pos_prob=False, true_position=None):
        ret = {
            'out': None,
            'logits': None,
            'mask': None
        }
        bsize, seq_len, model_dim = dec_inputs.size()
        if not self.use_pos_pred or enc_outputs is None:
            position_select = sequential_index(batch_size=bsize, n_class=seq_len)
            position_encoding = self.embed_table(position_select)
        else:
            if not self.use_rnn:
                position_dec_feature = self_concatenate(dec_inputs).view(bsize, -1, model_dim * 2)
                assert position_dec_feature.size(1) == seq_len * seq_len, "dimension mismatch"
                dec_query = self.gen_query(position_dec_feature)  # [batch_size,seq_len,seq_len,hidden_size]

                dec_query = F.max_pool1d(input=dec_query.transpose(-2, -1), kernel_size=seq_len).view(bsize, seq_len,
                                                                                                      -1)
            else:
                dec_query, _ = self.feat_rnn(dec_inputs)
                dec_query = self.feat_mlp(dec_query)
            mid, _, _ = self.att_to_enc(
                key=enc_outputs,
                value=enc_outputs,
                query=dec_query,
                mask=enc_mask
            )
            position_feature = self.pos_ffn_feature(self.dropout(mid) + dec_query)

            # position_enc_feature = self_concatenate(mid)
            # position_feature = torch.cat([position_enc_feature, position_dec_feature], dim=-1)
            relative_pos = self.predictor.forward(input=position_feature).view(bsize, -1)  # [batch_size,seq_len]

            position_select = torch.gt(relative_pos.unsqueeze(-1) - relative_pos.unsqueeze(-2), 0).sum(dim=-1)
            position_encoding = self.embed_table(position_select)

            ret['logits'] = relative_pos.view(bsize, -1)
        ret['pos'] = position_select
        ret['mask'] = torch.gt(position_select.unsqueeze(-2) - position_select.unsqueeze(-1), 0).contiguous()

        if true_position is not None:
            position_encoding = self.embed_table(true_position)
            ret['mask'] = torch.gt(true_position.unsqueeze(-2) - true_position.unsqueeze(-1), 0)

        ret['out'] = dec_inputs + position_encoding
        return ret


def get_position_select(pos_type, **kwargs):
    """

    Args:
        pos_type:
        **kwargs:

    Returns:

    """
    print("Position Select:\t{}".format(pos_type.upper()))
    return {
        'relative': RelativePosition,
        'absolute': AbsolutelyPosition,
        'position': PositionModel
    }[pos_type](**kwargs)
