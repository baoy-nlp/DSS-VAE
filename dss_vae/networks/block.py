import torch.nn as nn

from .sublayers import MultiHeadedAttention
from .sublayers import PositionwiseFeedForward


class MultiAttBlock(nn.Module):

    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1):
        super(MultiAttBlock, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)

        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        input_norm = self.layer_norm(enc_input)
        context, _, _ = self.slf_attn(input_norm, input_norm, input_norm, slf_attn_mask)
        out = self.dropout(context) + enc_input

        return self.pos_ffn(out)


class MultiAttGroup(nn.Module):
    def __init__(self, n_layers, hidden_size, inner_hidden, n_head, block_dropout, dim_per_head=None):
        super().__init__()
        self.num_layers = n_layers
        self.block_stack = nn.ModuleList(
            [MultiAttBlock(d_model=hidden_size, d_inner_hid=inner_hidden, n_head=n_head, dropout=block_dropout,
                           dim_per_head=dim_per_head)
             for _ in range(n_layers)])
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, out, enc_slf_attn_mask=None, ret_list=False):
        ret = []
        for i in range(self.num_layers):
            out = self.block_stack[i](out, enc_slf_attn_mask)
            ret.append(out)

        out = self.layer_norm(out)

        if ret_list:
            ret[-1] = out
            return ret

        return out

    @property
    def out_dim(self):
        return self.hidden_size


class ConditionAttBlock(nn.Module):
    """
    Decoder block.
    """

    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1):
        super(ConditionAttBlock, self).__init__()

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)
        self.ctx_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)
        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def compute_cache(self, enc_output):
        return self.ctx_attn.compute_cache(enc_output, enc_output)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None,
                enc_attn_cache=None, self_attn_cache=None):
        # Args Checks
        input_batch, input_len, _ = dec_input.size()

        contxt_batch, contxt_len, _ = enc_output.size()

        input_norm = self.layer_norm_1(dec_input)
        all_input = input_norm

        query, _, self_attn_cache = self.slf_attn(
            all_input, all_input, input_norm,
            mask=slf_attn_mask, self_attn_cache=self_attn_cache
        )

        query = self.dropout(query) + dec_input

        query_norm = self.layer_norm_2(query)
        mid, attn, enc_attn_cache = self.ctx_attn(
            enc_output, enc_output, query_norm,
            mask=dec_enc_attn_mask, enc_attn_cache=enc_attn_cache
        )

        output = self.pos_ffn(self.dropout(mid) + query)

        return output, attn, self_attn_cache, enc_attn_cache
