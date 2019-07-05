import torch
import torch.nn as nn

from dss_vae.utils.nn_funcs import positional_encodings_from_range
from .sublayers import MultiHeadedAttention


def reflect(x):
    return x


class MLPBridger(nn.Module):
    def __init__(self, rnn_type, mapper_type, encoder_dim, encoder_layer, decoder_dim, decoder_layer):
        super(MLPBridger, self).__init__()
        self.rnn_type = rnn_type
        self.parameter_type = mapper_type
        self.input_dim = encoder_dim * encoder_layer
        self.output_dim = decoder_dim * decoder_layer
        self.decoder_dim = decoder_dim
        self.decoder_layer = decoder_layer

        if self.parameter_type == "mapping":
            self.mapper = nn.Linear(in_features=self.input_dim, out_features=self.output_dim)
        elif self.input_dim == self.output_dim:
            self.mapper = reflect

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: [layers,batch,direction * hidden_size]
        """
        batch_size = input_tensor.size(1)
        reset_tensor = input_tensor.permute(1, 0, 2).contiguous()
        reset_tensor = reset_tensor.view(batch_size, -1)
        assert reset_tensor.size(1) == self.input_dim, "bridge dim is not right"
        output_tensor = self.mapper(reset_tensor).view(-1, self.decoder_layer, self.decoder_dim)
        return output_tensor.permute(1, 0, 2).contiguous()


class BaseBridger(nn.Module):

    def __init__(self, name='BaseBridger', **kwargs):
        super().__init__()
        print("Init:\t", name)

    def forward(self, encoder_output, **kwargs):
        return {
            'out': encoder_output,
            'mask': None
        }


class MatrixBridger(BaseBridger):
    """
    Input:
        z: batch_size,hidden
    Modules:
        mapper_k: z->hidden->k
        mapper_v: z->hidden->v
        k * v
    """

    def __init__(self, input_dim, hidden_dim, k_dim, v_dim, dropout=0.1, **kwargs):
        super(MatrixBridger, self).__init__(name="Matrix Bridger", **kwargs)
        self.k_mapper = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, k_dim, bias=True)
        )
        self.v_mapper = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, v_dim, bias=True),
        )
        self.hidden = v_dim

    def forward(self, encoder_output, **kwargs):
        """
        Mapper the inputs to a matrix
        Args:
            encoder_output:(Tensor: batch_size, hidden) encoder output or latent variable.
        """
        encoder_output = encoder_output.mean(dim=1)
        k_vec = self.k_mapper.forward(encoder_output)
        v_vec = self.v_mapper.forward(encoder_output)
        batch_size = encoder_output.size(0)
        post_k = k_vec.contiguous().view(batch_size, -1, 1)
        post_v = v_vec.contiguous().view(batch_size, 1, -1)

        return {
            "out": torch.bmm(post_k, post_v),
            "mask": None
        }

    @property
    def out_dim(self):
        return self.hidden


class SelfATTBridger(BaseBridger):
    def __init__(self, input_dim, head_num, k_dim, dropout=0.1, **kwargs):
        super().__init__(name="Self-Attention Bridger", **kwargs)
        self.attn_mapper = MultiHeadedAttention(input_dim, head_num, dropout=dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dec_max_len = k_dim
        self.hidden = input_dim

    def forward(self, encoder_output, encoder_mask=None):
        """

        Args:
            encoder_output:
            encoder_mask:

        Returns:

        """
        batch_size, seq_len, hidden = encoder_output.size()
        # out_len = seq_len * 2 if self.dec_max_len > (seq_len * 2) else self.dec_max_len
        out_len = self.dec_max_len
        dec_enc_mask = encoder_mask.unsqueeze(1).expand(batch_size, out_len, seq_len)
        query_base = positional_encodings_from_range(batch_size=batch_size, seq_len=out_len, hidden=hidden,
                                                     use_cuda=torch.cuda.is_available())
        query_norm = self.layer_norm(query_base)
        input_norm = self.layer_norm(encoder_output)
        context, _, _ = self.attn_mapper(key=input_norm, value=input_norm, query=query_norm, mask=dec_enc_mask)
        return {
            "out": context,
            "mask": dec_enc_mask
        }

    @property
    def out_dim(self):
        return self.hidden


def get_bridger(model_cls, **kwargs):
    model_cls_dict = {
        "base": BaseBridger,
        "mat": MatrixBridger,
        "att": SelfATTBridger
    }
    return model_cls_dict[model_cls](**kwargs)
