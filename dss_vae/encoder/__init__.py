from .attentive_encoder import TransformerEncoder
from .nonauto_encoder import NonAutoATTEncoder
from .rnn_encoder import RNNEncoder

MODEL_CLS = {
    'rnn': RNNEncoder,
    'att': TransformerEncoder,
    'non-att': NonAutoATTEncoder
}


def get_encoder(model: str, **kwargs):
    if model.lower() not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))

    args = kwargs['args']
    print("GET:\t{} Encoder".format(model.upper()))
    return MODEL_CLS[model.lower()](
        max_len=args.src_max_time_step,
        input_size=args.enc_embed_dim,
        hidden_size=args.enc_hidden_dim,
        embed_droprate=args.enc_ed,
        rnn_droprate=args.enc_rd,
        n_layers=args.enc_num_layers,
        bidirectional=args.bidirectional,
        rnn_cell=args.rnn_type,
        n_head=args.enc_head,
        inner_hidden=args.enc_inner_hidden,
        embed_dropout=args.enc_ed,
        block_dropout=args.enc_rd,
        variable_lengths=True,
        dim_per_head=None,
        **kwargs
    )
