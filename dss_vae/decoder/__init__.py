from .att_decoder import TransformerDecoder
from .att_nonauto_decoder import CondAttNAD
from .att_nonauto_decoder import SelfAttNAD
from .nonauto_decoder import NonAutoDecoder
from .rnn_decoder import RNNDecoder
from .synsup_nonauto_decoder import SynSupervisedNAD

MODEL_CLS = {
    'rnn': RNNDecoder,
    'mat': NonAutoDecoder,
    'self-att': SelfAttNAD,
    'cond-att': CondAttNAD,
    'syn-sup': SynSupervisedNAD
}


def get_decoder(model: str, **kwargs):
    if model.lower() not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))
    args = kwargs['args']
    print("GET:\t{} Decoder".format(model.upper()))
    return MODEL_CLS[model.lower()](
        max_len=args.tgt_max_time_step,
        n_layers=args.dec_num_layers,
        n_head=args.dec_head,
        inner_dim=args.dec_inner_hidden,
        block_dropout=args.dec_rd,
        out_dropout=args.dropo,
        dim_per_head=None,
        use_cuda=args.cuda,
        bow=args.sequence_bow_loss if "layer_bow_loss" in args else False,
        share_predictor=args.share_predictor if "share_predictor" in args else False,
        **kwargs
    )
