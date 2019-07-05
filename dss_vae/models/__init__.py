from dss_vae.vae.ae import AutoEncoder
from dss_vae.vae.disentangle_vae import DisentangleVAE
from dss_vae.vae.syntax_guide_vae import SyntaxGuideVAE
from dss_vae.vae.syntax_vae import SyntaxVAE
from dss_vae.vae.enhance_syntax_vae import EnhanceSyntaxVAE
from dss_vae.vae.vanilla_vae import VanillaVAE
from .nonauto_gen import SelfAttNAG, CondAttNAG, NonAutoReorderGEN
from .seq2seq import BaseSeq2seq
from .syntax_sup_gen import SyntaxSupervisedNAG
from .transformer import Transformer

MODEL_CLS = {
    'Seq2seq': BaseSeq2seq,
    'AutoEncoder': AutoEncoder,
    'VanillaVAE': VanillaVAE,
    'SyntaxGuideVAE': SyntaxGuideVAE,
    'DisentangleVAE': DisentangleVAE,
    'SyntaxVAE': SyntaxVAE,
    'SyntaxVAE2': EnhanceSyntaxVAE,
    'SynSupGEN': SyntaxSupervisedNAG,
    'SelfAttNAG': SelfAttNAG,
    'CondAttNAG': CondAttNAG,
    'Transformer': Transformer,
    'NonAutoReorderGEN': NonAutoReorderGEN
}


def init_create_model(model: str, **kwargs):
    if model not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))

    return MODEL_CLS[model](**kwargs)


def load_static_model(model: str, model_path: str):
    if model not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))

    return MODEL_CLS[model].load(model_path)
