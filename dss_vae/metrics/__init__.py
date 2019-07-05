from .base_evaluator import UniversalBleuEvaluator
from .paraphrase_evaluator import ParaphraseEvaluator
from .reorder_evaluator import ReorderEvaluator
from .syntax_vae_evaluator import SyntaxEvaluator
from .translation_evaluator import TranslationEvaluator
from .vae_metrics import SyntaxVaeEvaluator, VaeEvaluator

eval_dict = {
    'bleu': UniversalBleuEvaluator,
    'para': ParaphraseEvaluator,
    'mt': TranslationEvaluator,
    'reorder': ReorderEvaluator,
    'svae': SyntaxVaeEvaluator,
    'vae': VaeEvaluator,
    'svae2': SyntaxEvaluator

}


def get_evaluator(eval_choice, **kwargs):
    """
    Args:
        eval_choice: select:[bleu,para]
        **kwargs: for bleu, need model,eval_set,out_dir,batch_size, write_down
        for para, need model,eval_set,out_dir,batch_size, write_down

    Returns:

    """

    return eval_dict[eval_choice.lower()](
        **kwargs
    )
