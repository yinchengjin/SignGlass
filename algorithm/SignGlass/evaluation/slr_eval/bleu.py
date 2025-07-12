import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt', quiet=True)


def compute_bleu(predictions, references):
    """
    predictions: list of predicted sentences (str)
    references: list of reference sentences (str)
    returns: average BLEU score (0~100)
    """
    smoothie = SmoothingFunction().method4
    total_bleu = 0
    count = 0

    for pred, ref in zip(predictions, references):
        ref_tokens = nltk.word_tokenize(ref.lower())
        pred_tokens = nltk.word_tokenize(pred.lower())
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            continue
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        total_bleu += score
        count += 1

    if count == 0:
        return 0.0
    return 100 * total_bleu / count
