from typing import List, Tuple

from rouge import Rouge

from opengnn import constants


def compute_rouge(predictions, targets):
    predictions = [" ".join(prediction).lower() for prediction in predictions]
    predictions = [prediction if prediction else "EMPTY" for prediction in predictions]
    targets = [" ".join(target).lower() for target in targets]
    targets = [target if target else "EMPTY" for target in targets]
    rouge = Rouge()
    scores = rouge.get_scores(hyps=predictions, refs=targets, avg=True)
    return scores["rouge-2"]["f"]


# F1 implementation with respect to code2seq
# https://github.com/tech-srl/code2seq/blob/af04b4c5ff60c20a34bdc43a52538a69eb8fd9de/common.py#L65-L72
def filter_impossible_names(words: List[str]) -> List[str]:
    return [
        word
        for word in words
        if words not in [constants.UNKNOWN_TOKEN, constants.END_OF_SENTENCE_TOKEN, constants.PADDING_TOKEN]
    ]


# https://github.com/tech-srl/code2seq/blob/af04b4c5ff60c20a34bdc43a52538a69eb8fd9de/model.py#L311
def calculate_f1(true_positive: int, false_positive: int, false_negative: int) -> Tuple[float, float, float]:
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1


# https://github.com/tech-srl/code2seq/blob/af04b4c5ff60c20a34bdc43a52538a69eb8fd9de/model.py#L268
def compute_f1(predictions: List[List[str]], targets: List[List[str]]) -> Tuple[float, float, float]:
    true_positive, false_positive, false_negative = 0, 0, 0
    for prediction, target in zip(predictions, targets):
        filtered_predicted_names = filter_impossible_names(prediction)
        filtered_original_subtokens = filter_impossible_names(target)

        if "".join(filtered_original_subtokens) == "".join(filtered_predicted_names):
            true_positive += len(filtered_original_subtokens)
            continue

        for sub_token in filtered_predicted_names:
            if sub_token in filtered_original_subtokens:
                true_positive += 1
            else:
                false_positive += 1

        for sub_token in filtered_original_subtokens:
            if sub_token not in filtered_predicted_names:
                false_negative += 1

    return calculate_f1(true_positive, false_positive, false_negative)
