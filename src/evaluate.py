from math import ceil

import tensorflow as tf
from tqdm.auto import tqdm

from src.metrics import compute_rouge, compute_f1


def get_eval_predictions(session, model, predictions_fn, loss_fn=None, targets=None, batch_size=None):
    predictions = []
    loss, steps = 0, 0
    n_steps = ceil(len(targets) / batch_size) if targets is not None else None
    tqdm_pbar = tqdm(desc="Evaluation", total=n_steps, leave=False)

    while True:
        try:
            if loss_fn is not None:
                batch_loss, batch_predictions = session.run([loss_fn, predictions_fn])
                batch_predictions = [
                    model.process_prediction({"tokens": prediction}) for prediction in batch_predictions["tokens"]
                ]

                loss += batch_loss
                predictions += batch_predictions
                steps += 1
            else:
                batch_predictions = session.run(predictions_fn)
                batch_predictions = [
                    model.process_prediction({"tokens": prediction}) for prediction in batch_predictions["tokens"]
                ]

                predictions += batch_predictions

            tqdm_pbar.update()
        except tf.errors.OutOfRangeError:
            break

    tqdm_pbar.close()
    return (loss / steps, predictions) if loss_fn is not None else predictions


def evaluate(session, model, iterator, loss, predictions, targets, batch_size):
    session.run([iterator.initializer, tf.tables_initializer()])
    valid_loss, valid_predictions = get_eval_predictions(session, model, predictions, loss, targets, batch_size)
    rouge = compute_rouge(valid_predictions, targets)
    precision, recall, f1 = compute_f1(valid_predictions, targets)
    return valid_loss, rouge, precision, recall, f1
