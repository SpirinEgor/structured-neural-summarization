import os

import tensorflow as tf


def get_iterator_from_input_fn(input_fn):
    with tf.device("/cpu:0"):
        return input_fn().make_initializable_iterator()


class Summary(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, model_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(model_dir)

    def scalar(self, tag, value, step, family=None):
        """Log a scalar variable.

        Parameter
        ----------
        tag: basestring
            Name of the scalar
        value
        step: int
            training iteration
        """
        tag = os.path.join(family, tag) if family is not None else tag
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
