import json
import os
from math import ceil

import tensorflow as tf
from rouge import Rouge
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.python import debug as tf_debug
from tqdm.auto import tqdm

from opengnn.decoders.sequence import RNNDecoder, HybridPointerDecoder
from opengnn.encoders import GGNNEncoder, SequencedGraphEncoder
from opengnn.inputters import GraphEmbedder
from opengnn.inputters import SequencedGraphInputter
from opengnn.inputters import TokenEmbedder, CopyingTokenEmbedder
from opengnn.models import GraphToSequence, SequencedGraphToSequence
from opengnn.utils import CoverageBahdanauAttention, read_jsonl_gz_file
from training_arguments import configure_arg_parser

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    parser = configure_arg_parser()
    args = parser.parse_args()

    model = build_model(args)

    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.model_name

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        os.makedirs(os.path.join(args.checkpoint_dir, "valid"))
    elif not os.path.exists(os.path.join(args.checkpoint_dir, "valid")):
        os.makedirs(os.path.join(args.checkpoint_dir, "valid"))

    if args.infer_source_file is not None:
        infer(model, args)
    else:
        train_and_eval(model, args)


def train_and_eval(model, args):
    tf.set_random_seed(args.seed)

    optimizer = build_optimizer(args)
    metadata = build_metadata(args)
    config = build_config(args)
    params = build_params(args)

    train_input_fn = model.input_fn(
        mode=tf.estimator.ModeKeys.TRAIN,
        batch_size=args.batch_size,
        metadata=metadata,
        features_file=args.train_source_file,
        labels_file=args.train_target_file,
        features_bucket_width=args.bucket_width,
        sample_buffer_size=args.sample_buffer_size,
    )
    valid_input_fn = model.input_fn(
        mode=tf.estimator.ModeKeys.EVAL,
        batch_size=args.batch_size,
        metadata=metadata,
        features_file=args.valid_source_file,
        labels_file=args.valid_target_file,
    )
    valid_targets = read_jsonl_gz_file(args.valid_target_file)

    train_iterator = get_iterator_from_input_fn(train_input_fn)
    valid_iterator = get_iterator_from_input_fn(valid_input_fn)
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=False)
    )
    with tf.Session(config=session_config) as session:
        if args.debug_mode:
            session = tf_debug.LocalCLIDebugWrapperSession(session, dump_root="~/Downloads/tf-debug")

        # build train graph, loss and optimization ops
        features, labels = train_iterator.get_next()
        with tf.variable_scope(args.model_name):
            outputs, _ = model(features, labels, tf.estimator.ModeKeys.TRAIN, params, config)
            train_loss, train_tb_loss = model.compute_loss(
                features, labels, outputs, params, tf.estimator.ModeKeys.TRAIN
            )

        train_op = optimizer(train_loss)

        # build eval graph, loss and prediction ops
        features, labels = valid_iterator.get_next()
        with tf.variable_scope(args.model_name, reuse=True):
            outputs, predictions = model(features, labels, tf.estimator.ModeKeys.EVAL, params, config)
            _, valid_tb_loss = model.compute_loss(features, labels, outputs, params, tf.estimator.ModeKeys.EVAL)

        global_step = tf.train.get_global_step()

        best_loss = 0
        worse_epochs = 0

        saver = tf.train.Saver(max_to_keep=100)
        train_summary = Summary(args.checkpoint_dir)
        valid_summary = Summary(os.path.join(args.checkpoint_dir, "valid"))
        # TODO: Initialize tables some other way
        session.run([train_iterator.initializer, tf.tables_initializer()])

        # check if we are restarting a run
        latest_checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint is not None:
            saver.restore(session, latest_checkpoint)
        else:
            session.run(tf.global_variables_initializer())

        initial_step = session.run(global_step)

        window_loss = 0
        window_steps = 0
        tqdm_range = tqdm(range(initial_step + 1, args.train_steps + 1), desc="Training")
        for train_step in tqdm_range:
            step_loss, _ = session.run([train_tb_loss, train_op])
            window_loss += step_loss
            window_steps += 1

            # check if in logging schedule
            if train_step % args.logging_window == 0:
                cur_loss = window_loss / window_steps
                train_summary.scalar("loss", cur_loss, train_step)
                tqdm_range.set_postfix({f"loss (step {train_step})": cur_loss})
                # print("step %d, train loss: %0.2f" % (train_step, cur_loss)))
                window_loss = 0
                window_steps = 0

            # after training, do evaluation if on schedule
            if train_step % args.validation_interval == 0:
                valid_loss, valid_rouge = evaluate(
                    session, model, valid_iterator, valid_tb_loss, predictions, valid_targets, args.batch_size
                )
                tqdm.write("eval loss: %0.2f, eval rouge: %0.2f" % (valid_loss, valid_rouge))
                valid_summary.scalar("loss", valid_loss, train_step)
                valid_summary.scalar("rouge", valid_rouge, train_step)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    worse_epochs = 0
                else:
                    worse_epochs += 1

                tqdm.write("saving current model...")
                saver.save(session, os.path.join(args.checkpoint_dir, f"{args.model_name}.ckpt"), global_step)

                # and stop training if triggered patience
                if worse_epochs >= args.patience:
                    tqdm.write("early stopping triggered...")
                    break


def evaluate(session, model, iterator, loss, predictions, targets, batch_size):
    """ """
    valid_loss = 0
    valid_steps = 0

    valid_predictions = []

    session.run([iterator.initializer, tf.tables_initializer()])
    n_steps = ceil(len(targets) / batch_size)
    tqdm_pbar = tqdm(desc="Evaluation", total=n_steps, leave=False)
    while True:
        try:
            batch_loss, batch_predictions = session.run([loss, predictions])
            batch_predictions = [
                model.process_prediction({"tokens": prediction}) for prediction in batch_predictions["tokens"]
            ]

            valid_loss += batch_loss
            valid_predictions = valid_predictions + batch_predictions
            valid_steps += 1
            tqdm_pbar.update()
        except tf.errors.OutOfRangeError:
            break
    tqdm_pbar.close()

    loss = valid_loss / valid_steps
    rouge = compute_rouge(valid_predictions, targets)
    return loss, rouge


def get_iterator_from_input_fn(input_fn):
    with tf.device("/cpu:0"):
        return input_fn().make_initializable_iterator()


def build_model(args):
    """"""
    if args.coverage_layer:
        attention_layer = CoverageBahdanauAttention
    else:
        attention_layer = BahdanauAttention

    if args.copy_attention:
        node_embedder = CopyingTokenEmbedder(
            vocabulary_file_key="node_vocabulary",
            output_vocabulary_file_key="target_vocabulary",
            embedding_size=args.source_embeddings_size,
            dropout_rate=args.embeddings_dropout,
            lowercase=not args.case_sensitive,
        )
        target_inputter = CopyingTokenEmbedder(
            vocabulary_file_key="target_vocabulary",
            input_tokens_fn=lambda data: data["labels"],
            embedding_size=args.target_embeddings_size,
            dropout_rate=args.embeddings_dropout,
            truncated_sentence_size=args.truncated_target_size,
        )
        decoder = HybridPointerDecoder(
            num_units=args.rnn_hidden_size,
            num_layers=args.rnn_num_layers,
            output_dropout_rate=args.rnn_hidden_dropout,
            attention_mechanism_fn=attention_layer,
            coverage_loss_lambda=args.coverage_loss,
            copy_state=True,
        )
    else:
        node_embedder = TokenEmbedder(
            vocabulary_file_key="node_vocabulary",
            embedding_size=args.source_embeddings_size,
            dropout_rate=args.embeddings_dropout,
            lowercase=not args.case_sensitive,
        )
        target_inputter = TokenEmbedder(
            vocabulary_file_key="target_vocabulary",
            embedding_size=args.target_embeddings_size,
            dropout_rate=args.embeddings_dropout,
            truncated_sentence_size=args.truncated_target_size,
        )
        decoder = RNNDecoder(
            num_units=args.rnn_hidden_size,
            num_layers=args.rnn_num_layers,
            output_dropout_rate=args.rnn_hidden_dropout,
            attention_mechanism_fn=attention_layer,
            coverage_loss_lambda=args.coverage_loss,
            copy_state=True,
        )

    if args.only_graph_encoder:
        model = GraphToSequence(
            source_inputter=GraphEmbedder(edge_vocabulary_file_key="edge_vocabulary", node_embedder=node_embedder),
            target_inputter=target_inputter,
            encoder=GGNNEncoder(
                num_timesteps=[args.ggnn_timesteps_per_layer for _ in range(args.ggnn_num_layers)],
                node_feature_size=args.node_features_size,
                gru_dropout_rate=args.node_features_dropout,
            ),
            decoder=decoder,
            name=args.model_name,
        )
    else:
        model = SequencedGraphToSequence(
            source_inputter=SequencedGraphInputter(
                graph_inputter=GraphEmbedder(edge_vocabulary_file_key="edge_vocabulary", node_embedder=node_embedder),
                truncated_sequence_size=args.truncated_source_size,
            ),
            target_inputter=target_inputter,
            encoder=SequencedGraphEncoder(
                base_graph_encoder=GGNNEncoder(
                    num_timesteps=[args.ggnn_timesteps_per_layer for _ in range(args.ggnn_num_layers)],
                    node_feature_size=args.node_features_size,
                    gru_dropout_rate=args.node_features_dropout,
                ),
                gnn_input_size=args.node_features_size,
                encoder_type="bidirectional_rnn",
                num_units=args.rnn_hidden_size,
                num_layers=args.rnn_num_layers,
                dropout_rate=args.rnn_hidden_dropout,
                ignore_graph_encoder=args.ignore_graph_encoder,
            ),
            decoder=decoder,
            only_attend_primary=not args.attend_all_nodes,
            name=args.model_name,
        )
    return model


def infer(model, args):
    metadata = build_metadata(args)
    config = build_config(args)
    params = build_params(args)
    input_fn = model.input_fn(
        mode=tf.estimator.ModeKeys.PREDICT,
        batch_size=args.batch_size,
        metadata=metadata,
        features_file=args.infer_source_file,
    )
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=False)
    )

    _ = tf.Variable(initial_value="fake_variable")

    ckpt_name = args.infer_ckpt or "best.ckpt"
    iterator = get_iterator_from_input_fn(input_fn)
    with tf.Session(config=session_config) as session:
        saver = tf.train.Saver(max_to_keep=100)
        saver.restore(session, os.path.join(args.checkpoint_dir, ckpt_name))

        # build eval graph, loss and prediction ops
        features = iterator.get_next()
        with tf.variable_scope(args.model_name, reuse=True):
            _, predictions = model(features, None, tf.estimator.ModeKeys.PREDICT, params, config)

        session.run([iterator.initializer, tf.tables_initializer()])

        steps = 0
        infer_predictions = []
        while True:
            try:
                batch_predictions = session.run(predictions)
                batch_predictions = [
                    model.process_prediction({"tokens": prediction}) for prediction in batch_predictions["tokens"]
                ]

                infer_predictions = infer_predictions + batch_predictions
                steps += 1
            except tf.errors.OutOfRangeError:
                break

    with open(args.infer_predictions_file, "w") as out_file:
        for prediction in infer_predictions:
            out_file.write(json.dumps(prediction) + "\n")


def build_metadata(args):
    metadata = {
        "node_vocabulary": args.node_vocab_file,
        "edge_vocabulary": args.edge_vocab_file,
        "target_vocabulary": args.target_vocab_file,
    }
    return metadata


def build_config(args):
    config = {
        # TODO
    }
    return config


def build_params(args):
    params = {
        "maximum_iterations": args.max_iterations,
        "beam_width": args.beam_width,
        "length_penalty": args.length_penalty,
    }
    return params


def build_optimizer(args):
    global_step = tf.train.get_or_create_global_step()

    optimizer = args.optimizer
    if optimizer == "adam":
        optimizer_class = tf.train.AdamOptimizer
        kwargs = {}
    elif optimizer == "adagrad":
        optimizer_class = tf.train.AdagradOptimizer
        kwargs = {"initial_accumulator_value": args.adagrad_initial_accumulator}
    elif optimizer == "momentum":
        optimizer_class = tf.train.MomentumOptimizer
        kwargs = {"momentum": args.momentum_value, "use_nesterov": True}
    else:
        optimizer_class = getattr(tf.train, optimizer, None)
        if optimizer_class is None:
            raise ValueError("Unsupported optimizer %s" % optimizer)
        kwargs = {}
        # TODO: optimizer params
        # optimizer_params = params.get("optimizer_params", {})

    def optimizer(lr):
        return optimizer_class(lr, **kwargs)

    learning_rate = args.learning_rate
    if args.lr_decay_rate:
        learning_rate = tf.train.exponential_decay(
            learning_rate, global_step, decay_steps=args.lr_decay_steps, decay_rate=args.lr_decay_rate, staircase=True
        )

    return lambda loss: tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        clip_gradients=args.clip_gradients,
        summaries=[
            "learning_rate",
            "global_gradient_norm",
        ],
        optimizer=optimizer,
        name="optimizer",
    )


def compute_rouge(predictions, targets):
    predictions = [" ".join(prediction).lower() for prediction in predictions]
    predictions = [prediction if prediction else "EMPTY" for prediction in predictions]
    targets = [" ".join(target).lower() for target in targets]
    targets = [target if target else "EMPTY" for target in targets]
    rouge = Rouge()
    scores = rouge.get_scores(hyps=predictions, refs=targets, avg=True)
    return scores["rouge-2"]["f"]


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


if __name__ == "__main__":
    main()
