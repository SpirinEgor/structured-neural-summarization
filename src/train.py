import os

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tqdm.auto import tqdm

from opengnn.utils import read_jsonl_gz_file
from src.builders import build_optimizer, build_metadata, build_config, build_params
from src.utils import Summary, get_iterator_from_input_fn
from src.evaluate import evaluate


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
                valid_loss, valid_rouge, valid_prec, valid_rec, valid_f1 = evaluate(
                    session, model, valid_iterator, valid_tb_loss, predictions, valid_targets, args.batch_size
                )
                print_str = ""
                for value, name in [
                    (valid_loss, "loss"),
                    (valid_rouge, "rouge"),
                    (valid_prec, "precision"),
                    (valid_rec, "recall"),
                    (valid_f1, "f1"),
                ]:
                    print_str += f"eval {name}: {round(value, 2)}; "
                    valid_summary.scalar(name, value, train_step)
                tqdm.write(print_str)

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
