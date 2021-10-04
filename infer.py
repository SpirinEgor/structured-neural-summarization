import json
import os

import tensorflow as tf

from src.builders import build_model, build_metadata, build_config, build_params
from src.evaluate import get_eval_predictions
from src.training_arguments import configure_arg_parser
from src.utils import get_iterator_from_input_fn

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    parser = configure_arg_parser()
    args = parser.parse_args()

    model = build_model(args)
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
    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
    iterator = get_iterator_from_input_fn(input_fn)
    with tf.Session(config=session_config) as session:
        saver = tf.train.import_meta_graph(f"{ckpt_path}.meta")
        saver.restore(session, ckpt_path)

        # build eval graph, loss and prediction ops
        features = iterator.get_next()
        with tf.variable_scope(args.model_name, reuse=tf.AUTO_REUSE):
            _, predictions = model(features, None, tf.estimator.ModeKeys.PREDICT, params, config)

        session.run([tf.global_variables_initializer(), iterator.initializer, tf.tables_initializer()])

        infer_predictions = get_eval_predictions(session, model, predictions)

    with open(args.infer_predictions_file, "w") as out_file:
        for prediction in infer_predictions:
            out_file.write(json.dumps(prediction) + "\n")


if __name__ == "__main__":
    main()
