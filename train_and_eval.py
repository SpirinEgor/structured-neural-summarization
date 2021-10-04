import os

import tensorflow as tf

from src.builders import build_model
from src.train import train_and_eval
from src.training_arguments import configure_arg_parser

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

    train_and_eval(model, args)


if __name__ == "__main__":
    main()
