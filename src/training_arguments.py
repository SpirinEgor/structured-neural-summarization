from argparse import ArgumentParser
from os.path import join

from data_preprocessing import GRAPH_FILENAME, TARGET_FILENAME, NODE_VOCAB, EDGE_VOCAB, TARGET_VOCAB

DEFAULT_DATA_DIR = "../../data/java2graph/small-processed"

DEFAULT_TRAIN_SOURCE_FILE = join(DEFAULT_DATA_DIR, "train", GRAPH_FILENAME)
DEFAULT_TRAIN_TARGET_FILE = join(DEFAULT_DATA_DIR, "train", TARGET_FILENAME)

DEFAULT_VALID_SOURCE_FILE = join(DEFAULT_DATA_DIR, "val", GRAPH_FILENAME)
DEFAULT_VALID_TARGET_FILE = join(DEFAULT_DATA_DIR, "val", TARGET_FILENAME)

DEFAULT_NODE_VOCAB_FILE = join(DEFAULT_DATA_DIR, NODE_VOCAB)
DEFAULT_EDGE_VOCAB_FILE = join(DEFAULT_DATA_DIR, EDGE_VOCAB)
DEFAULT_TARGET_VOCAB_FILE = join(DEFAULT_DATA_DIR, TARGET_VOCAB)

DEFAULT_MODEL_NAME = "java_small_mnp"


def configure_arg_parser() -> ArgumentParser:
    # argument parsing
    parser = ArgumentParser()

    # optimization arguments
    parser.add_argument("--optimizer", default="adam", type=str, help="Number of epochs to train the model")
    parser.add_argument("--train_steps", default=300000, type=int, help="Number of steps to optimize")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The learning rate for the optimizer")
    parser.add_argument("--lr_decay_rate", default=0.0, type=float, help="Learning rate decay rate")
    parser.add_argument(
        "--lr_decay_steps", default=10000, type=float, help="Number of steps between learning rate decay application"
    )
    parser.add_argument(
        "--adagrad_initial_accumulator", default=0.1, type=float, help="Number of epochs to train the model"
    )
    parser.add_argument("--momentum_value", default=0.95, type=float, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", default=16, type=int, help="Number of epochs to train the model")
    parser.add_argument(
        "--sample_buffer_size",
        default=10000,
        type=int,
        help="The number of samples in the buffer shuffled before training",
    )
    parser.add_argument(
        "--bucket_width", default=5, type=int, help="Range of allowed lengths in a batch. Optimizes RNN loops"
    )
    parser.add_argument("--clip_gradients", default=5.0, type=float, help="Maximum norm of the gradients")
    parser.add_argument(
        "--validation_interval",
        default=20000,
        type=int,
        help="The number of training steps between each validation run",
    )

    parser.add_argument(
        "--patience", default=5, type=int, help="Number of worse validations needed to trigger early stop"
    )
    parser.add_argument("--logging_window", default=200, type=int, help="Number of steps taken when logging")

    # model options arguments
    parser.add_argument("--source_embeddings_size", default=128, type=int, help="Size of the input tokens embeddings")
    parser.add_argument("--target_embeddings_size", default=128, type=int, help="Size of the target token embeddings")
    parser.add_argument(
        "--embeddings_dropout", default=0.2, type=float, help="Dropout applied to the node embeddings during training"
    )
    parser.add_argument("--node_features_size", default=256, type=int, help="Size of the node features hidden state")
    parser.add_argument(
        "--node_features_dropout", default=0.2, type=float, help="Dropout applied to the node features during training"
    )
    parser.add_argument("--ggnn_num_layers", default=4, type=int, help="Number of GGNN layers with distinct weights")
    parser.add_argument("--ggnn_timesteps_per_layer", default=1, type=int, help="Number of GGNN propagations per layer")
    parser.add_argument("--rnn_num_layers", default=1, type=int, help="Number of layers in the input and output rnns")
    parser.add_argument(
        "--rnn_hidden_size", default=256, type=int, help="Size of the input and output rnns hidden state"
    )
    parser.add_argument(
        "--rnn_hidden_dropout", default=0.3, type=float, help="Dropout applied to the rnn hidden state during training"
    )
    parser.add_argument(
        "--attend_all_nodes",
        default=False,
        action="store_true",
        help="If enabled, attention and copying will consider all nodes "
        "rather than only the ones in the primary sequence",
    )
    parser.add_argument(
        "--only_graph_encoder",
        default=False,
        action="store_true",
        help="If enabled, the model will ignore the sequence encoder, " "using only the graph structure",
    )
    parser.add_argument(
        "--ignore_graph_encoder",
        default=False,
        action="store_true",
        help="If enabled, the model ignore the graph encoder, using only " "the primary sequence encoder",
    )
    parser.add_argument(
        "--copy_attention", default=False, action="store_true", help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--coverage_layer", default=False, action="store_true", help="Number of epochs to train the model"
    )
    parser.add_argument("--coverage_loss", default=0.0, type=float, help="Number of epochs to train the model")
    parser.add_argument(
        "--max_iterations", default=120, type=int, help="The maximum number of decoding iterations at inference time"
    )
    parser.add_argument("--beam_width", default=10, type=int, help="The number of beam to search while decoding")
    parser.add_argument("--length_penalty", default=1.0, type=float, help="The length ")
    parser.add_argument(
        "--case_sensitive", default=False, action="store_true", help="If enabled, node labels are case sentitive"
    )

    # arguments for loading data
    parser.add_argument(
        "--train_source_file",
        default=DEFAULT_TRAIN_SOURCE_FILE,
        type=str,
        help="Path to the jsonl.gz file containing the train input graphs",
    )
    parser.add_argument(
        "--train_target_file",
        default=DEFAULT_TRAIN_TARGET_FILE,
        type=str,
        help="Path to the jsonl.gz file containing the train input graphs",
    )
    parser.add_argument(
        "--valid_source_file",
        default=DEFAULT_VALID_SOURCE_FILE,
        type=str,
        help="Path to the jsonl.gz file containing the valid input graphs",
    )
    parser.add_argument(
        "--valid_target_file",
        default=DEFAULT_VALID_TARGET_FILE,
        type=str,
        help="Path to the jsonl.gz file containing the valid input graphs",
    )
    parser.add_argument(
        "--infer_source_file",
        default=None,
        help="Path to the jsonl.gz file in which we wish to do inference " "after training is complete",
    )
    parser.add_argument(
        "--infer_predictions_file", default=None, help="Path to the file to save the results from inference"
    )
    parser.add_argument(
        "--node_vocab_file", default=DEFAULT_NODE_VOCAB_FILE, type=str, help="Path to the json containing the dataset"
    )
    parser.add_argument(
        "--edge_vocab_file", default=DEFAULT_EDGE_VOCAB_FILE, type=str, help="Path to the json containing the dataset"
    )
    parser.add_argument(
        "--target_vocab_file",
        default=DEFAULT_TARGET_VOCAB_FILE,
        type=str,
        help="Path to the json containing the dataset",
    )
    parser.add_argument(
        "--truncated_source_size",
        default=500,
        type=int,
        help="Max size for source sequences in the input graphs after truncation",
    )
    parser.add_argument(
        "--truncated_target_size", default=100, type=int, help="Max size for target sequences after truncation"
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, type=str, help="Model name")

    # arguments for persistence
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Directory to where to save the checkpoints")
    parser.add_argument("--seed", default=7, type=int, help="Random seed")

    # arguments for debugging
    parser.add_argument(
        "--debug_mode", default=False, action="store_true", help="If true, it will enable the tensorflow debugger"
    )

    parser.add_argument("--infer_ckpt", default=None, type=str, help="Name of checkpoint to infer")
    return parser
