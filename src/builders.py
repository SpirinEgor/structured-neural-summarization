import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention

from opengnn.decoders.sequence import HybridPointerDecoder, RNNDecoder
from opengnn.encoders import GGNNEncoder, SequencedGraphEncoder
from opengnn.inputters import CopyingTokenEmbedder, TokenEmbedder, GraphEmbedder, SequencedGraphInputter
from opengnn.models import GraphToSequence, SequencedGraphToSequence
from opengnn.utils import CoverageBahdanauAttention


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
