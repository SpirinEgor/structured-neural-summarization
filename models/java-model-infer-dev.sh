DATA_FOLDER="../data/java2graph/test-processed"

TRAIN="train"
VAL="val"
TEST="test"

GRAPHS="inputs.jsonl.gz"
TARGETS="targets.jsonl.gz"
NODE_VOCAB="node.vocab"
EDGE_VOCAB="edge.vocab"
TARGET_VOCAB="output.vocab"

CHECKPOINT_DIR="checkpoints"
MODEL_NAME="java_dev_1633356670"
CHECKPOINT_NAME="java_dev_1633356670.ckpt-5000"

PREDICTIONS="predictions.jsonl"

python infer.py --node_vocab_file "$DATA_FOLDER/$NODE_VOCAB" \
                --edge_vocab_file "$DATA_FOLDER/$EDGE_VOCAB" \
                --target_vocab_file "$DATA_FOLDER/$TARGET_VOCAB" \
                --model_name "$MODEL_NAME"  \
                --checkpoint_dir "$CHECKPOINT_DIR/$MODEL_NAME" \
                --infer_source_file "$DATA_FOLDER/$TEST/$GRAPHS" \
                --infer_predictions_file "$DATA_FOLDER/$TEST/$PREDICTIONS" \
                --infer_ckpt "$CHECKPOINT_NAME" \
                --copy_attention  \
                --rnn_hidden_dropout 0.0 \
                --node_features_dropout 0.0 \
                --embeddings_dropout 0.0 \
                --case_sensitive  \
                --attend_all_nodes \
                --source_embeddings_size 10 \
                --target_embeddings_size 10 \
                --node_features_size 10 \
                --ggnn_num_layers 1 \
                --rnn_hidden_size 10 \
