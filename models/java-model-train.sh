DATA_FOLDER="../data/java2graph/small-processed"

TRAIN="train"
VAL="val"

GRAPHS="inputs.jsonl.gz"
TARGETS="targets.jsonl.gz"
NODE_VOCAB="node.vocab"
EDGE_VOCAB="edge.vocab"
TARGET_VOCAB="output.vocab"

CHECKPOINT_DIR="checkpoints"
MODEL_NAME="java_small_$(date +%s)"

python train_and_eval.py --train_source_file "$DATA_FOLDER/$TRAIN/$GRAPHS" \
                         --train_target_file "$DATA_FOLDER/$TRAIN/$TARGETS" \
                         --valid_source_file "$DATA_FOLDER/$VAL/$GRAPHS" \
                         --valid_target_file "$DATA_FOLDER/$VAL/$TARGETS" \
                         --node_vocab_file "$DATA_FOLDER/$NODE_VOCAB" \
                         --edge_vocab_file "$DATA_FOLDER/$EDGE_VOCAB" \
                         --target_vocab_file "$DATA_FOLDER/$TARGET_VOCAB" \
                         --train_steps 50000 \
                         --optimizer momentum \
                         --learning_rate 0.1 \
                         --lr_decay_rate 0.95 \
                         --lr_decay_steps 5000 \
                         --copy_attention  \
                         --model_name "$MODEL_NAME"  \
                         --checkpoint_dir "$CHECKPOINT_DIR/$MODEL_NAME" \
                         --rnn_hidden_size 256 \
                         --rnn_hidden_dropout 0.0 \
                         --node_features_dropout 0.0 \
                         --validation_interval 5000 \
                         --embeddings_dropout 0.0 \
                         --case_sensitive  \
                         --attend_all_nodes
