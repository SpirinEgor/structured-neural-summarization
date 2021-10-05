import codecs
import gzip
import json
from argparse import ArgumentParser
from os import listdir, makedirs
from os.path import join
from typing import Dict, List, Tuple

from tqdm import tqdm

from data.utils import save_jsonl_gz
from opengnn import constants
from opengnn.utils.vocab import Vocab
from parsers.sourcecode.csharp.programgraphs2opengnn import process_sample
from parsers.sourcecode.graph2subtokengraph import graph_transformer

HOLDOUT_NAMES = ["train", "val", "test"]
DATA_NAME = "data"
EXTENSION = ".json.gz"

GRAPH_FILENAME = "inputs.jsonl.gz"
TARGET_FILENAME = "targets.jsonl.gz"
NODE_VOCAB = "node.vocab"
EDGE_VOCAB = "edge.vocab"
TARGET_VOCAB = "output.vocab"


Graph = Dict[str, List]


def process_graph_file(input_graphs_path: str) -> Tuple[List[Graph], List[str]]:
    reader = codecs.getreader("utf-8")
    with gzip.open(input_graphs_path, "r") as f:
        graphs = json.loads(reader(f).read())
    processed_graphs = []
    labels = []
    for graph in tqdm(graphs, desc="Graphs", leave=False):
        sample = process_sample(graph, names=True)
        if sample is None:
            continue
        sub_tokenized_graph = graph_transformer(sample[0])
        processed_graphs.append(sub_tokenized_graph)
        labels.append(sample[1])
    return processed_graphs, labels


def process_holdout(input_folder: str, holdout_name: str, output_folder: str):
    holdout_folder = join(input_folder, holdout_name)
    input_graph_files = [it for it in listdir(holdout_folder) if it.endswith(EXTENSION)]
    transformed_graphs = []
    targets = []
    for input_graph_file in tqdm(input_graph_files, desc=f"{holdout_name} files"):
        cur_graphs, cur_targets = process_graph_file(join(holdout_folder, input_graph_file))
        transformed_graphs += cur_graphs
        targets += cur_targets

    output_folder = join(output_folder, holdout_name)
    makedirs(output_folder, exist_ok=True)
    save_jsonl_gz(join(output_folder, GRAPH_FILENAME), transformed_graphs)
    save_jsonl_gz(join(output_folder, TARGET_FILENAME), targets)


def main(input_folder: str, output_folder: str):
    # convert java graphs into model format
    print("Start processing extracted java graphs")
    for holdout in HOLDOUT_NAMES:
        process_holdout(input_folder, holdout, output_folder)

    train_graphs = join(output_folder, "train", GRAPH_FILENAME)
    train_targets = join(output_folder, "train", TARGET_FILENAME)

    # collect vocabulary
    print("Start collecting vocabulary for node labels")
    node_vocabulary = Vocab(special_tokens=[constants.PADDING_TOKEN])
    # models/java-model.sh contain --case-sensitive key, thus collect vocabulary with enabled case sensitive option too.
    node_vocabulary.add_from_file(train_graphs, "node_labels", case_sentitive=True)
    node_vocabulary.serialize(join(output_folder, NODE_VOCAB))

    print("Start collecting vocabulary for edges")
    edge_vocabulary = Vocab(special_tokens=[])
    edge_vocabulary.add_from_file(train_graphs, "edges", index=0)
    edge_vocabulary.serialize(join(output_folder, EDGE_VOCAB))

    print("Start collecting vocabulary for targets")
    target_vocabulary = Vocab(
        special_tokens=[constants.PADDING_TOKEN, constants.START_OF_SENTENCE_TOKEN, constants.END_OF_SENTENCE_TOKEN]
    )
    target_vocabulary.add_from_file(train_targets)
    target_vocabulary.serialize(join(output_folder, TARGET_VOCAB))


if __name__ == "__main__":
    __args_parser = ArgumentParser()
    __args_parser.add_argument(
        "-i", "--input", required=True, help="Path to folder with extract graphs and dataset structure"
    )
    __args_parser.add_argument("-o", "--output", required=True, help="Path to output folder")

    __args = __args_parser.parse_args()
    main(__args.input, __args.output)
