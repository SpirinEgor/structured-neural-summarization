# Structured Neural Summarization

For the original README refer to [original repo](https://github.com/CoderPat/structured-neural-summarization).

## Requirements

Use `requirements.txt` to install all necessary dependencies:
```shell
pip install -r requirements.txt
```

Installing OpenGNN may require manual installation from the repo:
1. Clone GitHub repo with sources:
```shell
git clone https://github.com/CoderPat/OpenGNN/tree/3c1229ef58c0d95fcbe58082e89eb9a2a2694011
```
2. Move to the folder with it and install OpenGNN as python package:
```shell
pip install -e .
```

## Data preprocessing

Assume that you have a Java dataset with source code already split into holdouts:
```
dataset-name/
    train/
    val/
    test/
```
**First**, we need to extract graphs from the data.
Navigate to the folder with parser: [`parsers/sourcecode/java`](parsers/sourcecode/java).
There are already prepared configuration files for the IntelliJ IDEA,
so you can open parser as project in it and use run button to process files.

Of course, you can do it manually. You need the Maven and Java at least 8th version.
1. Build jar executable file:
```shell
mvn clean compile assembly:single
```
2. Run jar with paths to original dataset and output folder:
```shell
java -jar target/java2graph-1.0-jar-with-dependencies.jar <path>/dataset-name <output-path>
```

During graph extraction all logs are stored in `log.txt` file.

**Second**, we need to process graphs to model format. Use `data_preprocessing` script for it:
```shell
python data_preprocessing.py -i <input-path> -o <output-path>
```
`<input-path>` — path to graphs extracted from the first step.

`<output-path>` — path to output folder.

## Model training

[`Models`](models) folder contains multiple example of scripts to train and evaluate the model.
Use them and modify to suit your needs.
Quick overview:
- [`train_and_eval.py`](train_and_eval.py) script is used to train GNN with evaluation every N steps.
- [`infer.py`](infer.py) script is used to run trained model on test data.

## Known issues

This repo doesn't change or modify any behaviour of the original model.
This repo makes step forward for easy usage and running it.
Unfortunately, some errors and bugs appear, all known issues collected in this section.

### `'MapDataset' object has no attribute 'output_shapes'`

In OpenGNN sources navigate to [`opengnn/utils/data.py`](OpenGNN/opengnn/utils/data.py).
Find `get_padded_shapes` function and change `dataset.output_shapes` to `tf.compat.v1.data.get_output_shapes(dataset)`.

Corresponding [GutHub issue](https://github.com/tensorflow/tensorflow/issues/28148).