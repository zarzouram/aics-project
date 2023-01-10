# Project for the AICS course (MLT LT2318)

This repository hosts the course project for the "LT2318: Artificiell
intelligens. This project is a reimpelemntation of a paper titled "Encoding
Spatial Relations from Natural Language" [[1]](#1).

## 1. Introduction

Spatial language involves words and phrases describing objects' position,
orientation, movement, and relationships in space. Examples of spatial language
include terms such as "above," "below," "near," "far," etc. Spatial language is
an important aspect of human communication, as it allows us to describe and
understand the world around us. For example, when we see a car moving, we might
describe its movement using spatial language, saying that the car is moving
"backward" or "forward." This allows us to communicate our observations clearly
and helps us understand the spatial relationships between objects in the world
around us.

The paper "Encoding Spatial Relations from Natural Language" presents a system
capable of capturing the semantics of spatial relations from natural language
descriptions. The system uses a multi-modal objective to generate images of
scenes from their textual descriptions. The SLIM dataset was proposed for this
reason.

## 2. Dataset

Dataset information can be found
[here](./data/README.md#1-general-description)

## 3. Code

### 3.1. Requirements

Code is tested using python 3.9. Use `conda create --name <env> --file
requirements.txt` to create virual env and to install the required libraries.

### 3.2. Create Dataset

Dataset download and conversion can be found
[here](./data/README.md#2-download)

Processed dataset files are under `/srv/data/zarzouram/lt2318/slim/turk_torch/`

### 3.3. training the model

`run_train.py` expects the following arguments:

1. dataset_dir: The parent directory contains the processed dataset files. It is the sas the output_dir in Section 2.3 Create Dataset
2. config_path: Path for the configuration json file default is `./codes/config.json`
3. checkpoints_dir: directory where checkpoints are saved
4. checkpoint_model: if train resuming is needed pass the checkpoint filename
5. device: either gpu or cpu

Loss are tracked using Tensorboard. The path to
tensoboard files is `./logs`.

```bash
python code/run_train.py [ARGUMENT]
```

You do not need to create new datasets, use `python code/run_train.py` to use
the default arguments.

### 3.4. Testing

Model testing are done in the `experiments.ipynb` notebook. The notebook is
configured to load the test results done by me from
`/srv/data/zarzouram/lt2318/test_outputs`.

If you want to re-test the model, under the `Testing Model` section, please
change the `retest` value to `True`.  Please do not forget to change the path
in the `save_data` otherwise you will override the saved test results.

## 4. Results

Please see the attached report, under `paper`


## 5. Reference

<a id="1">[1]</a>  Ramalho, T., Kočiský, T., Besse, F., Eslami, S. M., Melis,
G., Viola, F., ... & Hermann, K. M. (2018). Encoding spatial relations from
natural language. arXiv preprint arXiv:1807.01670.
