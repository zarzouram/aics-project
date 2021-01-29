# Data

- [1. General Description](#1-general-description)
- [2. Download](#2-download)
- [3. Dataset Conversion](#3-dataset-conversion)
- [References](#references)

## 1. General Description

This dataset is used to train Spatial Language Integrating Model (SLIM) in [[1]](#1).
It consists of virtual scenes with ten views for each scene. Each scene
consists of two or three objects placed on a square walled room. Each view is
represented by an image, and synthetic or natural language descriptions. View
images are 3D pictures rendered from a particular scene from ten different
camera viewpoints.

## 2. Download

`gustil` need to be installed to download the dataset. The dataset is available
from [here](https://console.cloud.google.com/storage/slim-dataset). Because the
dataset is huge, about 600 GB, I downloaded the whole dataset except the
`synthetic_data/train` data.

Use the command below to download the dataset. After that, make sure to manually
create `test`, `valid`, and `train` directories under the `turk_data` and move
the respective files under them. The dataset directory should be as shown below.

```bash
gsutil -m cp -c -L manifest.log -r \
  "gs://slim-dataset/synthetic_data/test" \
  "gs://slim-dataset/synthetic_data/valid" \
  "gs://slim-dataset/turk_data/" \
  ./<DATASET_FOLDER>/
```

```none
<DATASET_FOLDER>
└── synthetic_data
    │   ├── test
    │   └── valid
    └── turk_data
        ├── test
        ├── train
        └── valid
```

## 3. Dataset Conversion

Dataset files are in `tfrecord` fromat. TFRecord file format is binary storage
format optimized use with Tensorflow. As I will use pyTorch, the dataset files
are converted to `pt.gz` format. To convert the dataset, when you are under
`data` directory, use the following command:

```bash
./convert_slim_dataset.sh <absolute_path/to/dataset_folder>
```

Note that the conversion process takes a lot of time. After conversion, dataset
directory will be as shown below. Code is based on [this](https://github.com/rnagumo/gqnlib/blob/master/examples/convert_slim_dataset.py) and [this](https://github.com/rnagumo/gqnlib/blob/master/bin/download_slim.sh),

```none
<DATASET_FOLDER>
├── synthetic_data
│   │   ├── test
│   │   └── valid
│   └── turk_data
│       ├── test
│       ├── train
│       └── valid
├── synthetic_data_torch
│   ├── test
│   └── valid
└── turk_data_torch
    ├── test
    ├── train
    └── valid
```

## References

<a id="1">[1]</a>
Ramalho, T., Kočiský, T., Besse, F., Eslami, S. M., Melis, G., Viola, F., ... &
Hermann, K. M. (2018). Encoding spatial relations from natural language. arXiv
preprint arXiv:1807.01670.
