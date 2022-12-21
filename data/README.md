# Data

- [1. General Description](#1-general-description)
- [2. Download](#2-download)
- [3. Dataset Conversion](#3-dataset-conversion)
- [4. References](#4-references)

## 1. General Description

This dataset is used to train Spatial Language Integrating Model (SLIM) in
[[1]](#1).  It consists of virtual scenes with ten views for each scene. Each
scene consists of two or three objects placed on a square walled room. Each
view is represented by an image, and synthetic or natural language
descriptions. View images are 3D pictures rendered from a particular scene from
ten different camera viewpoints.

## 2. Download

`gustil` need to be installed to download the dataset. The dataset is available
from [here](https://console.cloud.google.com/storage/slim-dataset). Because the
dataset is huge, about 600 GB, I downloaded the whole dataset except the
`synthetic_data/train` data.

Use the command below to download the dataset.

```bash
gsutil -m cp -c -L manifest.log -r \
  "gs://slim-dataset/turk_data/" \
  ./<DATASET_FOLDER>/
```

After downloading is finished, make sure to manually create `test`, `valid`,
and `train` directories under the `turk_data` and move the respective files
under them. The dataset directory should be as shown below.


```none
<DATASET_FOLDER>
└── synthetic_data
    └── turk_data
        ├── test
        ├── train
        └── valid
```

## 3. Dataset Conversion

Dataset files are in `tfrecord` fromat. TFRecord file format is a binary
storage format which are optimized to be used with Tensorflow. As I will use
pyTorch, the dataset files are converted to `pt.gz` format. To convert the
dataset use the following command:

```bash
./convert_slim_dataset.sh <absolute_path/to/dataset_folder>
```

If you want to use the default value, just use: `./convert_slim_dataset.sh `

After conversion, dataset directory will be as follows:

```none
<DATASET_FOLDER>
├── synthetic_data
│   └── turk_data
│       ├── test
│       ├── train
│       └── valid
└── turk_data_torch
    ├── test
    ├── train
    └── valid
```


## 4. References

<a id="1">[1]</a>
Ramalho, T., Kočiský, T., Besse, F., Eslami, S. M., Melis, G., Viola, F., ... &
Hermann, K. M. (2018). Encoding spatial relations from natural language. arXiv
preprint arXiv:1807.01670.
