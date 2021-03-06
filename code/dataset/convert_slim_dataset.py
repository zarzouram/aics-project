"""
Modified from
https://github.com/rnagumo/gqnlib/blob/master/gqnlib/slim_dataset.py
"""
"""Convert slim tfrecords to pt.gz files.

This file converts tfrecords in DeepMind slim dataset to gzip files. Each
tfrecord will be converted to multiple gzip files which contain a list of
tuples.

For example, when converting `turk_data` dataset with batch size of
`100`, a single tfrecord file which contains `2000` sequences is converted
to `20` gzip files which contain a list of `100` tuples.

ex) train.tfrecord -> train-1.pt.gz, ..., train-20.pt.gz

Each tuple has elements as following.

basic: `(frames, cameras, top_down, captions, simple_captions)`.
metadata: `(meta_shape, meta_color, meta_size, meta_obj_positions,
meta_obj_rotations, meta_obj_colors)`.

ref)
https://github.com/deepmind/slim-dataset/blob/master/reader.py
"""
# from typing import List
import argparse
import logging
import time
import functools
import gzip
import multiprocessing as mp
import os
import pathlib
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# import stanza
import torch
# from torch import Tensor
# from torch.nn.utils.rnn import pad_sequence
import numpy as np

import flair
from flair.data import Sentence
flair.device = torch.device("cpu")

_NUM_VIEWS = 10
_NUM_RAW_CAMERA_PARAMS = 3
_IMAGE_SCALE = 0.5
_USE_SIMPLIFIED_CAPTIONS = False
_PARSE_METADATA = True
logging.getLogger('tensorflow').disabled = True
# nlp = stanza.Pipeline(lang='en', processors='tokenize', use_gpu=False)
# vocab = Vocabulary()


def calculate_time(start_time, end_time):
    time = end_time - start_time
    hours = int(time // 3600)
    time = time - 3600 * hours
    mins = int(time // 60)
    secs = int(time - (mins * 60))
    return hours, mins, secs


def np_to_list_str(str_byte_array):
    byte_decoder = np.vectorize(lambda x: x.decode())
    np2str = np.vectorize(lambda x: str(x))

    str_decoded = byte_decoder(str_byte_array)
    list_str = np2str(str_decoded.reshape(-1)).tolist()
    return list_str


def convert_record(path: pathlib.Path, save_dir: pathlib.Path, batch_size: int,
                   first_n: int, vocab_filepath: str) -> None:
    """Main process for one tfrecord file.

This method load one tfrecord file, and preprocess each (frames, cameras,
captions). The maximum number of processed (frames, cameras,
captions) are bounded by `batch_size`.

    Args:
        path:           pathlib.Path
                        Path to original data.
        save_dir:       pathlib.Path
                        Path to saved data.
        batch_size:     int
                        Batch size of dataset for each tfrecord.
        first_n:        int
                        Number of data to read (-1 means all).
        vocab_filepath: str
                        Path to vocab file.
    """
    # Load vocabulary file
    # if pathlib.Path(vocab_filepath).is_file():
    #     vocab.read_json(vocab_filepath)

    # Load tfrecord
    dataset = tf.data.TFRecordDataset(str(path))

    # Preprocess for each data
    images_list = []
    views_list = []
    captions_text = []
    batch = 1
    for i, raw_data in enumerate(dataset.take(first_n)):
        # image, views, texts, captions = preprocess_data(raw_data)
        image, views, texts = preprocess_data(raw_data)
        images_list.append(image)
        views_list.append(views)
        captions_text.append(texts)

        # Save batch to a gzip file
        if (i + 1) % batch_size == 0:
            images = torch.squeeze(torch.stack(images_list))
            views = torch.squeeze(torch.stack(views_list))

            scene_list = [images, views, captions_text]

            save_path = save_dir / f"{path.stem}-{batch}.pt.gz"
            with gzip.open(str(save_path), "wb") as f:
                torch.save(scene_list, f)

            images_list = []
            views_list = []
            captions_text = []
            scene_list = []
            batch += 1
    else:
        # Save rest
        if images_list:
            images = torch.squeeze(torch.stack(images_list))
            views = torch.squeeze(torch.stack(views_list))
            # captions_text = np.squeeze(captions_text)

            scene_list = [images, views, captions_text]

            save_path = save_dir / f"{path.stem}-{batch}.pt.gz"
            with gzip.open(str(save_path), "wb") as f:
                torch.save(scene_list, f)

    # vocab.to_json(vocab_filepath)


def preprocess_data(raw_data: tf.Tensor) -> tuple:
    """Converts raw data to tensor and saves into torch gziped file.

    Args:
        raw_data (tf.Tensor): Buffer.
    """

    tensor_dict = _parse_proto(raw_data)

    # Preprocess
    frames = _preprocess_images(tensor_dict["images"]).numpy()
    cameras = _preprocess_cameras(tensor_dict["cameras"]).numpy()
    captions = tensor_dict["captions"].numpy()

    all_caption = []
    for caption in captions:
        all_caption.append(Sentence(str(caption.decode()).lower()))

    # Frames size: (10, 64, 64, 3) ==change==> (10, 64, 64, 3)
    # cameras size: (10, 4).
    frames = frames.transpose(0, 3, 1, 2)

    # returned_values = [frames, cameras, top_down, captions, simple_captions]
    frames = torch.from_numpy(frames)
    cameras = torch.from_numpy(cameras)
    # returned_values = (frames, cameras, captions, captions_)
    returned_values = (frames, cameras, all_caption)
    return returned_values


def _parse_proto(buf: tf.Tensor) -> dict:
    """Parse binary protocol buffer into tensors.

    The protocol buffer is expected to contain the following fields:

        * frames: 10 views of the scene rendered as images.
        * top_down_frame: single view of the scene from above rendered as an
            image.
        * cameras: 10 vectors describing the camera position from which the
            frames have been rendered
        * captions: A string description of the scene. For the natural language
            dataset, contains descriptions written by human annotators. For
            synthetic data contains a string describing each relation between
            objects in the scene exactly once.
        * simplified_captions: A string description of the scene. For the
            natural language dataset contains a string describing each relation
            between objects in the scene exactly once. For synthetic datasets
            contains a string describing every possible pairwise relation
            between objects in the scene.
        * meta_shape: A vector of strings describing the object shapes.
        * meta_color: A vector of strings describing the object colors.
        * meta_size: A vector of strings describing the object sizes.
        * meta_obj_positions: A matrix of floats describing the position of
            each object in the scene.
        * meta_obj_rotations: A matrix of floats describing the rotation of
            each object in the scene.
        * meta_obj_rotations: A matrix of floats describing the color of each
            object in the scene as RGBA in the range [0, 1].

    Args:
        buf: A string containing the serialized protocol buffer.

    Returns:
        A dictionary containing tensors for each of the fields in the protocol
        buffer. If _PARSE_METADATA is False, will omit fields starting with
        'meta_'.
    """

    feature_map = {
        "frames":
        tf.io.FixedLenFeature(shape=[_NUM_VIEWS], dtype=tf.string),
        # "top_down_frame":
        # tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
        "cameras":
        tf.io.FixedLenFeature(shape=[_NUM_VIEWS * _NUM_RAW_CAMERA_PARAMS],
                              dtype=tf.float32),
        "captions":
        tf.io.VarLenFeature(dtype=tf.string),
        # "simplified_captions":
        # tf.io.VarLenFeature(dtype=tf.string),
        # "meta_shape":
        # tf.io.VarLenFeature(dtype=tf.string),
        # "meta_color":
        # tf.io.VarLenFeature(dtype=tf.string),
        # "meta_size":
        # tf.io.VarLenFeature(dtype=tf.string),
        # "meta_obj_positions":
        # tf.io.VarLenFeature(dtype=tf.float32),
        # "meta_obj_rotations":
        # tf.io.VarLenFeature(dtype=tf.float32),
        # "meta_obj_colors":
        # tf.io.VarLenFeature(dtype=tf.float32),
    }

    example = tf.io.parse_single_example(buf, feature_map)
    images = tf.concat(example["frames"], axis=0)
    images = tf.nest.map_structure(
        tf.stop_gradient,
        tf.map_fn(tf.image.decode_jpeg,
                  tf.reshape(images, [-1]),
                  dtype=tf.uint8))
    # top_down = tf.image.decode_jpeg(tf.squeeze(example["top_down_frame"]))
    cameras = tf.reshape(example["cameras"],
                         shape=[-1, _NUM_RAW_CAMERA_PARAMS])
    captions = tf.sparse.to_dense(example["captions"], default_value="")
    # simplified_captions = tf.sparse.to_dense(example["simplified_captions"],
    #                                          default_value="")
    # meta_shape = tf.sparse.to_dense(example["meta_shape"], default_value="")
    # meta_color = tf.sparse.to_dense(example["meta_color"], default_value="")
    # meta_size = tf.sparse.to_dense(example["meta_size"], default_value="")
    # meta_obj_positions = tf.sparse.to_dense(example["meta_obj_positions"],
    #                                         default_value=0)
    # meta_obj_positions = tf.reshape(meta_obj_positions, shape=[-1, 3])
    # meta_obj_rotations = tf.sparse.to_dense(example["meta_obj_rotations"],
    #                                         default_value=0)
    # meta_obj_rotations = tf.reshape(meta_obj_rotations, shape=[-1, 4])
    # meta_obj_colors = tf.sparse.to_dense(example["meta_obj_colors"],
    #                                      default_value=0)
    # meta_obj_colors = tf.reshape(meta_obj_colors, shape=[-1, 4])

    data_tensors = {
        "images": images,
        "cameras": cameras,
        "captions": captions,
        # "simplified_captions": simplified_captions,
        # "top_down": top_down
    }
    # if _PARSE_METADATA:
    #     data_tensors.update({
    #         "meta_shape": meta_shape,
    #         "meta_color": meta_color,
    #         "meta_size": meta_size,
    #         "meta_obj_positions": meta_obj_positions,
    #         "meta_obj_rotations": meta_obj_rotations,
    #         "meta_obj_colors": meta_obj_colors
    #     })

    return data_tensors


def _convert_and_resize_images(images: tf.Tensor,
                               old_size: tf.Tensor) -> tf.Tensor:
    """Resizes images with `_IMAGE_SCALE` ratio."""

    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    new_size = tf.cast(old_size, tf.float32) * _IMAGE_SCALE
    new_size = tf.cast(new_size, tf.int32)
    images = tf.image.resize(images, new_size)
    return images


def _preprocess_images(images: tf.Tensor) -> tf.Tensor:
    old_size = tf.shape(images)[1:3]
    images = _convert_and_resize_images(images, old_size)
    return images


def _preprocess_topdown(td_image: tf.Tensor) -> tf.Tensor:
    old_size = tf.shape(td_image)[0:2]
    td_image = _convert_and_resize_images(td_image, old_size)
    return td_image


def _preprocess_cameras(raw_cameras: tf.Tensor) -> tf.Tensor:
    azimuth = raw_cameras[:, 0]
    pos = raw_cameras[:, 1:]
    cameras = tf.concat([
        pos,
        tf.expand_dims(tf.sin(azimuth), -1),
        tf.expand_dims(tf.cos(azimuth), -1),
    ],
                        axis=1)
    return cameras


# def build_vocab(captions: np.ndarray) -> List[Tensor]:
#     """tokinize captions text then create vocabulary dict"""
#     sents = []
#     for caption in captions:
#         doc = nlp(caption.decode())  # caption tokinization
#         word_ids = []
#         for sentence in doc.sentences:
#             for word in sentence.words:
#                 word_ids.append(vocab.token2index(word.text))
#         sents.append(torch.tensor(word_ids, device=torch.device("cpu")))

#     return sents


def main():
    # Specify dataset name
    parser = argparse.ArgumentParser(description="Convert tfrecord to torch")
    parser.add_argument("--dataset_path",
                        type=str,
                        default="~/resources/slim",
                        help="Dataset path.")

    parser.add_argument("--dataset",
                        type=str,
                        default="turk_data",
                        help="Dataset name.")

    parser.add_argument("--bert_dir",
                        type=str,
                        help="Path to Bert Model to tokenize Texts")

    parser.add_argument("--mode",
                        type=str,
                        default="train",
                        help="Mode {train, test, valid}")

    parser.add_argument("--vocab_path",
                        type=str,
                        help="Path to the vocabulary json file")

    parser.add_argument("--batch-size",
                        type=int,
                        default=10,
                        help="Batch size of tfrecords.")

    parser.add_argument("--first-n",
                        type=int,
                        default=10,
                        help="Read only first n data in single a record "
                        "(-1 means all data).")
    args = parser.parse_args()

    # Path
    # root = pathlib.Path().absolute()
    # root = pathlib.Path(os.getenv("DATA_DIR", "./data/"))
    dataset_path = os.path.expanduser(args.dataset_path)
    tf_dir = pathlib.Path(f"{dataset_path}/{args.dataset}/{args.mode}/")
    dataset_parent_path = pathlib.Path(f"{dataset_path}").parent
    torch_dir = pathlib.Path(
        f"{dataset_parent_path}/{args.dataset}_torch/{args.mode}/")

    torch_dir.mkdir(parents=True, exist_ok=True)

    if not tf_dir.exists():
        raise FileNotFoundError(f"TFRecord path `{tf_dir}` does not exists.")

    # File list of original dataset
    record_list = sorted(tf_dir.glob("*.tfrecord"))

    # Multi process
    vocab_filepath = os.path.expanduser(args.vocab_path)
    num_proc = mp.cpu_count()

    start_time = time.time()
    with mp.Pool(processes=num_proc) as pool:
        f = functools.partial(convert_record,
                              save_dir=torch_dir,
                              batch_size=args.batch_size,
                              first_n=args.first_n,
                              vocab_filepath=vocab_filepath)
        pool.map(f, record_list)

    end_time = time.time()
    hr, mins, secs = calculate_time(start_time, end_time)
    print("\n\n")
    print("=" * 100)
    print(f"\n\nProcess data finished. ET:{hr:2d}:{mins:2d}:{secs:2d}\n\n")
    print("----" * 20)


if __name__ == "__main__":
    # tf.enable_eager_execution()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    main()
