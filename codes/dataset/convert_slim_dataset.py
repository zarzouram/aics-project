"""
Modified from
https://github.com/deepmind/slim-dataset/blob/master/reader.py#L60-L84
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
from typing import Dict

import logging
import argparse
import os
import pathlib
from tqdm import tqdm

import re
from collections import Counter, OrderedDict
from itertools import chain
import h5py

from concurrent.futures import ThreadPoolExecutor
from threading import current_thread, Lock

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

_NUM_VIEWS = 10
_NUM_RAW_CAMERA_PARAMS = 3
_IMAGE_SCALE = 0.25
logging.getLogger('tensorflow').disabled = True


class TfDataset(object):

    def __init__(self, dataset, split, tf_files_path):
        self.dataset = dataset
        self.split = split
        self.file_paths = tf_files_path

        self.images = []
        self.views = []
        if dataset == "turk":
            self.texts = []
            self.tokens = []

    def run_process(self):
        num_workers = 1 if self.dataset == "turk" else 30
        lock = Lock()
        self.main_pb = tqdm(total=len(self.file_paths),
                            desc=f"Preparing {self.split} Dataset",
                            unit="tf_file")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for idx, file_path in enumerate(self.file_paths):
                future = executor.submit(self.convert_tf_file, file_path, idx,
                                         lock)
                future.add_done_callback(self.done_callback)

        self.main_pb.close()

    def done_callback(self, future):
        self.main_pb.update(1)

    def convert_tf_file(self, tf_file_path, idx=None, lock=None):
        """Main process for one tfrecord file.

        Args:
            dataset:  TFRecords Dataset
        """

        # Preprocess for each data
        # read tfdataset file, return list of records
        records = list(
            tf.data.TFRecordDataset(str(tf_file_path)).map(self.parse))

        # progress bar
        position = self.get_bar_position()
        tffile_name = tf_file_path.name
        length = len(records)
        with lock:
            records_pb = tqdm(total=length,
                              desc=f"process {tffile_name:13s}",
                              unit="record",
                              position=position,
                              leave=False)

        # process records
        scene_images = []
        scene_views = []
        scene_texts = []
        scene_tokens = []
        for record in records:
            scene_data = preprocess_data(record)
            scene_images.append(scene_data[0])
            scene_views.append(scene_data[1])
            if self.dataset == "turk":
                scene_texts.append(scene_data[2])
                scene_tokens.append(scene_data[3])
            with lock:
                records_pb.update(1)

        # save outputs
        with lock:
            self.images.extend(scene_images)
            self.views.extend(scene_views)
            if self.dataset == "turk":
                self.texts.extend(scene_texts)
                self.tokens.extend(scene_tokens)

        return idx

    def get_bar_position(self):
        thread, w_id = current_thread().name.split("_")
        thread_id = thread.split("-")[-1]
        position = int(w_id) + int(thread_id) + 1

        return position

    def parse(self, buf: tf.Tensor) -> dict:
        """Parse binary protocol buffer into tensors.

        The protocol    buffer is expected to contain the following fields:
            * frames:   10 views of the scene rendered as images.
            * cameras:  10 vectors describing the camera position from which
                        the frames have been rendered
            * captions: A string description of the scene. For the natural
                        language dataset, contains descriptions written by
                        human annotators. For synthetic data contains a string
                        describing each relation  between objects in the scene
                        exactly once.
            * OTHERS:   Other fields are in the buffer. For more info see:
                        https://github.com/deepmind/slim-dataset/blob/0022ab07c692a630afe5144d8d369d29d7e97172/reader.py#L60-L84

        Args:
            buf: A string containing the serialized protocol buffer.

        Returns:
            A dictionary containing tensors for the frames fields. The
            dictionary will contain cameras and captions if the buffer is
            belong to the turk dataset.
        """

        feature_map = {
            "frames":
            tf.io.FixedLenFeature(shape=[_NUM_VIEWS], dtype=tf.string),
            "cameras":
            tf.io.FixedLenFeature(shape=[_NUM_VIEWS * _NUM_RAW_CAMERA_PARAMS],
                                  dtype=tf.float32)
        }

        example = tf.io.parse_single_example(buf, feature_map)

        images = tf.concat(example["frames"], axis=0)
        images = tf.nest.map_structure(
            tf.stop_gradient,
            tf.map_fn(tf.image.decode_jpeg,
                      tf.reshape(images, [-1]),
                      dtype=tf.uint8))

        cameras = tf.reshape(example["cameras"],
                             shape=[-1, _NUM_RAW_CAMERA_PARAMS])

        data_tensors = {"images": images, "cameras": cameras}

        if self.dataset == "turk":
            feature_map = {"captions": tf.io.VarLenFeature(dtype=tf.string)}

            example = tf.io.parse_single_example(buf, feature_map)
            captions = tf.sparse.to_dense(example["captions"],
                                          default_value="")

            data_tensors["captions"] = captions

        return data_tensors


def np_to_list_str(str_byte_array):
    byte_decoder = np.vectorize(lambda x: x.decode())
    np2str = np.vectorize(lambda x: str(x))

    str_decoded = byte_decoder(str_byte_array)
    list_str = np2str(str_decoded.reshape(-1)).tolist()
    return list_str


def tokenize(text):
    return [token
            for token in re.split(r"(\W)", text) if token.strip()] + ["<eos>"]


def preprocess_data(tensor_dict: Dict[str, tf.Tensor]) -> tuple:
    """Converts raw data to tensor and saves into torch gziped file.

    Args:
        raw_data (tf.Tensor): Buffer.
    """

    # Preprocess

    # Frames size: (10, 32, 32, 3) ==> (10, 32, 32, 3)
    frames = _preprocess_images(tensor_dict["images"]).numpy()
    frames = frames.transpose(0, 3, 1, 2)

    # cameras size: (10)
    cameras = _preprocess_cameras(tensor_dict["cameras"]).numpy()

    if "captions" in tensor_dict:
        # convert np array of bytes to list of str
        # captions size: (B)
        captions = tensor_dict["captions"].numpy()
        texts = np_to_list_str(captions)
        tokens = [tokenize(t.lower()) for t in texts]  # tokenize

        returned_values = (frames, cameras, texts, tokens)
    else:
        returned_values = (frames, cameras, None, None)

    return returned_values


def _convert_and_resize_images(images: tf.Tensor,
                               old_size: tf.Tensor) -> tf.Tensor:
    """Resizes images with `_IMAGE_SCALE` ratio."""

    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    new_size = tf.cast(old_size, tf.float32) * _IMAGE_SCALE
    new_size = tf.cast(new_size, tf.int32)
    images = tf.image.resize(images, new_size, preserve_aspect_ratio=True)
    return images


def _preprocess_images(images: tf.Tensor) -> tf.Tensor:
    old_size = tf.shape(images)[1:3]
    images = _convert_and_resize_images(images, old_size)
    return images


def _preprocess_cameras(raw_cameras: tf.Tensor) -> tf.Tensor:
    azimuth = raw_cameras[:, 0]
    # pos = raw_cameras[:, 1:]
    cameras = tf.concat([
        tf.expand_dims(tf.cos(azimuth), -1),
        tf.expand_dims(tf.sin(azimuth), -1)
    ],
                        axis=1)
    return cameras


def get_data(tf_dir, dataset, mode):

    # Setting some paths
    print("Reading dataset...")
    # File list of original dataset
    tf_files_path = sorted(tf_dir.glob("*.tfrecord"))
    tfdataset = TfDataset(dataset, mode, tf_files_path)  # construct tfdataset
    tfdataset.run_process()  # process dataset

    return tfdataset


if __name__ == "__main__":
    # tf.enable_eager_execution()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Specify dataset name
    parser = argparse.ArgumentParser(description="Convert tfrecord to torch")
    parser.add_argument("--dataset_path",
                        type=str,
                        default="/srv/data/zarzouram/lt2318/slim/original/",
                        help="Dataset path.")

    parser.add_argument("--dataset",
                        type=str,
                        default="turk",
                        help="Dataset name {turk, synth}.")

    parser.add_argument("--mode",
                        type=str,
                        default="test",
                        help="Mode {train, test, valid}")

    parser.add_argument(
        "--min_freq",
        type=int,
        default=3,
        help="minimum frequency needed to include a token in the vocabulary")

    args = parser.parse_args()

    # Pathes
    dataset_path = args.dataset_path
    dataset = args.dataset
    mode = args.mode

    dataset_path = os.path.expanduser(dataset_path)
    tf_dir = pathlib.Path(f"{dataset_path}/{dataset}/{mode}/")
    if not tf_dir.exists():
        raise FileNotFoundError(f"TFRecord path `{tf_dir}` does not exists.")

    # make dir to save data if not exist
    dataset_parent_path = pathlib.Path(f"{dataset_path}").parent
    torch_dir = pathlib.Path(f"{dataset_parent_path}/{dataset}_torch/{mode}/")
    torch_dir.mkdir(parents=True, exist_ok=True)

    # get dataset
    ds = get_data(tf_dir, dataset, mode)
    images = np.stack(ds.images)
    cameras = np.stack(ds.views)

    if args.dataset == "turk":
        # build vocab
        words = list(chain.from_iterable(chain.from_iterable(ds.tokens)))
        bow = Counter(words)
        bow_dict = OrderedDict(bow.most_common())

    # save images
    print("writing data to the desk...")
    with h5py.File(str(torch_dir / "images.hdf5"), "w") as h5f:
        images_ds = h5f.create_group("images")
        images_ds.create_dataset(name=f"{dataset}_{mode}",
                                 data=images,
                                 shape=images.shape,
                                 dtype=images.dtype,
                                 compression="gzip",
                                 compression_opts=9)

        cameras_ds = h5f.create_group("cameras")
        cameras_ds.create_dataset(name=f"{dataset}_{mode}",
                                  data=cameras,
                                  shape=cameras.shape,
                                  dtype=cameras.dtype,
                                  compression="gzip",
                                  compression_opts=9)

    print("writing finished.")

    print("Done")
