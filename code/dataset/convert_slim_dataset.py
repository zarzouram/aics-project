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

import argparse
import logging
import os
import pathlib

import re
from tqdm import tqdm
# from tqdm.contrib.concurrent import process_map
# from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import current_thread

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import torch

import numpy as np

_NUM_VIEWS = 10
_NUM_RAW_CAMERA_PARAMS = 3
_IMAGE_SCALE = 0.25
_USE_SIMPLIFIED_CAPTIONS = False
_PARSE_METADATA = True
logging.getLogger('tensorflow').disabled = True


class TfDataset(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.images = [None]
        if dataset == "turk":
            self.views = [None]
            self.texts = [None]
            self.tokens = [None]

    def convert_tf_file(self, tf_file_path, idx=None):
        """Main process for one tfrecord file.

        Args:
            dataset:  TFRecords Dataset
        """
        # Preprocess for each data
        thread, w_id = current_thread().name.split("_")
        thread_id = thread.split("-")[-1]
        position = int(w_id) + int(thread_id) + 1

        records = list(
            tf.data.TFRecordDataset(str(tf_file_path)).map(self.parse))
        length = len(records)

        for record in tqdm(records,
                           total=length,
                           desc="process files",
                           unit="record",
                           position=position,
                           leave=False):

            scene_data = preprocess_data(record)
            self.images.append(scene_data[0])
            if self.dataset == "turk":
                self.views.append(scene_data[1])
                self.texts.append(scene_data[2])
                self.tokens.append(scene_data[3])

        return idx

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
            "frames": tf.io.FixedLenFeature(shape=[_NUM_VIEWS],
                                            dtype=tf.string)
        }

        example = tf.io.parse_single_example(buf, feature_map)

        images = tf.concat(example["frames"], axis=0)
        images = tf.nest.map_structure(
            tf.stop_gradient,
            tf.map_fn(tf.image.decode_jpeg,
                      tf.reshape(images, [-1]),
                      dtype=tf.uint8))

        data_tensors = {"images": images}

        if self.dataset == "turk":
            feature_map = {
                "cameras":
                tf.io.FixedLenFeature(
                    shape=[_NUM_VIEWS * _NUM_RAW_CAMERA_PARAMS],
                    dtype=tf.float32),
                "captions":
                tf.io.VarLenFeature(dtype=tf.string)
            }

            example = tf.io.parse_single_example(buf, feature_map)

            cameras = tf.reshape(example["cameras"],
                                 shape=[-1, _NUM_RAW_CAMERA_PARAMS])
            captions = tf.sparse.to_dense(example["captions"],
                                          default_value="")

            data_tensors["cameras"] = cameras
            data_tensors["captions"] = captions

        return data_tensors


def np_to_list_str(str_byte_array):
    byte_decoder = np.vectorize(lambda x: x.decode())
    np2str = np.vectorize(lambda x: str(x))

    str_decoded = byte_decoder(str_byte_array)
    list_str = np2str(str_decoded.reshape(-1)).tolist()
    return list_str


def tokenize(text):
    return [token for token in re.split(r"(\W)", text) if token.strip()]


def preprocess_data(tensor_dict: Dict[str, tf.Tensor]) -> tuple:
    """Converts raw data to tensor and saves into torch gziped file.

    Args:
        raw_data (tf.Tensor): Buffer.
    """

    # Preprocess

    # Frames size: (10, 32, 32, 3) ==> (10, 32, 32, 3)
    frames = _preprocess_images(tensor_dict["images"]).numpy()
    frames = frames.transpose(0, 3, 1, 2)
    frames = torch.from_numpy(frames)

    if "cameras" in tensor_dict:
        # cameras size: (10) "in rad"
        cameras = _preprocess_cameras(tensor_dict["cameras"]).numpy()
        cameras = torch.round(torch.from_numpy(cameras), decimals=3)

        # convert np array of bytes to list of str
        # captions size: (B)
        captions = tensor_dict["captions"].numpy()
        texts = np_to_list_str(captions)
        tokens = [tokenize(t.lower()) for t in texts]  # tokenize

        returned_values = (frames, cameras, texts, tokens)
    else:
        returned_values = (frames, None, None, None)

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
    # cameras = tf.concat([
    #     pos,
    #     tf.expand_dims(tf.sin(azimuth), -1),
    #     tf.expand_dims(tf.cos(azimuth), -1),
    # ],
    #                     axis=1)
    return azimuth


def get_data(dataset_path, dataset, mode):

    # Setting some paths
    print("Reading dataset...")
    dataset_path = os.path.expanduser(dataset_path)
    tf_dir = pathlib.Path(f"{dataset_path}/{dataset}/{mode}/")
    if not tf_dir.exists():
        raise FileNotFoundError(f"TFRecord path `{tf_dir}` does not exists.")

    # make dir to save data if not exist
    dataset_parent_path = pathlib.Path(f"{dataset_path}").parent
    torch_dir = pathlib.Path(f"{dataset_parent_path}/{dataset}_torch/{mode}/")
    torch_dir.mkdir(parents=True, exist_ok=True)

    # File list of original dataset
    tf_files_path = sorted(tf_dir.glob("*.tfrecord"))
    tfdataset = TfDataset(dataset)  # construct tfdataset

    # Read tfdataset file by file and process the records
    num_workers = 1 if dataset == "turk" else 10
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(tf_files_path)) as pbar:
            pbar.desc = f"Preparing {mode} Dataset"
            pbar.unit = "tf_file"
            futures = []
            for idx, tf_file_path in enumerate(tf_files_path):
                i = executor.submit(tfdataset.convert_tf_file, tf_file_path,
                                    idx)
                futures.append(i)
            for _ in as_completed(futures):
                pbar.update(1)

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
                        default="train",
                        help="Mode {train, test, valid}")

    args = parser.parse_args()

    ds = get_data(args.dataset_path, args.dataset, args.mode)

    print("Done")
