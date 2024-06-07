# Lint as: python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs an Image Classification model."""

import datetime
import json
import os
import pprint
import random
from typing import Any, Mapping, Optional, Text, Tuple

import numpy as np
import requests
import tensorflow as tf
from absl import app, flags, logging
from official.common import distribute_utils
from official.modeling import hyperparams, performance
from official.utils import hyperparams_flags
from official.utils.misc import keras_utils
from official.vision.image_classification import callbacks as custom_callbacks
from official.vision.image_classification import (dataset_factory,
                                                  optimizer_factory)
from official.vision.image_classification.configs import base_configs, configs
from official.vision.image_classification.efficientnet import \
    efficientnet_model
from official.vision.image_classification.resnet import common, resnet_model
from sklearn.metrics import accuracy_score


def get_image_size_from_model(params: base_configs.ExperimentConfig) -> Optional[int]:
    """If the given model has a preferred image size, return it."""
    if params.model_name == "efficientnet":
        efficientnet_name = params.model.model_params.model_name
        if efficientnet_name in efficientnet_model.MODEL_CONFIGS:
            return efficientnet_model.MODEL_CONFIGS[efficientnet_name].resolution
    return None


def _get_dataset_builders(
    params: base_configs.ExperimentConfig,
    strategy: tf.distribute.Strategy,
    one_hot: bool,
) -> Tuple[Any, Any]:
    """Create and return train and validation dataset builders."""
    if one_hot:
        logging.warning("label_smoothing > 0, so datasets will be one hot encoded.")
    else:
        logging.warning(
            "label_smoothing not applied, so datasets will not be one " "hot encoded."
        )

    num_devices = strategy.num_replicas_in_sync if strategy else 1

    image_size = get_image_size_from_model(params)

    dataset_configs = [params.train_dataset, params.validation_dataset]
    builders = []

    for config in dataset_configs:

        if config is not None and config.has_data:
            builder = dataset_factory.DatasetBuilder(
                config,
                image_size=image_size or config.image_size,
                num_devices=num_devices,
                one_hot=one_hot,
            )
        else:
            builder = None
        builders.append(builder)

    return builders


def _get_params_from_flags(flags_obj: flags.FlagValues):
    """Get ParamsDict from flags."""
    model = flags_obj.model_type.lower()
    dataset = flags_obj.dataset.lower()
    params = configs.get_config(model=model, dataset=dataset)

    flags_overrides = {
        "model_dir": flags_obj.model_dir,
        "mode": flags_obj.mode,
        "model": {
            "name": model,
        },
        "runtime": {
            "run_eagerly": flags_obj.run_eagerly,
            "tpu": flags_obj.tpu,
        },
        "train_dataset": {
            "data_dir": flags_obj.data_dir,
        },
        "validation_dataset": {
            "data_dir": flags_obj.data_dir,
        },
        "train": {
            "time_history": {
                "log_steps": flags_obj.log_steps,
            },
        },
    }

    overriding_configs = (
        flags_obj.config_file,
        flags_obj.params_override,
        flags_overrides,
    )

    pp = pprint.PrettyPrinter()

    logging.info("Base params: %s", pp.pformat(params.as_dict()))

    for param in overriding_configs:
        logging.info("Overriding params: %s", param)
        params = hyperparams.override_params_dict(params, param, is_strict=True)

    params.validate()
    params.lock()

    logging.info("Final model parameters: %s", pp.pformat(params.as_dict()))
    return params


def define_classifier_flags():
    """Defines common flags for image classification."""
    hyperparams_flags.initialize_common_flags()
    flags.DEFINE_string(
        "data_dir", default=None, help="The location of the input data."
    )
    flags.DEFINE_string(
        "mode",
        default=None,
        help="Mode to run: `train`, `eval`, `train_and_eval` or `export`.",
    )
    flags.DEFINE_bool(
        "run_eagerly",
        default=None,
        help="Use eager execution and disable autograph for debugging.",
    )
    flags.DEFINE_string(
        "model_type",
        default=None,
        help="The type of the model, e.g. EfficientNet, etc.",
    )
    flags.DEFINE_string(
        "dataset", default=None, help="The name of the dataset, e.g. ImageNet, etc."
    )
    flags.DEFINE_integer(
        "log_steps",
        default=100,
        help="The interval of steps between logging of batch level stats.",
    )


def eval(
    params: base_configs.ExperimentConfig, strategy_override: tf.distribute.Strategy
) -> Mapping[str, Any]:
    """Runs the train and eval path using compile/fit."""
    logging.info("Running train and eval.")

    distribute_utils.configure_cluster(
        params.runtime.worker_hosts, params.runtime.task_index
    )

    # Note: for TPUs, strategy and scope should be created before the dataset
    strategy = strategy_override or distribute_utils.get_distribution_strategy(
        distribution_strategy=params.runtime.distribution_strategy,
        all_reduce_alg=params.runtime.all_reduce_alg,
        num_gpus=0,
        tpu_address=params.runtime.tpu,
    )

    logging.info(
        "Detected %d devices.", strategy.num_replicas_in_sync if strategy else 1
    )

    label_smoothing = params.model.loss.label_smoothing
    one_hot = label_smoothing and label_smoothing > 0

    builders = _get_dataset_builders(params, strategy, one_hot)
    datasets = [builder.build() if builder else None for builder in builders]

    # Unpack datasets and builders based on train/val/test splits
    _, validation_builder = builders  # pylint: disable=unbalanced-tuple-unpacking
    _, validation_dataset = datasets
    validation_steps = params.evaluation.steps or validation_builder.num_steps
    cur_step = 0
    stats = {}
    y_true = []
    y_pred = []
    total_elapsed_time = 0
    for imgs, labels in validation_dataset:
        xs = (imgs.numpy() / 255).tolist()
        print(imgs.numpy().shape)
        data = json.dumps({"signature_name": "serving_default", "instances": xs})
        headers = {"content-type": "application/json"}
        start = datetime.datetime.now()
        json_response = requests.post(
            "http://localhost:8501/v1/models/resnet:predict", data=data, headers=headers
        )
        print(json_response)
        end = datetime.datetime.now()
        total_elapsed_time += (end - start).total_seconds() * 1000
        predictions = __resp_to_nparray(json_response)
        y_true.extend(np.argmax(labels, axis=1).tolist())
        y_pred.extend(predictions.tolist())
        cur_step += 1
        if cur_step == validation_steps:
            break
    print(y_true)
    print(y_pred)
    acc = accuracy_score(y_true, y_pred)
    stats["acc"] = acc
    stats["elapsed time"] = total_elapsed_time
    return stats


def __resp_to_nparray(resp):
    prob = json.loads(resp.text)["predictions"]
    prob = np.array(prob)
    pred = np.argmax(prob, axis=1) - 1

    return pred


def run(
    flags_obj: flags.FlagValues, strategy_override: tf.distribute.Strategy = None
) -> Mapping[str, Any]:
    """Runs Image Classification model using native Keras APIs.

    Args:
      flags_obj: An object containing parsed flag values.
      strategy_override: A `tf.distribute.Strategy` object to use for model.

    Returns:
      Dictionary of training/eval stats
    """
    params = _get_params_from_flags(flags_obj)

    print(eval(params, strategy_override))


def main(_):
    stats = run(flags.FLAGS)
    if stats:
        logging.info("Run stats:\n%s", stats)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)
    logging.set_verbosity(logging.INFO)
    define_classifier_flags()
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("dataset")

    app.run(main)
