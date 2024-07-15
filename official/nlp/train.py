# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""TFM common training driver."""
import os
import random

import numpy as np

SEED = 0

import gin
import tensorflow as tf
from absl import app, flags, logging
# pylint: enable=unused-import
# pylint: disable=unused-import
from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.common import registry_imports
from official.core import task_factory, train_lib, train_utils
from official.modeling import performance
from official.nlp import continuous_finetune_lib

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'pretrain_steps',
    default=None,
    help='The number of total training steps for the pretraining job.')


def _run_experiment_with_preemption_recovery(params, model_dir):
  """Runs experiment and tries to reconnect when encounting a preemption."""
  keep_training = True
  while keep_training:
    preemption_watcher = None
    try:
      distribution_strategy = distribute_utils.get_distribution_strategy(
          distribution_strategy=params.runtime.distribution_strategy,
          all_reduce_alg=params.runtime.all_reduce_alg,
          num_gpus=params.runtime.num_gpus,
          tpu_address=params.runtime.tpu,
          **params.runtime.model_parallelism())
      with distribution_strategy.scope():
        task = task_factory.get_task(params.task, logging_dir=model_dir)
      preemption_watcher = tf.distribute.experimental.PreemptionWatcher()

      train_lib.run_experiment(
          distribution_strategy=distribution_strategy,
          task=task,
          mode=FLAGS.mode,
          params=params,
          model_dir=model_dir)

      keep_training = False
    except tf.errors.OpError as e:
      if preemption_watcher and preemption_watcher.preemption_message:
        preemption_watcher.block_until_worker_exit()
        logging.info(
            'Some TPU workers had been preempted (message: %s), '
            'retarting training from the last checkpoint...',
            preemption_watcher.preemption_message)
        keep_training = True
      else:
        raise e from None


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = train_utils.parse_configuration(FLAGS)
  model_dir = FLAGS.model_dir
  if 'train' in FLAGS.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  if FLAGS.mode == 'continuous_train_and_eval':
    continuous_finetune_lib.run_continuous_finetune(
        FLAGS.mode, params, model_dir, pretrain_steps=FLAGS.pretrain_steps)

  else:
    # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
    # can have significant impact on model speeds by utilizing float16 in case
    # of GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only
    # when dtype is float16
    if params.runtime.mixed_precision_dtype:
      performance.set_mixed_precision_policy(
          params.runtime.mixed_precision_dtype)
    _run_experiment_with_preemption_recovery(params, model_dir)

  train_utils.save_gin_config(FLAGS.mode, model_dir)


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    tf.config.optimizer.set_jit(False)



if __name__ == '__main__':
  set_global_determinism(seed=SEED)

  tfm_flags.define_flags()
  flags.mark_flags_as_required(['experiment', 'mode', 'model_dir'])
  app.run(main)
