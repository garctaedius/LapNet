# Copyright 2020 DeepMind Technologies Limited.
# Copyright 2023 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Super simple checkpoints using numpy."""

import datetime
import os
from typing import Optional
import zipfile
import functools

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from .allow_multi_node import is_main_process

def find_last_checkpoint(ckpt_path: Optional[str] = None) -> Optional[str]:
  """Finds most recent valid checkpoint in a directory.

  Args:
    ckpt_path: Directory containing checkpoints.

  Returns:
    Last QMC checkpoint (ordered by sorting all checkpoints by name in reverse)
    or None if no valid checkpoint is found or ckpt_path is not given or doesn't
    exist. A checkpoint is regarded as not valid if it cannot be read
    successfully using np.load.
  """
  if ckpt_path and os.path.exists(ckpt_path):
    files = [f for f in os.listdir(ckpt_path) if 'qmcjax_ckpt_' in f]
    # Handle case where last checkpoint is corrupt/empty.
    for file in sorted(files, reverse=True):
      fname = os.path.join(ckpt_path, file)
      with open(fname, 'rb') as f:
        try:
          np.load(f, allow_pickle=True)
          return fname
        except (OSError, EOFError, zipfile.BadZipFile):
          logging.warning('Error loading checkpoint %s. Trying next checkpoint...',
                       fname)
  return None


def create_save_path(save_path: Optional[str]) -> str:
  """Creates the directory for saving checkpoints, if it doesn't exist.

  Args:
    save_path: directory to use. If false, create a directory in the working
      directory based upon the current time.

  Returns:
    Path to save checkpoints to.
  """
  timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
  default_save_path = os.path.join(os.getcwd(), f'ferminet_{timestamp}')
  ckpt_save_path = save_path or default_save_path
  if is_main_process() and ckpt_save_path and not os.path.isdir(ckpt_save_path):
    os.makedirs(ckpt_save_path)
  return ckpt_save_path


def get_restore_path(restore_path: Optional[str] = None) -> Optional[str]:
  """Gets the path containing checkpoints from a previous calculation.

  Args:
    restore_path: path to checkpoints.

  Returns:
    The path or None if restore_path is falsy.
  """
  if restore_path:
    ckpt_restore_path = restore_path
  else:
    ckpt_restore_path = None
  return ckpt_restore_path


def gather_data(data):
  pgather = functools.partial(jax.lax.all_gather, axis_name="pmap_axis")
  @functools.partial(jax.pmap, axis_name="pmap_axis")
  def gather_electrons(electrons):
     return pgather(electrons, axis=0, tiled=True)
  electrons = gather_electrons(data)
  instance = functools.partial(jax.tree_util.tree_map, lambda x: x[0])
  electrons = instance(electrons)
  return electrons


def save(save_path: str, t: int, data, params, opt_state, mcmc_width, sharded_key) -> str:
  """Saves checkpoint information to a npz file.

  Args:
    save_path: path to directory to save checkpoint to. The checkpoint file is
      save_path/qmcjax_ckpt_$t.npz, where $t is the number of completed
      iterations.
    t: number of completed iterations.
    data: MCMC walker configurations.
    params: pytree of network parameters.
    opt_state: optimization state.
    mcmc_width: width to use in the MCMC proposal distribution.
    sharded_key (chex.PRNGKey): JAX RNG state.

  Returns:
    path to checkpoint file.
  """
  combined_data = gather_data(data)

  if not is_main_process():
    return
  ckpt_filename = os.path.join(save_path, f'qmcjax_ckpt_{t:06d}.npz')
  logging.info('Saving checkpoint %s', ckpt_filename)
  instance = functools.partial(jax.tree_util.tree_map, lambda x: x[0])
  with open(ckpt_filename, 'wb') as f:
    np.savez(
        f,
        t=t,
        data=combined_data,
        params=instance(params),
        opt_state=instance(opt_state),
        mcmc_width=mcmc_width,
        sharded_key=sharded_key)

  return ckpt_filename

def restore(restore_filename: str, batch_size: Optional[int] = None):
  """Restores data saved in a checkpoint.

  Args:
    restore_filename: filename containing checkpoint.
    batch_size: total batch size to be used. If present, check the data saved in
      the checkpoint is consistent with the batch size requested for the
      calculation.

  Returns:
    (t, data, params, opt_state, mcmc_width) tuple, where
    t: number of completed iterations.
    data: MCMC walker configurations.
    params: pytree of network parameters.
    opt_state: optimization state.
    mcmc_width: width to use in the MCMC proposal distribution.

  Raises:
    ValueError: if the leading dimension of data does not match the number of
    devices (i.e. the number of devices being parallelised over has changed) or
    if the total batch size is not equal to the number of MCMC configurations in
    data.
  """
  logging.info('Loading checkpoint %s', restore_filename)
  with open(restore_filename, 'rb') as f:
    ckpt_data = np.load(f, allow_pickle=True)
    # Retrieve data from npz file. Non-array variables need to be converted back
    # to natives types using .tolist().
    t = ckpt_data['t'].tolist() + 1  # Return the iterations completed.
    combined_data = ckpt_data['data']
    params = ckpt_data['params'].tolist()
    opt_state = ckpt_data['opt_state'].tolist()
    mcmc_width = jnp.array(ckpt_data['mcmc_width'].tolist())
    sharded_key = ckpt_data['sharded_key'] if 'sharded_key' in ckpt_data else None

    params = jax.tree_util.tree_map(lambda x: x[None, ...], params)
    opt_state = jax.tree_util.tree_map(lambda x: x[None, ...], opt_state)
    
    num_devices = jax.process_count() 
    sharded_key = jax.random.split(sharded_key[0], num_devices)
    sharded_key = sharded_key[jax.process_index()]

    if combined_data.shape[0] != batch_size*num_devices:
      raise ValueError(
          'Wrong batch size in loaded data. Expected {}, found {}.'.format(
              batch_size*num_devices, combined_data.shape[0]))

    data = combined_data.reshape(jax.process_count(), jax.local_device_count(), -1, *combined_data.shape[1:])
    data = data[jax.process_index()]
  logging.info("finished restoring")
  return t, data, params, opt_state, mcmc_width, sharded_key
