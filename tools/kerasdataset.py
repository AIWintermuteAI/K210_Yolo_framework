import tensorflow as tf
import numpy as np
from tools.base import BaseHelper
from typing import Tuple
from tools.dcasetask5 import FixMatchSSLHelper
from transforms.image.rand_augment import RandAugment
from transforms.image.ct_augment import CTAugment


class KerasDatasetHelper(BaseHelper, FixMatchSSLHelper):

  def __init__(self, dataset: str, label_ratio: float, unlabel_dataset_ratio: int,
               augment_kwargs: dict):
    self.train_dataset: tf.data.Dataset = None
    self.val_dataset: tf.data.Dataset = None
    self.test_dataset: tf.data.Dataset = None

    self.train_epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None
    dataset_dict = {
        'mnist': tf.keras.datasets.mnist,
        'cifar10': tf.keras.datasets.cifar10,
        'cifar100': tf.keras.datasets.cifar100,
        'fashion_mnist': tf.keras.datasets.fashion_mnist
    }
    if dataset == None:
      self.train_list: str = None
      self.val_list: str = None
      self.test_list: str = None
      self.unlabel_list: str = None
    else:
      assert dataset in dataset_dict.keys(), 'dataset is invalid!'
      (x_train, y_train), (x_test, y_test) = dataset_dict[dataset].load_data()
      # NOTE can use dict set trian and test dataset
      label_set = set(y_train.ravel())
      label_idxs = []
      unlabel_idxs = []
      for l in label_set:
        idxes = np.where(y_train == l)[0]
        label_idxs.append(idxes[:int(len(idxes) * label_ratio)])
        unlabel_idxs.append(idxes[int(len(idxes) * label_ratio):])
      label_idxs = np.concatenate(label_idxs, 0)
      unlabel_idxs = np.concatenate(unlabel_idxs, 0)

      self.train_list: Tuple[np.ndarray, np.ndarray] = (x_train[label_idxs],
                                                        y_train[label_idxs])
      self.unlabel_list: Tuple[np.ndarray, np.ndarray] = (x_train[unlabel_idxs],
                                                          y_train[unlabel_idxs])
      self.val_list: Tuple[np.ndarray, np.ndarray] = (x_test, y_test)
      self.test_list: Tuple[np.ndarray, np.ndarray] = None
      self.train_total_data: int = len(label_idxs)
      self.unlabel_total_data: int = len(unlabel_idxs)
      self.val_total_data: int = len(x_test)
      self.test_total_data: int = None

    self.in_hw: list = list(x_train.shape[1:])
    self.nclasses = len(label_set)
    self.unlabel_dataset_ratio = unlabel_dataset_ratio
    tmp = self.create_augmenter(**augment_kwargs)
    self.augmenter: CTAugment = tmp[0]
    self.sup_aug_fn = tmp[1]
    self.unsup_aug_fn = tmp[2]

  @staticmethod
  def create_augmenter(name: str, kwarg: dict):
    if not name or (name == 'none') or (name == 'noop'):
      return (lambda x: x)
    elif name == 'randaugment':
      base_augmenter = RandAugment(**kwarg)
      return (None, lambda data: {
          'data': data
      }, lambda x: base_augmenter(x, aug_key='aug_data'))
    elif name == 'ctaugment':
      base_augmenter = CTAugment(**kwarg)
      return (base_augmenter,
              lambda x: base_augmenter(x, probe=True, aug_key=None),
              lambda x: base_augmenter(x, probe=False, aug_key='aug_data'))
    else:
      raise ValueError('Invalid augmentation type {0}'.format(name))

  def build_train_datapipe(self, batch_size: int,
                           is_augment: bool) -> tf.data.Dataset:

    def label_pipe(img: tf.Tensor, label: tf.Tensor):
      img.set_shape(self.in_hw)
      if is_augment:
        data_dict = self.sup_aug_fn(img)
      else:
        data_dict = {'data': img}

      data_dict['label'] = label
      return data_dict

    def unlabel_pipe(img: tf.Tensor, label: tf.Tensor):
      img.set_shape(self.in_hw)
      if is_augment:
        data_dict = self.unsup_aug_fn(img)
      else:
        data_dict = {'data': img}

      data_dict['label'] = label
      return data_dict

    label_ds = tf.data.Dataset.from_tensor_slices(self.train_list).shuffle(
        batch_size * 300).repeat().map(label_pipe, -1).batch(
            batch_size, drop_remainder=True)
    unlabel_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(
        self.unlabel_list).shuffle(batch_size * 300).repeat().map(
            unlabel_pipe, -1).batch(
                batch_size * self.unlabel_dataset_ratio, drop_remainder=True)

    ds = tf.data.Dataset.zip((label_ds, unlabel_ds)).map(
        self._combine_sup_unsup_datasets).prefetch(tf.data.experimental.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    return ds

  def build_val_datapipe(self, batch_size: int) -> tf.data.Dataset:

    def _pipe(img: tf.Tensor, label: tf.Tensor):
      img.set_shape(self.in_hw)
      return {'data': img, 'label': label}

    ds: tf.data.Dataset = (
        tf.data.Dataset.from_tensor_slices(self.val_list).map(
            _pipe, num_parallel_calls=-1).batch(batch_size).prefetch(None))
    return ds

  def set_dataset(self, batch_size, is_augment):
    self.batch_size = batch_size
    self.train_dataset = self.build_train_datapipe(batch_size, is_augment)
    self.val_dataset = self.build_val_datapipe(batch_size)
    self.train_epoch_step = self.train_total_data // self.batch_size
    self.val_epoch_step = self.val_total_data // self.batch_size
