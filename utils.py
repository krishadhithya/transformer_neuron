from loguru import logger
import numpy as np
import tensorflow_datasets as tfds


def load_data_and_constants(hparams):
    '''Returns the dataset and sets hparams.n_inputs and hparamsn_targets.'''
    logger.info("Loading dataset '{}'".format(hparams.dataset))
    # Load mnist data

    dataset_builder = tfds.builder(hparams.dataset)
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    dataset_info = dataset_builder.info

    # Load data features to extract shapes and targets
    dataset_features = dataset_info.features
    inputs_shape = [i for i in dataset_features['sentence'].shape if i is not None]

    hparams.n_inputs = np.prod(inputs_shape)
    hparams.n_targets = dataset_features['label'].num_classes

    dataset = prepare_dataset(dataset, hparams.batch_size, hparams.n_inputs,
                              hparams.n_targets)

    return dataset, hparams

def prepare_dataset(dataset, batch_size, n_inputs, n_targets):
    pass