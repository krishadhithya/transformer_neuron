#!/bin/python
import argparse
import tensorflow_datasets as tfds

from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder
from loguru import logger
from utils import load_data_and_constants


def main(hparams):
    logger.info(tfds.list_builders())
    dataset, hparams = load_data_and_constants(hparams)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--batch_size',
        default=50,
        type=int,
        help='The number of examples per batch. Default batch_size=128')
    parser.add_argument(
        '--learning_rate',
        default=1e-5,
        type=float,
        help='Component learning rate. Default learning_rate=1e-4')
    parser.add_argument(
        '--log_dir',
        default='logs',
        type=str,
        help='location of tensorboard logs. Default log_dir=logs'
    )
    parser.add_argument(
        '--n_train_steps',
        default=10000000,
        type=int,
        help='Training steps. Default n_train_steps=1000000'
    )
    parser.add_argument(
        '--dataset',
        default='mnist',
        type=str,
        help='Dataset on which to run the network. Default is mnist.')

    hparams = parser.parse_args()
    main(hparams)