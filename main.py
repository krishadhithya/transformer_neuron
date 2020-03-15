#!/bin/python
import argparse
import tensorflow_datasets as tfds

from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder
from loguru import logger

from utils import (
    load_data_and_constants, 
    SPT
)

PRC_DATA_FPATH = "data/proc_dataset.txt" 

def prepare_dataset(hparams):
    dataset, hparams = load_data_and_constants(hparams)
    
    logger.info("Dataset: {}".format(dataset['train'][0]))
    
    total_lines = len(dataset['train'])
    
    with open(PRC_DATA_FPATH, "w", encoding="utf-8") as prc_file:
        for data in dataset['train']:
            for sample in data:
                prc_file.write(sample['sentence'] + '\n')
        
    spt = SPT(PRC_DATA_FPATH)
    spt.train_sentencetrainer()
    

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
        help='Dataset on which to run the network. Default is mnist.'
    )
    parser.add_argument(
        '--corpus_size',
        default=1000000,
        type=str,
        help='Dataset corpus size.'
    )

    hparams = parser.parse_args()
    prepare_dataset(hparams)