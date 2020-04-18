import argparse
import tensorflow as tf
from Transformer.WarmupThenDecaySchedule import WarmupThenDecaySchedule
from utils import (
    download_and_read_file
)
from model import Transformer
from loguru import logger

tf.compat.v1.enable_eager_execution()


def main(hparams):
    
    logger.info("Building Components..")
    
    # Build components.
    components = []
    for i in range(hparams.n_components):
        # Get raw data
        lines = download_and_read_file(hparams.dataset)
        
        # Setup transformer network
        model = Transformer(hparams, lines)
        model.train_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--batch_size',
        default=50,
        type=int,
        help='The number of examples per batch. Default batch_size=128'
    )
    parser.add_argument(
        '--learning_rate',
        default=WarmupThenDecaySchedule(128),
        type=float,
        help='Component learning rate. Default .'
    )
    parser.add_argument(
        '--model_size',
        default=128,
        type=int,
        help='Size of the model.'
    )
    parser.add_argument(
        '--attention_heads',
        default=8,
        type=int,
        help='The number of attention heads in the Encoder and Decoder.'
    )
    parser.add_argument(
        '--num_layers',
        default=8,
        type=int,
        help='The number of hidden layers in the FF network.'
    )
    parser.add_argument(
        '--dataset',
        default='data/fra-eng.zip',
        type=str,
        help='Dataset on which to run the model.'
    )
    parser.add_argument(
        '--num_epochs',
        default=15,
        type=int,
        help='Number of epochs to train in.'
    )
    parser.add_argument(
        '--n_components',
        default=1,
        type=int,
        help='The number of running components. Default n_components=2')
    
    hparams = parser.parse_args()
    main(hparams)
    
    