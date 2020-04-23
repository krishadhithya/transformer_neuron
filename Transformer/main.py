import argparse
import tensorflow as tf
import time

from Transformer.WarmupThenDecaySchedule import WarmupThenDecaySchedule
from utils import (
    download_and_read_file
)
from model import Transformer
from loguru import logger
from timeloop import Timeloop
from datetime import timedelta
from metagraph import Metagraph
from neuron import Neuron

tf.compat.v1.enable_eager_execution()

def set_timed_loops(tl, config, neuron, metagraph):
    # Pull updated graph state
    @tl.job(interval=timedelta(seconds=7))
    def pull_metagraph():
        metagraph.pull_metagraph()
    
    # Publish attributions
    @tl.job(interval=timedelta(seconds=3))
    def publish_attributions():
        metagraph.publish_attributions()
    
    # Reselect channels
    @tl.job(interval=timedelta(seconds=10))
    def connect():
        neuron.connect()
    
def main(hparams):
    
    logger.info("Establishing Metagraph Component...")
    metagraph = Metagraph(hparams)
    
    logger.info("Building Transformer Components...")
    
    logger.info("Transforming Dataset...")
    lines = download_and_read_file(hparams.dataset)
    
    logger.info("Building Transformer...")
    nucleus = Transformer(hparams, lines)

    neuron = Neuron(hparams, nucleus, metagraph)
    neuron.serve()
    
    tl = Timeloop()
    set_timed_loops(tl, hparams, neuron, metagraph)
    tl.start(block=False)
    logger.info("Started timers...")    
    
    def tear_down(_hparams, _neuron, _nucleus, _metagraph):
        logger.debug("Tear down...")
        del _neuron
        del _nucleus
        del _metagraph
        del _hparams
    
    try:
        logger.info("Begin wait on main...")
        while True:
            logger.debug('heartbeat')
            time.sleep(100)
    except KeyboardInterrupt:
        logger.debug("Neuron stopped with keyboard interrupt.")
        tear_down(hparams, neuron, nucleus, metagraph)
    
    except Exception as e:
        logger.error("Neuron stopped with interrupt on error: {}".format(e))
        tear_down(hparams, neuron, nucleus, metagraph)


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
    
    parser.add_argument(
        '--eosurl',
        default='http://0.0.0.0:8888',
        type=str,
        help="Address to eos chain. Default eosurl=http://0.0.0.0:8888")
    
    parser.add_argument(
        '--identity',
        default='abcd',
        type=str,
        help="network identity. Default identity=abcd")
    
    parser.add_argument(
        '--n_children',
        default=2,
        type=int,
        help='The number of graph neighbors. Default n_children=2')
    
    parser.add_argument(
        '--bind_address',
        default='0.0.0.0',
        type=str,
        help="Address to bind neuron. Default bind_address=0.0.0.0")
    
    parser.add_argument(
        '--serve_address',
        default='0.0.0.0',
        type=str,
        help="Address to server neuron. Default serve_address=0.0.0.0")
    
    parser.add_argument(
        '--port',
        default='9090',
        type=str,
        help="Port to serve neuron on. Default port=9090")

    parser.add_argument(
        '--logdir',
        default="/tmp/",
        type=str,
        help="logging output directory. Default logdir=/tmp/")

        
    hparams = parser.parse_args()
    main(hparams)
    
    