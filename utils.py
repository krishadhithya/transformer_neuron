from loguru import logger
import numpy as np
import tensorflow_datasets as tfds
import nltk
import sentencepiece as spm

class SPT:
    MODEL_PREFIX = "tokenizer" #@param {type: "string"}
    VOC_SIZE = 32000 #@param {type:"integer"}
    SUBSAMPLE_SIZE = 12800000 #@param {type:"integer"}
    NUM_PLACEHOLDERS = 256 #@param {type:"integer"}

    def __init__(self, PRC_DATA_FPATH):
        self.SPM_COMMAND = ('--input={} --model_prefix={} '
                '--vocab_size={} --input_sentence_size={} '
                '--shuffle_input_sentence=true ' 
                '--bos_id=-1 --eos_id=-1 --hard_vocab_limit=false').format(
                PRC_DATA_FPATH, self.MODEL_PREFIX, 
                self.VOC_SIZE - self.NUM_PLACEHOLDERS, self.SUBSAMPLE_SIZE)

    def train_sentencetrainer(self):
        spm.SentencePieceTrainer.Train(self.SPM_COMMAND)
        
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

    dataset = prepare_dataset(dataset, hparams.batch_size)

    return dataset, hparams

def normalize_text(text):
    regex_tokenizer = nltk.RegexpTokenizer("\w+")
    # lowercase text
    text = str(text).lower()
    # remove non-UTF
    text = text.encode("utf-8", "ignore").decode()
    # remove punktuation symbols
    text = " ".join(regex_tokenizer.tokenize(text))
    
    #text = "[CLS] {} [SEP]".format(text)
    
    return text

def prepare_dataset(dataset, batch_size):
    training_data = list(tfds.as_numpy(dataset['train']))
    testing_data = list(tfds.as_numpy(dataset['test']))
    train_set = []
    test_set = []
    
    for train in range(0, len(training_data), batch_size):
        train_batch = []

        local_batch_limit = train + batch_size
        if local_batch_limit > len(training_data):
            batch_size = abs(len(training_data) - train)

        for i in range(train, train + batch_size):
            training_data[i]['sentence'] = normalize_text(training_data[i]['sentence'].decode("utf-8"))
            train_batch.append(training_data[i])
        train_set.append(train_batch)
            
    
    for test in range(0, len(testing_data)):
        testing_data[test]['sentence'] = normalize_text(testing_data[test]['sentence'].decode("utf-8"))
        test_set.append(testing_data[test])
    
    dataset['train'] = train_set
    dataset['test'] = test_set
    
    return dataset
    