import tensorflow as tf
import numpy as np
import os
import requests
from zipfile import ZipFile
import unicodedata
from Decoder import Decoder
from Encoder import Encoder
from MultiHeadAttention import MultiHeadAttention
from loguru import logger
import re

URL = 'https://www.manythings.org/anki/fra-eng.zip'
FILENAME = 'fra-eng.zip'
NUM_EPOCHS = 15

def download_and_read_file(url, filename):
    if not os.path.exists(filename):
        session = requests.Session()
        response = session.get(url, stream=True)

        CHUNK_SIZE = 32768
        with open(filename, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    zipf = ZipFile(filename)
    filename = zipf.namelist()
    with zipf.open('fra.txt') as f:
        lines = f.read()

    return lines

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s

def prepare_dataset(lines, model_size=128, batch_size=64, test_mode=False):
    lines = lines.decode('utf-8')
    raw_data = []
    for line in lines.split('\n'):
        raw_data.append(line.split('\t'))
    
    raw_data.pop()
    
    raw_data_en, raw_data_fr, _ = list(zip(*raw_data))
    raw_data_en = [normalize_string(data) for data in raw_data_en]
    raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
    raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]
    
    # Tokenization
    en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    en_tokenizer.fit_on_texts(raw_data_en)
    data_en = en_tokenizer.texts_to_sequences(raw_data_en)
    data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,
                                                            padding='post')

    fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    fr_tokenizer.fit_on_texts(raw_data_fr_in)
    fr_tokenizer.fit_on_texts(raw_data_fr_out)
    data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
    data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in,
                                                            padding='post')

    data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
    data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,
                                                                padding='post')

    pes = setup_pes(data_en, data_fr_in)
    if test_mode:
        test_size_shape_mechanisms(en_tokenizer, fr_tokenizer, data_fr_in, model_size=model_size, pes=pes)

    # Create dataset object
    dataset = tf.data.Dataset.from_tensor_slices(
        (data_en, data_fr_in, data_fr_out))
    dataset = dataset.shuffle(len(data_en)).batch(batch_size)
    
    return dataset, pes

def test_size_shape_mechanisms(en_tokenizer, fr_tokenizer, data_fr_in, num_layers=2, h=2, model_size=128, pes=None):
    H = 2
    NUM_LAYERS = 2

    en_vocab_size = len(en_tokenizer.word_index) + 1
    encoder = Encoder(en_vocab_size, model_size, NUM_LAYERS, H, pes)

    en_sequence_in = tf.constant([[1, 2, 3, 4, 6, 7, 8, 0, 0, 0], 
                            [1, 2, 3, 4, 6, 7, 8, 0, 0, 0]])
    encoder_output = encoder(en_sequence_in)

    logger.info('Input vocabulary size {}'.format(en_vocab_size))
    logger.info('Encoder input shape {}'.format(en_sequence_in.shape))
    logger.info('Encoder output shape {}'.format(encoder_output.shape))

    fr_vocab_size = len(fr_tokenizer.word_index) + 1
    max_len_fr = data_fr_in.shape[1]
    decoder = Decoder(fr_vocab_size, model_size, NUM_LAYERS, H, pes)

    fr_sequence_in = tf.constant([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0],
                            [1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0]])
    decoder_output = decoder(fr_sequence_in, encoder_output)

    logger.info('Target vocabulary size {}'.format(fr_vocab_size))
    logger.info('Decoder input shape {}'.format(fr_sequence_in.shape))
    logger.info('Decoder output shape {}'.format(decoder_output.shape))

    
def positional_embedding(pos, model_size=128):
    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    
    return PE


def setup_pes(data_en, data_fr_in, model_size=128):
    max_length = max(len(data_en[0]), len(data_fr_in[0]))

    pes = []
    for i in range(max_length):
        pes.append(positional_embedding(i, model_size))

    pes = np.concatenate(pes, axis=0)
    pes = tf.constant(pes, dtype=tf.float32)
    
    return pes

if __name__ == "__main__":
    lines = download_and_read_file(URL, FILENAME)
    dataset, pes = prepare_dataset(lines)


