import tensorflow as tf
import numpy as np
import os
import requests
from zipfile import ZipFile
import unicodedata
import Decoder, Encoder, MultiHeadAttention

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

def prepare_dataset(lines):
    lines = lines.decode('utf-8')
    raw_data = []
    for line in lines.split('\n'):
        raw_data.append(line.split('\t'))
    
    raw_data = raw_data[:-1]
    
    raw_data_en, raw_data_fr = list(zip(*raw_data))
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

    # Create dataset object
    BATCH_SIZE = 64
    dataset = tf.data.Dataset.from_tensor_slices(
        (data_en, data_fr_in, data_fr_out))
    dataset = dataset.shuffle(len(data_en)).batch(BATCH_SIZE)
    
    H = 2
    NUM_LAYERS = 2

    en_vocab_size = len(en_tokenizer.word_index) + 1
    encoder = Encoder(en_vocab_size, MODEL_SIZE, NUM_LAYERS, H)

    en_sequence_in = tf.constant([[1, 2, 3, 4, 6, 7, 8, 0, 0, 0], 
                            [1, 2, 3, 4, 6, 7, 8, 0, 0, 0]])
    encoder_output = encoder(en_sequence_in)

    print('Input vocabulary size', en_vocab_size)
    print('Encoder input shape', en_sequence_in.shape)
    print('Encoder output shape', encoder_output.shape)

    fr_vocab_size = len(fr_tokenizer.word_index) + 1
    max_len_fr = data_fr_in.shape[1]
    decoder = Decoder(fr_vocab_size, MODEL_SIZE, NUM_LAYERS, H)

    fr_sequence_in = tf.constant([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0],
                            [1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0]])
    decoder_output = decoder(fr_sequence_in, encoder_output)

    print('Target vocabulary size', fr_vocab_size)
    print('Decoder input shape', fr_sequence_in.shape)
    print('Decoder output shape', decoder_output.shape)
    
    return dataset
    
if __name__ == "__main__":
    lines = download_and_read_file(URL, FILENAME)
    dataset = prepare_dataset(lines)
    