import tensorflow as tf
import numpy as np
import os
import requests
from zipfile import ZipFile
import unicodedata
from Transformer.Decoder import Decoder
from Transformer.Encoder import Encoder
from Transformer.MultiHeadAttention import MultiHeadAttention
from loguru import logger
import re
import time
import tensorflow_datasets as tfds
from Transformer.WarmupThenDecaySchedule import WarmupThenDecaySchedule

tf.compat.v1.enable_eager_execution()
    
URL = 'https://www.manythings.org/anki/fra-eng.zip'
FILENAME = 'data/fra-eng.zip'
NUM_EPOCHS = 15
CROSSENTROPY = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
lr = WarmupThenDecaySchedule(128)
OPTIMIZER = tf.keras.optimizers.Adam(lr,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)
en_tokenizer = None
fr_tokenizer = None
raw_data_en = None
raw_data_fr = None

encoder = None
decoder = None

vocab_size_overhead = 10000

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
    global en_tokenizer
    global fr_tokenizer
    
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

def loss_function(
    targets, 
    logits, 
    crossentropy=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    ):
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)
    
    return loss

def setup_transformer(model_size=128, h=8, num_layers=4, pes=None):
    global encoder
    global decoder
    
    vocab_size = len(en_tokenizer.word_index) + vocab_size_overhead
    encoder = Encoder(vocab_size, model_size, num_layers, h, pes)
    decoder = Decoder(vocab_size, model_size, num_layers, h, pes)
    

def predict(test_source_text=None):
    """ Predict the output sentence for a given input sentence

    Args:
        test_source_text: input sentence (raw string)
    
    Returns:
        The encoder's attention vectors
        The decoder's bottom attention vectors
        The decoder's middle attention vectors
        The input string array (input sentence split by ' ')
        The output string array
    """
    global fr_tokenizer
    global en_tokenizer

    if test_source_text is None:
        test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

    en_output, en_alignments = encoder(tf.constant(test_source_seq), training=False)

    de_input = tf.constant(
        [[fr_tokenizer.word_index['<start>']]], dtype=tf.int64)

    out_words = []

    while True:
        de_output, de_bot_alignments, de_mid_alignments = decoder(de_input, en_output, training=False)
        new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
        out_words.append(fr_tokenizer.index_word[new_word.numpy()[0][0]])

        # Transformer doesn't have sequential mechanism (i.e. states)
        # so we have to add the last predicted word to create a new input sequence
        de_input = tf.concat((de_input, new_word), axis=-1)

        # TODO: get a nicer constraint for the sequence length!
        if out_words[-1] == '<end>' or len(out_words) >= 14:
            break

    print(' '.join(out_words))
    return en_alignments, de_bot_alignments, de_mid_alignments, test_source_text.split(' '), out_words

def train_model(dataset, num_epochs=15):
    starttime = time.time()
    for e in range(NUM_EPOCHS):
        for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
            loss, gradient = train_step(source_seq, target_seq_in,
                            target_seq_out)

            logger.info('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
                e + 1, batch, loss.numpy(), time.time() - starttime))
            starttime = time.time()

        try:
            predict()
        except Exception as e:
            print(e)
            continue


@tf.function
def train_step(source_seq, target_seq_in, target_seq_out):
    """ Execute one training step (forward pass + backward pass)
    
    Args:
        source_seq: source sequences
        target_seq_in: input target sequences (<start> + ...)
        target_seq_out: output target sequences (... + <end>)
    
    Returns:
        The loss value of the current pass
    """
    with tf.GradientTape() as tape:
        encoder_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
        # encoder_mask has shape (batch_size, source_len)
        # we need to add two more dimensions in between
        # to make it broadcastable when computing attention heads
        encoder_mask = tf.expand_dims(encoder_mask, axis=1)
        encoder_mask = tf.expand_dims(encoder_mask, axis=1)
        encoder_output, _ = encoder(source_seq, encoder_mask=encoder_mask)

        decoder_output, _, _ = decoder(
            target_seq_in, encoder_output, encoder_mask=encoder_mask)

        loss = loss_function(target_seq_out, decoder_output)
    
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    OPTIMIZER.apply_gradients(zip(gradients, variables))
    
    return loss, gradients

if __name__ == "__main__":
    
    lines = download_and_read_file(URL, FILENAME)
    dataset, pes = prepare_dataset(lines)
    #import pdb
    #pdb.set_trace()
    setup_transformer(model_size=128, h=8, num_layers=4, pes=pes)
    #assert not isinstance(dataset, tf.data.Dataset)
    logger.info("Transformer setup complete, training model...")
    train_model(dataset)

