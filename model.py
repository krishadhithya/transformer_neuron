import re
import time
import tensorflow as tf
import numpy as np
import unicodedata
from Transformer.Decoder import Decoder
from Transformer.Encoder import Encoder
from Transformer.MultiHeadAttention import MultiHeadAttention
from loguru import logger


class Transformer():
    def __init__(self, hparams, raw_dataset):
        self.hparams = hparams
        self.en_tokenizer = None
        self.fr_tokenizer = None
        self.encoder = None
        self.decoder = None
        self.vocab_size_overhead = 10000
        self.optimizer = tf.keras.optimizers.Adam(hparams.learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)
        
        # Get dataset and positional embeddings
        self.dataset, self.pes = self.prepare_dataset(raw_dataset)
        
        logger.info("Setting up transformer")
        self.setup_transformer(
            model_size=hparams.model_size, 
            h=hparams.attention_heads, 
            num_layers=hparams.num_layers, 
            pes=self.pes)
        
    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s)
        s = re.sub(r'([!.?])', r' \1', s)
        s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
        s = re.sub(r'\s+', r' ', s)
        return s

    def prepare_dataset(self, lines, model_size=128, batch_size=64, test_mode=False):
        
        lines = lines.decode('utf-8')
        raw_data = []
        for line in lines.split('\n'):
            raw_data.append(line.split('\t'))
        
        raw_data.pop()
        
        raw_data_en, raw_data_fr, _ = list(zip(*raw_data))
        raw_data_en = [self.normalize_string(data) for data in raw_data_en]
        raw_data_fr_in = ['<start> ' + self.normalize_string(data) for data in raw_data_fr]
        raw_data_fr_out = [self.normalize_string(data) + ' <end>' for data in raw_data_fr]
        
        # Tokenization
        self.en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.en_tokenizer.fit_on_texts(raw_data_en)
        data_en = self.en_tokenizer.texts_to_sequences(raw_data_en)
        data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,
                                                                padding='post')

        self.fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.fr_tokenizer.fit_on_texts(raw_data_fr_in)
        self.fr_tokenizer.fit_on_texts(raw_data_fr_out)
        data_fr_in = self.fr_tokenizer.texts_to_sequences(raw_data_fr_in)
        data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in,
                                                                padding='post')

        data_fr_out = self.fr_tokenizer.texts_to_sequences(raw_data_fr_out)
        data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,
                                                                    padding='post')

        pes = self.setup_pes(data_en, data_fr_in)
        if test_mode:
            test_size_shape_mechanisms(self.en_tokenizer, self.fr_tokenizer, data_fr_in, model_size=model_size, pes=pes)

        # Create dataset object
        dataset = tf.data.Dataset.from_tensor_slices(
            (data_en, data_fr_in, data_fr_out))
        dataset = dataset.shuffle(len(data_en)).batch(batch_size)
        
        return dataset, pes

    def test_size_shape_mechanisms(self, en_tokenizer, fr_tokenizer, data_fr_in, num_layers=2, h=2, model_size=128, pes=None):
        H = 2
        NUM_LAYERS = 2

        en_vocab_size = len(self.en_tokenizer.word_index) + 1
        encoder = Encoder(en_vocab_size, model_size, NUM_LAYERS, H, pes)

        en_sequence_in = tf.constant([[1, 2, 3, 4, 6, 7, 8, 0, 0, 0], 
                                [1, 2, 3, 4, 6, 7, 8, 0, 0, 0]])
        encoder_output = encoder(en_sequence_in)

        logger.info('Input vocabulary size {}'.format(en_vocab_size))
        logger.info('Encoder input shape {}'.format(en_sequence_in.shape))
        logger.info('Encoder output shape {}'.format(encoder_output.shape))

        fr_vocab_size = len(self.fr_tokenizer.word_index) + 1
        max_len_fr = data_fr_in.shape[1]
        decoder = Decoder(fr_vocab_size, model_size, NUM_LAYERS, H, pes)

        fr_sequence_in = tf.constant([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0],
                                [1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0]])
        decoder_output = decoder(fr_sequence_in, encoder_output)

        logger.info('Target vocabulary size {}'.format(fr_vocab_size))
        logger.info('Decoder input shape {}'.format(fr_sequence_in.shape))
        logger.info('Decoder output shape {}'.format(decoder_output.shape))

        
    def positional_embedding(self, pos, model_size=128):
        PE = np.zeros((1, model_size))
        for i in range(model_size):
            if i % 2 == 0:
                PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
            else:
                PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
        
        return PE


    def setup_pes(self, data_en, data_fr_in, model_size=128):
        max_length = max(len(data_en[0]), len(data_fr_in[0]))

        pes = []
        for i in range(max_length):
            pes.append(self.positional_embedding(i, model_size))

        pes = np.concatenate(pes, axis=0)
        pes = tf.constant(pes, dtype=tf.float32)
        
        return pes

    def loss_function(self,
        targets, 
        logits, 
        crossentropy=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        ):
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss = crossentropy(targets, logits, sample_weight=mask)
        
        return loss

    def setup_transformer(self, model_size=128, h=8, num_layers=4, pes=None):
        
        vocab_size = len(self.en_tokenizer.word_index) + self.vocab_size_overhead
        self.encoder = Encoder(vocab_size, model_size, num_layers, h, pes)
        self.decoder = Decoder(vocab_size, model_size, num_layers, h, pes)
        

    def predict(self, test_source_text=None):
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

        if test_source_text is None:
            test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
        
        logger.info(test_source_text)
        test_source_seq = self.en_tokenizer.texts_to_sequences([test_source_text])
        logger.info(test_source_seq)

        en_output, en_alignments = self.encoder(tf.constant(test_source_seq), training=False)

        de_input = tf.constant(
            [[self.fr_tokenizer.word_index['<start>']]], dtype=tf.int64)

        out_words = []

        while True:
            de_output, de_bot_alignments, de_mid_alignments = self.decoder(de_input, en_output, training=False)
            new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
            out_words.append(self.fr_tokenizer.index_word[new_word.numpy()[0][0]])

            # Transformer doesn't have sequential mechanism (i.e. states)
            # so we have to add the last predicted word to create a new input sequence
            de_input = tf.concat((de_input, new_word), axis=-1)

            # TODO: get a nicer constraint for the sequence length!
            if out_words[-1] == '<end>' or len(out_words) >= 14:
                break

        print(' '.join(out_words))
        return en_alignments, de_bot_alignments, de_mid_alignments, test_source_text.split(' '), out_words

    def train_model(self, num_epochs=15):
        """ Execute num_epoch training steps to train the model, then do a prediction
        
        Args:
            num_epochs: number of epochs to train over
        """
        starttime = time.time()
        for e in range(self.hparams.num_epochs):
            for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(self.dataset.take(-1)):
                loss, gradient = self.train_step(source_seq, target_seq_in,
                                target_seq_out)

                logger.info('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
                    e + 1, batch, loss.numpy(), time.time() - starttime))
                starttime = time.time()

            try:
                self.predict()
            except Exception as e:
                print(e)
                continue
        logger.info("Finished training model...")


    @tf.function
    def train_step(self, source_seq, target_seq_in, target_seq_out):
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
            encoder_output, _ = self.encoder(source_seq, encoder_mask=encoder_mask)

            decoder_output, _, _ = self.decoder(
                target_seq_in, encoder_output, encoder_mask=encoder_mask)

            loss = self.loss_function(target_seq_out, decoder_output)
        
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return loss, gradients