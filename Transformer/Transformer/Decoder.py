
import numpy as np
import tensorflow as tf
from Transformer.MultiHeadAttention import MultiHeadAttention

class Decoder(tf.keras.Model):
    """
    # Similar to encoder, sort of.
    # Here there are TWO multi-head attention blocks in a single layer:
    #   - one for the target sequences, and one for the encoder's output. 
    # Bottom multi-head attention is masked. 
    Class for the Decoder
    Args:
        model_size: d_model in the paper (depth size of the model)
        num_layers: number of layers (Multi-Head Attention + FNN)
        h: number of attention heads
        embedding: Embedding layer
        embedding_dropout: Dropout layer for Embedding
        attention_bot: array of bottom Multi-Head Attention layers (self attention)
        attention_bot_dropout: array of Dropout layers for bottom Multi-Head Attention
        attention_bot_norm: array of LayerNorm layers for bottom Multi-Head Attention
        attention_mid: array of middle Multi-Head Attention layers
        attention_mid_dropout: array of Dropout layers for middle Multi-Head Attention
        attention_mid_norm: array of LayerNorm layers for middle Multi-Head Attention
        dense_1: array of first Dense layers for FFN
        dense_2: array of second Dense layers for FFN
        ffn_dropout: array of Dropout layers for FFN
        ffn_norm: array of LayerNorm layers for FFN
        dense: Dense layer to compute final output
    """
    def __init__(self, vocab_size, model_size, num_layers, h, pes):
        
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.pes = pes
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.embedding_dropout = tf.keras.layers.Dropout(0.1)
        self.attention_bottom = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bottom_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.attention_bottom_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        
        self.attention_middle = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_middle_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.attention_middle_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        
        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        
        self.ffn_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        
        self.dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, sequence, encoder_output, training=True, encoder_mask=None):
        """ Forward pass for the Decoder
        Args:
            sequence: source input sequences
            encoder_output: output of the Encoder (for computing middle attention)
            training: whether training or not (for Dropout)
            encoder_mask: padding mask for the Encoder's Multi-Head Attention
        
        Returns:
            The output of the Encoder (batch_size, length, model_size)
            The bottom alignment (attention) vectors for all layers
            The middle alignment (attention) vectors for all layers
        
        - Embedding and positional embedding
        - Pass sequences thru the embedding layer and add up positional encoding info
        """
        embed_out = self.embedding(sequence)
        embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        embed_out += self.pes[:sequence.shape[1], :]
        embed_out = self.embedding_dropout(embed_out)

        # For loop to create a bunch of layers similarly to the encoder. 
        # in each layer the first block is the multi-head attention in which the target
        # sequence draws attention to itself (self-attention). BUT this block NEEDS TO BE MASKED. 
        # This is so that each token in the target sequence is not trained to depend on its 
        # neighbors to the right. 
        
        bottom_sub_in = embed_out
        bottom_alignments = []
        middle_alignments = []
        
        for i in range(self.num_layers):
            # bottom multi-head sub layer
            # BOTTOM MULTIHEAD SUB LAYER
            seq_len = bottom_sub_in.shape[1]

            if training:
                mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            else:
                mask = None
            bottom_sub_out, bottom_alignment = self.attention_bottom[i](bottom_sub_in, bottom_sub_in, mask)
            bottom_sub_out = self.attention_bottom_dropout[i](bottom_sub_out, training=training)
            bottom_sub_out = bottom_sub_in + bottom_sub_out
            bottom_sub_out = self.attention_bottom_norm[i](bottom_sub_out)
            
            bottom_alignments.append(bottom_alignment)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bottom_sub_out

            mid_sub_out, mid_alignment = self.attention_middle[i](
                mid_sub_in, encoder_output, encoder_mask)
            mid_sub_out = self.attention_middle_dropout[i](mid_sub_out, training=training)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_middle_norm[i](mid_sub_out)
            
            middle_alignments.append(mid_alignment)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bottom_sub_in = ffn_out

        logits = self.dense(ffn_out)

        return logits, bottom_alignments, middle_alignments