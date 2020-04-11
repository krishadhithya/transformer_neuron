
import numpy as np
import tensorflow as tf
from MultiHeadAttention import MultiHeadAttention

class Decoder(tf.keras.Model):
    # Similar to encoder, sort of.
    # Here there are TWO multi-head attention blocks in a single layer:
    #   - one for the target sequences, and one for the encoder's output. 
    # Bottom multi-head attention is masked. 
    def __init__(self, vocab_size, model_size, num_layers, h, pes):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.pes = pes
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.attention_bottom = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bottom_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.attention_middle = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_middle_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        
        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        
        self.dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, sequence, encoder_output):
        # Embedding and positional embedding
        # Pass sequences thru the embedding layer and add up positional encoding info
        embed_out = []
        for i in range(sequence.shape[1]):
            embed = self.embedding(tf.expand_dims(sequence[:, i], axis=1))
            embed_out.append(embed + self.pes[i, :])
        
        embed_out = tf.concat(embed_out, axis=1)
        
        # For loop to create a bunch of layers similarly to the encoder. 
        # in each layer the first block is the multi-head attention in which the target
        # sequence draws attention to itself (self-attention). BUT this block NEEDS TO BE MASKED. 
        # This is so that each token in the target sequence is not trained to depend on its 
        # neighbors to the right. 
        
        bottom_sub_in = embed_out
        
        for i in range(self.num_layers):
            # bottom multi-head sub layer
            bottom_sub_out = []
            
            for j in range(bottom_sub_in.shape[1]):
                # The value vector must not contain tokens that lie to the right of the current token.
                values = bottom_sub_in[:, :j, :]
                attention = self.attention_bottom[i](tf.expand_dims(bottom_sub_in[:, j, :], axis=1), values)
                bottom_sub_out.append(attention)
            
            bottom_sub_out = tf.concat(bottom_sub_out, axis=1)
            bottom_sub_out = bottom_sub_in + bottom_sub_out
            bottom_sub_out = self.attention_bottom_norm[i](bottom_sub_out)
        
            # Next, another multi-head attention layer in which the query is the output of the multi-head attention 
            # above, and the value is the output of the encoder. 
            middle_sub_in = bottom_sub_out
            
            middle_sub_out = []
            for j in range(middle_sub_in.shape[1]):
                attention = self.attention_middle[i](tf.expand_dims(middle_sub_in[:, j, :], axis=1), encoder_output)
                middle_sub_out.append(attention)
            
            middle_sub_out = tf.concat(middle_sub_out, axis=1)
            middle_sub_out = middle_sub_out + middle_sub_in
            middle_sub_out = self.attention_middle_norm[i](middle_sub_out)
            
            # Final piece is the feed forward layer. 
            ffn_in = middle_sub_out
            
            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)            
            
            bottom_sub_in = ffn_out    
        
        logits = self.dense(ffn_out)
        
        return logits