import numpy as np
import tensorflow as tf
from MultiHeadAttention import MultiHeadAttention

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h, pes):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.pes = pes
        
        # One embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        
        # num_layers Multi-Head attention and normalization layers
        self.attention = [MultiHeadAttention (model_size, h) for _ in range(num_layers)]
        self.attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        
        # num_layers FFN and normalization layers
        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        
        # Should use LayerNorm, technically, but BatchNormalization seems to work fine as is.
        # TODO: Explore using LayerNorm later.
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        
    def call(self, sequence):
        
        # Forward pass of the encoder
        sub_in = []
        for i in range(sequence.shape[1]):
            # compute the embedded vector
            embed = self.embedding(tf.expand_dims(sequence[:, i], axis = 1))
            
            # Add positional encoding to the embedded vector
            sub_in.append(embed + self.pes[i, :])
        
        # Concatenate the result so that the shape is (batch_size, length, model_size)
        sub_in = tf.concat(sub_in, axis=1)
        
        # Layers of the multi-head attention 
        # We will have num_layers of (Attention + FFN)
        for i in range(self.num_layers):
            sub_out = []
            
            # Iterate along the sequence length
            for j in range(sub_in.shape[1]):
                # Compute the context vector towards the whole sequence
                attention = self.attention[i](tf.expand_dims(sub_in[:, j, :], axis=1), sub_in)
                sub_out.append(attention)
            
            # Concatenate the result to get shape (batch_size, length, model_size)
            sub_out = tf.concat(sub_out, axis=1)
            
            # Residual connection followed by normalization layer
            sub_out = sub_in + sub_out
            # Normalize the output
            sub_out = self.attention_norm[i](sub_out)
            
            # Layers of the feed-forward network
            # The feed forward network input is the output of the multi-headed attention
            ffn_in = sub_out
            
            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            # Add the residual connection
            ffn_out = ffn_in + ffn_out
            # Normalize the output
            ffn_out = self.ffn_norm[i](ffn_out)
            
            # Assign the feed forward output to the next layer's multi-head attention input
            sub_in = ffn_out
            
        # Return the result when done
        return ffn_out