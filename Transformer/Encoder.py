import numpy as np
import tensorflow as tf
import bert.MultiHeadAttention as MultiHeadAttention

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        
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
        