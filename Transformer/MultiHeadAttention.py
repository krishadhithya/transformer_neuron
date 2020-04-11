import numpy as np
import tensorflow as tf

class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.query_size = model_size // h
        self.key_size = model_size // h
        self.value_size = model_size // h
        self.h = h
        self.wq = [tf.keras.layers.Dense(self.query_size) for _ in range(h)]
        self.wk = [tf.keras.layers.Dense(self.key_size) for _ in range(h)]
        self.wv = [tf.keras.layers.Dense(self.value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)
    
    def call(self, query, value):
        # query has shape (batch, query_len, model_size)
        # value has shape (batch, value_len, model_size)
        heads = []
        for i in range(self.h):
            score = tf.matmul(self.wq[i](query), self.wk[i](value), transpose_b=True)
            
            # Here we scale the score as described in the paper (Attention is all you need)
            # score has shape (batch, query_len, value_len)
            score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
            
            # alignment has shape (batch, query_len, value_len)
            alignment = tf.nn.softmax(score, axis=2)
            
            # head has shape (batch, decoder_len, value_size)
            head = tf.matmul(alignment, self.wv[i](value))
            heads.append(head)
        
        # Concatinate all attention heads
        # Such that the last dimension sums up to model_size
        heads = tf.concat(heads, axis=2)
        heads = self.wo(heads)
        
        # heads has shape (batch, query_len, model_size)
        return heads
