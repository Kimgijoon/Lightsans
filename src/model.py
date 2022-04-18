from typing import Optional, Dict

import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dropout, Embedding
from tensorflow.keras.initializers import RandomNormal

from src.layer import LightTransformerEncoder


class LightSANs(tf.keras.Model):

    def __init__(self, config: dict, name: Optional[str]=None, is_training=True):
        super().__init__(name=name)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.k_interests = config['k_interests']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']

        self.seq_len = config['max_seq_length']
        self.n_items = config['n_items']

        self.trm_encoder = LightTransformerEncoder(n_layers=self.n_layers,
                                                   n_heads=self.n_heads,
                                                   k_interests=self.k_interests,
                                                   hidden_size=self.hidden_size,
                                                   seq_len=self.seq_len,
                                                   inner_size=self.inner_size,
                                                   hidden_dropout_prob=self.hidden_dropout_prob,
                                                   attn_dropout_prob=self.attn_dropout_prob,
                                                   hidden_act=self.hidden_act,
                                                   layer_norm_eps=self.layer_norm_eps,
                                                   initializer_range=self.initializer_range)
        self.item_embedding = Embedding(self.n_items,
                                   self.hidden_size,
                                   embeddings_initializer=RandomNormal(mean=0., stddev=self.initializer_range),
                                   name='item_embedding',
                                   mask_zero=True)
        self.position_embedding = Embedding(self.seq_len,
                                       self.hidden_size,
                                       embeddings_initializer=RandomNormal(mean=0., stddev=self.initializer_range),
                                       name='position_embedding')
        self.layernorm = LayerNormalization(epsilon=self.layer_norm_eps)
        self.dropout = Dropout(self.hidden_dropout_prob)
        self.is_training = is_training

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Define LightSANs model

        Args:
            features (Dict): feature dict

        Returns:
            output (tf.Tensor): output of LightSANS
        """
        item_seq = tf.squeeze(features['sequence'], axis=1)
        item_seq_len = tf.squeeze(features['sequence_length'], axis=1)
        position_ids = tf.range(item_seq.shape[1], dtype=tf.int64)
        position_emb = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        item_emb = self.layernorm(item_emb)
        item_emb = self.dropout(item_emb, self.is_training)

        trm_output = self.trm_encoder(item_emb, position_emb, self.is_training)
        output = tf.gather(trm_output, item_seq_len -1, axis=1, batch_dims=1)
        logit = tf.matmul(output, self.item_embedding.embeddings, transpose_b=True)

        return logit

