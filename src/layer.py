import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from tensorflow.keras.initializers import RandomNormal


class LightMultiHeadAttention(tf.Module):

    def __init__(self,
                 n_heads: int,
                 k_interests: int,
                 d_model: int,
                 seq_len: int,
                 hidden_dropout_prob: float,
                 attn_dropout_prob: float,
                 layer_norm_eps: float,
                 initializer_range: float):
        super(LightMultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.k_interest = k_interests

        self.attention_head_size = int(d_model / n_heads)
        self.all_head_size = self.n_heads * self.attention_head_size

        self.wq = Dense(self.all_head_size,
                        input_shape=(d_model,),
                        kernel_initializer=RandomNormal(mean=0., stddev=initializer_range))
        self.wk = Dense(self.all_head_size,
                        input_shape=(d_model,),
                        kernel_initializer=RandomNormal(mean=0., stddev=initializer_range))
        self.wv = Dense(self.all_head_size,
                        input_shape=(d_model,),
                        kernel_initializer=RandomNormal(mean=0., stddev=initializer_range))

        self.attpooling_key = ItemToInterestAggregation(d_model, k_interests)
        self.attpooling_value = ItemToInterestAggregation(d_model, k_interests)

        self.attn_scale_factor = 2
        self.pos_q_linear = Dense(self.all_head_size,
                                  input_shape=(d_model,),
                                  kernel_initializer=RandomNormal(mean=0., stddev=initializer_range))
        self.pos_k_linear = Dense(self.all_head_size,
                                  input_shape=(d_model,),
                                  kernel_initializer=RandomNormal(mean=0., stddev=initializer_range))
        self.pos_scaling = float(self.attention_head_size * self.attn_scale_factor) ** -0.5
        self.pos_ln = LayerNormalization(epsilon=layer_norm_eps)

        self.attn_dropout = Dropout(attn_dropout_prob)

        self.dense = Dense(d_model, kernel_initializer=RandomNormal(mean=0., stddev=initializer_range))
        self.layernorm = LayerNormalization(epsilon=layer_norm_eps)
        self.out_dropout = Dropout(hidden_dropout_prob)

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Split the last dimension into (num_heads, depth)

        Args:
            x (tf.Tensor): [description]

        Returns:
            result (tf.Tensor): [description]
        """
        x = tf.reshape(x, [-1, x.shape[1], self.n_heads, self.attention_head_size])
        result = tf.transpose(x, perm=[0, 2, 1, 3])

        return result

    def __call__(self, x: tf.Tensor, pos_emb: tf.Tensor, is_training: bool) -> tf.Tensor:
        """Multi-head Self-attention layers, a attention score dropout layer is introduced

        Args:
            x (tf.Tensor): [description]
            pos_emb (tf.Tensor): [description]

        Returns:
            hidden_states (tf.Tensor): [description]
        """
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # low-rank decomposed self-attention: relation of items
        query_layer = self._split_heads(q)
        key_layer = self._split_heads(self.attpooling_key(k))
        value_layer = self._split_heads(self.attpooling_value(v))

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.attention_head_size, tf.float32))

        # normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-2)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer_item = tf.matmul(attention_probs, value_layer)

        # decoupled position encoding: relation of positions
        value_layer_pos = self._split_heads(v)
        pos_emb = tf.expand_dims(self.pos_ln(pos_emb), 0)
        pos_query_layer = self._split_heads(self.pos_q_linear(pos_emb)) * self.pos_scaling
        pos_key_layer = self._split_heads(self.pos_k_linear(pos_emb))

        abs_pos_bias = tf.matmul(pos_query_layer, pos_key_layer, transpose_b=True)
        abs_pos_bias = abs_pos_bias / tf.math.sqrt(tf.cast(self.attention_head_size, tf.float32))
        abs_pos_bias = tf.nn.softmax(abs_pos_bias, axis=-2)

        context_layer_pos = tf.matmul(abs_pos_bias, value_layer_pos)

        context_layer = context_layer_item + context_layer_pos
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, [-1, context_layer.shape[1], self.all_head_size])
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states, is_training)
        hidden_states = self.layernorm(hidden_states + x)

        return hidden_states
    

class ItemToInterestAggregation(tf.Module):

    def __init__(self, hidden_size: int, k_interests: int = 5):
        super(ItemToInterestAggregation, self).__init__()

        self.k_interests = k_interests
        self.theta = tf.Variable(tf.random.normal([hidden_size, k_interests]))

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """the user history is projected into k latent interests,
        and each of the user’s historical items merely needs to interact with the k latent interests to establish its context-awareness

        Args:
            x (tf.Tensor): input tensor [B, L, D]

        Returns:
            result (tf.Tensor): item-to-item interaction [B, k, d]
        """
        D_matrix = tf.matmul(x, self.theta)
        D_matrix = tf.nn.softmax(D_matrix, axis=-2)
        result = tf.einsum('nij,nik->nkj', x, D_matrix)

        return result
    
    
class FeedForward(tf.Module):

    def __init__(self,
                 hidden_size: int,
                 inner_size: int,
                 hidden_dropout_prob: float,
                 hidden_act: str,
                 layer_norm_eps: float):
        super(FeedForward, self).__init__()

        self.dense_1 = Dense(inner_size,
                             input_shape=(hidden_size,),
                             activation=hidden_act)
        self.dense_2 = Dense(hidden_size,
                             input_shape=(inner_size,))
        self.layernorm = LayerNormalization(epsilon=layer_norm_eps)
        self.dropout = Dropout(hidden_dropout_prob)

    def __call__(self, x: tf.Tensor, is_training: bool) -> tf.Tensor:
        """Point-wise feed-forward layer is implemented by two dense layers

        Args:
            x (tf.Tensor): the input of the point-wise feed-forward layer

        Returns:
            hidden_states (tf.Tensor): the output of the point-wise feed-forward layer
        """
        hidden_states = self.dense_1(x)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states, is_training)
        hidden_states = self.layernorm(hidden_states + x)

        return hidden_states
