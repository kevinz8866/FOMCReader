import math
import numpy as np
import tensorflow as tf

class AttentionMatrixRankThree(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        """
        STUDENT MUST WRITE:

        Computes attention given key and query matrices.

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys   = K.get_shape()[1]  # window size of keys

        ## Fill triangle below diagonal of matrix with negative infinity and top part with 0.
        ## This helps to avoid over-contribution, since adjacency matrix is symmetric across diagonal. 
        ## Tile this upward to be compatible with addition against computed attention scores.
        mask_vals = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

        # TODO:
        # 1) compute attention weights using queries and key matrices 
        #       - if use_mask==True, then make sure to add the attention mask before softmax
        _ = tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])) / np.sqrt(window_size_keys)
        if self.use_mask:
            _ = tf.math.add(_,atten_mask)
        # 2) return the attention matrix
        return tf.nn.softmax(_,axis=-1)


class AttentionHeadRankThree(tf.keras.layers.Layer):
    
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHeadRankThree, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        # TODO:
        # Initialize the weight matrices for K, V, and Q.
        # They should be able to multiply an input_size vector to produce an output_size vector
        w_init1 = tf.keras.initializers.GlorotNormal()
        w_init2 = tf.keras.initializers.GlorotNormal()
        w_init3 = tf.keras.initializers.GlorotNormal()
        self.k_weight = tf.Variable(initial_value=w_init1(shape=(input_size,output_size)),dtype='float32')
        self.v_weight = tf.Variable(initial_value=w_init2(shape=(input_size,output_size)),dtype='float32')
        self.q_weight = tf.Variable(initial_value=w_init3(shape=(input_size,output_size)),dtype='float32')                   
                             
        self.attn_mtx = AttentionMatrixRankThree(use_mask=self.use_mask)
        
    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        STUDENT MUST WRITE:

        This functions runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        K = tf.einsum('ijk,kl->ijl', inputs_for_keys,self.k_weight)
        V = tf.einsum('ijk,kl->ijl', inputs_for_values,self.v_weight)
        Q = tf.einsum('ijk,kl->ijl', inputs_for_queries,self.q_weight)
        
        scores = self.attn_mtx([K,Q])
        _ = tf.matmul(scores, V)
        return _

    
class TransformerBlockRankThree(tf.keras.layers.Layer):
    
    def __init__(self, emb_sz, useMultiHeadedAttention=False, **kwargs):
        super(TransformerBlockRankThree, self).__init__(**kwargs)

        self.ff_layer = tf.keras.layers.Dense(emb_sz)
        self.self_atten = AttentionHeadRankThree(emb_sz, emb_sz, True)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    @tf.function
    def call(self, inputs, context_sequence):
        """
        This functions calls a transformer block.

        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """
        _=inputs + self.self_atten(inputs,inputs,inputs)
        _=self.layer_norm(_)  
        _=_ + self.ff_layer(_)
        _=self.layer_norm(_)
        _=tf.nn.relu(_)
        return _

class AttentionMatrixRankFour(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        """
        Computes attention given key and query matrices.

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """
        K, Q = inputs
        window_size_queries = Q.get_shape()[2]  # window size of queries
        window_size_keys   = K.get_shape()[2]  # window size of keys

        mask_vals = np.triu(np.ones((K.shape[1], window_size_queries, window_size_keys)) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, K.shape[1], window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1, 1])

        _ = tf.matmul(Q, tf.transpose(K, perm=[0, 1, 3, 2])) / np.sqrt(window_size_keys)
        if self.use_mask:
            _ = tf.math.add(_,atten_mask)

        return tf.nn.softmax(_,axis=-1)


class AttentionHeadRankFour(tf.keras.layers.Layer):
    
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHeadRankFour, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        w_init1 = tf.keras.initializers.GlorotNormal()
        w_init2 = tf.keras.initializers.GlorotNormal()
        w_init3 = tf.keras.initializers.GlorotNormal()
        self.k_weight = tf.Variable(initial_value=w_init1(shape=(input_size,output_size)),dtype='float32')
        self.v_weight = tf.Variable(initial_value=w_init2(shape=(input_size,output_size)),dtype='float32')
        self.q_weight = tf.Variable(initial_value=w_init3(shape=(input_size,output_size)),dtype='float32')                      
                             
        self.attn_mtx = AttentionMatrixRankFour(use_mask=self.use_mask)
        
    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        This functions runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        K = tf.einsum('ixjk,kl->ixjl', inputs_for_keys,self.k_weight)
        V = tf.einsum('ixjk,kl->ixjl', inputs_for_values,self.v_weight)
        Q = tf.einsum('ixjk,kl->ixjl', inputs_for_queries,self.q_weight)
        
        scores = self.attn_mtx([K,Q])
        _ = tf.matmul(scores, V)
        return _


class TransformerBlockRankFour(tf.keras.layers.Layer):
    
    def __init__(self, emb_sz, useMultiHeadedAttention=False, **kwargs):
        super(TransformerBlockRankFour, self).__init__(**kwargs)

        self.ff_layer = tf.keras.layers.Dense(emb_sz)
        self.self_atten = AttentionHeadRankFour(emb_sz, emb_sz, True)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    @tf.function
    def call(self, inputs, context_sequence):
        """
        This functions calls a transformer block.

        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """
        _=inputs + self.self_atten(inputs,inputs,inputs)
        _=self.layer_norm(_)  
        _=_ + self.ff_layer(_)
        _=self.layer_norm(_)
        _=tf.nn.relu(_)
        return _

    
