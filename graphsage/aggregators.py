import tensorflow as tf

from .layers import Layer, Dense
from .inits import glorot, zeros

class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, 
            name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
       
        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
         
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        means = tf.reduce_mean(tf.concat([neigh_vecs, 
            tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)
       
        # [nodes] x [out_dim]
        output = tf.matmul(means, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class SeqAggregator(Layer):
    """ Aggregates via a standard LSTM."""
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None,  concat=False, **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        dims = tf.shape(neigh_vecs)
        batch_size = dims[0]
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.maximum(length, tf.constant(1.))
        length = tf.cast(length, tf.int32)

        with tf.variable_scope(self.name) as scope:
            try:
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
            except ValueError:
                scope.reuse_variables()
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
        batch_size = tf.shape(rnn_outputs)[0]
        max_len = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        neigh_h = tf.gather(flat, index)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
         
        output = tf.add_n([from_self, from_neighs])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)
    
###########################################
class DASGNNAggregator(Layer):
    """
    DAS-GNN Aggregator that dynamically adjusts aggregation structure using attention mechanisms.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, num_sampled_neighbors=0, **kwargs):
        super(DASGNNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.num_sampled_neighbors = num_sampled_neighbors

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        self.neigh_input_dim = neigh_input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['self_weights'] = tf.get_variable('self_weights', shape=[input_dim, output_dim],
                                                        initializer=tf.glorot_uniform_initializer())
            self.vars['neigh_weights'] = tf.get_variable('neigh_weights', shape=[neigh_input_dim, output_dim],
                                                         initializer=tf.glorot_uniform_initializer())
            self.vars['attention_weights'] = tf.get_variable('attention_weights', shape=[output_dim, 1],
                                                             initializer=tf.glorot_uniform_initializer())
            if self.bias:
                self.vars['bias'] = tf.get_variable('bias', shape=[output_dim], initializer=tf.zeros_initializer())

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)

        # Transform self and neighbor vectors
        self_transformed = tf.matmul(self_vecs, self.vars['self_weights'])
        neigh_transformed = tf.matmul(tf.reshape(neigh_vecs, [-1, self.neigh_input_dim]), self.vars['neigh_weights'])
        neigh_transformed = tf.reshape(neigh_transformed, [-1, tf.shape(neigh_vecs)[1], self.output_dim])

        # Compute attention scores
        expanded_self_transformed = tf.expand_dims(self_transformed, axis=1)
        concat_self_neigh = tf.concat([expanded_self_transformed, neigh_transformed], axis=1)

        attention_logits = tf.nn.relu(tf.matmul(tf.reshape(concat_self_neigh, [-1, self.output_dim]), self.vars['attention_weights']))
        attention_logits = tf.reshape(attention_logits, [-1, tf.shape(concat_self_neigh)[1], 1])
        attention_scores = tf.nn.softmax(attention_logits, dim=1)

        # Sample neighbors based on attention scores
        top_k_values, top_k_indices = tf.nn.top_k(attention_scores[:, 1:, 0], k=self.num_sampled_neighbors)
        batch_indices = tf.reshape(tf.range(tf.shape(attention_scores)[0]), [-1, 1, 1])
        batch_indices = tf.tile(batch_indices, [1, self.num_sampled_neighbors, 1])
        gather_indices = tf.concat([batch_indices, tf.expand_dims(top_k_indices, axis=-1)], axis=-1)
        sampled_neighs = tf.gather_nd(neigh_transformed, gather_indices)
        sampled_attention_scores = tf.gather_nd(attention_scores[:, 1:, :], gather_indices)

        # Aggregate sampled neighbor vectors with attention scores
        weighted_neighs = sampled_neighs * sampled_attention_scores
        neigh_aggregated = tf.reduce_sum(weighted_neighs, axis=1)

        if not self.concat:
            output = tf.add(self_transformed, neigh_aggregated)
        else:
            output = tf.concat([self_transformed, neigh_aggregated], axis=1)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)


