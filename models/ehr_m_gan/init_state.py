import tensorflow as tf

def rnn_init_state(init_, batch_size, num_layers, num_units, rnn_network=None, initial_stddev=0.02):

    if init_ == "zero":
        initial_state = rnn_network.zero_state(batch_size, tf.float32)

    elif init_ == "random":
        initial_state = tf.random_normal(shape=(num_layers, 2, batch_size, num_units),
                                         mean=0.0, stddev=1.0)
        initial_state = tf.unstack(initial_state, axis=0)
        initial_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(initial_state[idx][0], initial_state[idx][1]) \
                               for idx in range(num_layers)])

    elif init_ == "variable":
        initial_state = []
        with tf.variable_scope("continuous_discriminator", reuse=tf.AUTO_REUSE):
            for i in range(num_layers):
                sub_initial_state1 = tf.get_variable("layer{}_initial_state1".format(i), (1, num_units),
                                                     initializer=tf.random_normal_initializer(stddev=initial_stddev))
                sub_initial_state1 = tf.tile(sub_initial_state1, (batch_size, 1))
                sub_initial_state2 = tf.get_variable("layer{}_initial_state2".format(i), (1, num_units),
                                                    initializer=tf.random_normal_initializer(stddev=initial_stddev))
                sub_initial_state2 = tf.tile(sub_initial_state2, (batch_size, 1))
                sub_initial_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(sub_initial_state1, sub_initial_state2)
                initial_state.append(sub_initial_state)
            initial_state = tuple(initial_state)

    return initial_state