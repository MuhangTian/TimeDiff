import tensorflow as tf

class Bilateral_LSTM_cell():
    def __init__(self, input_dim, hidden_dim, scope_name):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scope_name = scope_name

    def __call__(self, x, hidden_memory_tm1, hidden_memory_tm2):
        ## unstack hidden vectors and context vectors
        previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
        previous_hidden_state_, _ = tf.unstack(hidden_memory_tm2)

        # Input Gate (Wi, Ui, Vi)
        with tf.variable_scope(self.scope_name + "Input_gate", reuse=tf.AUTO_REUSE):
            Wi = tf.get_variable(name='Wi', shape=[self.input_dim,  self.hidden_dim], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
            Ui = tf.get_variable(name='Ui', shape=[self.hidden_dim, self.hidden_dim], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
            Vi = tf.get_variable(name='Vi', shape=[self.hidden_dim, self.hidden_dim], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))

            i = tf.sigmoid(
                tf.matmul(x, Wi) +
                tf.matmul(previous_hidden_state, Ui) +
                tf.matmul(previous_hidden_state_, Vi)
            )

        # Forget gate (Wf, Uf, Vf)
        with tf.variable_scope(self.scope_name + "Forget_gate", reuse=tf.AUTO_REUSE):
            Wf = tf.get_variable(name='Wf', shape=[self.input_dim,  self.hidden_dim], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
            Uf = tf.get_variable(name='Uf', shape=[self.hidden_dim, self.hidden_dim], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
            Vf = tf.get_variable(name='Vf', shape=[self.hidden_dim, self.hidden_dim], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))

            f = tf.sigmoid(
                tf.matmul(x, Wf) +
                tf.matmul(previous_hidden_state, Uf) +
                tf.matmul(previous_hidden_state_, Vf)
            )

        # Output gate (Wo, Uo, Vo)
        with tf.variable_scope(self.scope_name + "Output_gate", reuse=tf.AUTO_REUSE):
            Wo = tf.get_variable(name='Wo', shape=[self.input_dim,  self.hidden_dim], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
            Uo = tf.get_variable(name='Uo', shape=[self.hidden_dim, self.hidden_dim], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
            Vo = tf.get_variable(name='Vo', shape=[self.hidden_dim, self.hidden_dim], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))

            o = tf.sigmoid(
                tf.matmul(x, Wo) +
                tf.matmul(previous_hidden_state, Uo) +
                tf.matmul(previous_hidden_state_, Vo)
            )

        # Updated part for new cell state (Wc, Uc, Vc)
        with tf.variable_scope(self.scope_name + "Cell_gate", reuse=tf.AUTO_REUSE):
            Wc = tf.get_variable(name='Wc', shape=[self.input_dim,  self.hidden_dim], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
            Uc = tf.get_variable(name='Uc', shape=[self.hidden_dim, self.hidden_dim], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
            Vc = tf.get_variable(name='Vc', shape=[self.hidden_dim, self.hidden_dim], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))

            c_ = tf.nn.tanh(
                tf.matmul(x, Wc) +
                tf.matmul(previous_hidden_state, Uc) +
                tf.matmul(previous_hidden_state_, Vc)
            )

        # Final Memory cell
        c = f * c_prev + i * c_

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(c)

        return current_hidden_state, tf.stack([current_hidden_state, c])


class MultilayerCells():
    def __init__(self, cells):
        self.cells = cells

    def __call__(self, input, state, state_):
        cur_inp = input
        new_states = []
        for i in range(len(self.cells)):
            with tf.variable_scope("cell_%d" % i):
                cell = self.cells[i]
                cur_inp, new_state = cell(x=cur_inp, hidden_memory_tm1=state[i], hidden_memory_tm2=state_[i])
            new_states.append(new_state)

        return cur_inp, new_states