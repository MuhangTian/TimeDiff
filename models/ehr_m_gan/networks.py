import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from tensorflow.contrib.layers import l2_regularizer
from models.ehr_m_gan.init_state import rnn_init_state
# from Bilateral_lstm_cell import recurrent_unit_bilateral
from models.ehr_m_gan.Bilateral_lstm_class import Bilateral_LSTM_cell, MultilayerCells

class C_VAE_NET(object):
    def __init__(self,
                 batch_size, time_steps, dim, z_dim,
                 enc_size, dec_size,
                 enc_layers, dec_layers,
                 keep_prob, l2scale,
                 conditional=False, num_labels=0):

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.dim = dim
        self.z_dim = z_dim
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.keep_prob = keep_prob
        self.l2scale = l2scale
        self.conditional = conditional
        self.num_labels = num_labels
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers

    def build_vae(self, input_data, conditions=None):
        if self.conditional:
            # cVAE
            assert not self.num_labels == 0
            repeated_encoding = tf.stack([conditions]*self.time_steps, axis=1)
            input_data_cond = tf.concat([input_data, repeated_encoding], axis=-1)
            input_enc = tf.unstack(input_data_cond, axis=1)
        else:
            input_enc = tf.unstack(input_data, axis=1)

        # multicell RNN -----------------------------------------------------
        self.cell_enc = self.buildEncoder()
        self.cell_dec = self.buildDecoder()
        enc_state = self.cell_enc.zero_state(self.batch_size, tf.float32)
        dec_state = self.cell_dec.zero_state(self.batch_size, tf.float32)

        self.e = tf.random_normal((self.batch_size, self.z_dim))
        self.c, mu, logsigma, sigma, z = [0] * self.time_steps, [0] * self.time_steps, [0] * self.time_steps, \
                                         [0] * self.time_steps, [0] * self.time_steps  
        w_mu, b_mu, w_sigma, b_sigma, self.w_h_dec, self.b_h_dec = self.buildSampling()

        for t in range(self.time_steps):
            if t == 0:
                c_prev = tf.zeros((self.batch_size, self.dim))
            else:
                c_prev = self.c[t - 1]

            c_sigmoid = tf.sigmoid(c_prev)

            if self.conditional:
                x_hat = tf.unstack(input_data, axis=1)[t] - c_sigmoid
            else:
                x_hat = input_enc[t] - c_sigmoid

            with tf.variable_scope('Encoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.AUTO_REUSE):
                h_enc, enc_state = self.cell_enc(tf.concat([input_enc[t], x_hat], 1), enc_state)

            # sampling layer
            mu[t] = tf.matmul(h_enc, w_mu) + b_mu  
            logsigma[t] = tf.matmul(h_enc, w_sigma) + b_sigma 
            sigma[t] = tf.exp(logsigma[t])

            #cVAE
            if self.conditional:
                z[t] = mu[t] + sigma[t] * self.e
                # conditional information
                z[t] = tf.concat([z[t], conditions], axis=-1)
            else:
                z[t] = mu[t] + sigma[t] * self.e

            with tf.variable_scope('Decoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.AUTO_REUSE):
                h_dec, dec_state = self.cell_dec(z[t], dec_state)

            self.c[t] = tf.nn.sigmoid( tf.matmul(h_dec, self.w_h_dec) + self.b_h_dec )

        self.decoded = tf.stack(self.c, axis=1)

        return self.decoded, sigma, mu, logsigma, z

    def reconstruct_decoder(self, dec_input, conditions=None):
        rec_decoded = [0] * self.time_steps
        rec_dec_state = self.cell_dec.zero_state(self.batch_size, dtype=tf.float32)
        for t in range(self.time_steps):
            if self.conditional:
                dec_input_with_c = tf.concat([dec_input[t], conditions], axis=-1)
                rec_h_dec, rec_dec_state = self.cell_dec(dec_input_with_c, rec_dec_state)
            else:
                rec_h_dec, rec_dec_state = self.cell_dec(dec_input[t], rec_dec_state)

            rec_decoded[t] = tf.nn.sigmoid( tf.matmul(rec_h_dec, self.w_h_dec) + self.b_h_dec )

        return tf.stack(rec_decoded, axis=1)

    def buildEncoder(self):
        cell_units = []
        for num_units in range(self.enc_layers-1):
            cell = tf.nn.rnn_cell.LSTMCell(self.enc_size, name="Continuous_VAE", reuse=tf.compat.v1.AUTO_REUSE)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            cell_units.append(cell)

        # weight-sharing in the last layer of encoder
        cell = tf.nn.rnn_cell.LSTMCell(self.enc_size, name="Shared_VAE", reuse=tf.compat.v1.AUTO_REUSE)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        cell_units.append(cell)

        cell_enc = tf.nn.rnn_cell.MultiRNNCell(cell_units)
        return cell_enc

    def buildDecoder(self):
        cell_units = []

        # weight-sharing in the first layer of decoder
        cell = tf.nn.rnn_cell.LSTMCell(self.dec_size, name="Shared_VAE", reuse=tf.AUTO_REUSE)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        cell_units.append(cell)

        for num_units in range(self.dec_layers-1):
            cell = tf.nn.rnn_cell.LSTMCell(self.dec_size, name="Continuous_VAE", reuse=tf.AUTO_REUSE)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            cell_units.append(cell)

        cell_dec = tf.nn.rnn_cell.MultiRNNCell(cell_units)
        return cell_dec

    def buildSampling(self):
        w_mu = self.weight_variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='w_mu') 
        b_mu = self.bias_variable([self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='b_mu')
        w_sigma = self.weight_variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='w_sigma')
        b_sigma = self.bias_variable([self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='b_sigma')
        w_h_dec = self.weight_variable([self.dec_size, self.dim], scope_name='Decoder/Linear/Continuous_VAE', name='w_h_dec')
        b_h_dec = self.bias_variable([self.dim], scope_name='Decoder/Linear/Continuous_VAE', name='b_h_dec')

        return w_mu, b_mu, w_sigma, b_sigma, w_h_dec, b_h_dec

    def weight_variable(self, shape, scope_name, name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # initial = tf.truncated_normal(shape, stddev=0.1)
            wv = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        return wv

    def bias_variable(self, shape, scope_name, name=None):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # initial = tf.constant(0.1, shape=shape)
            bv = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        return bv

class D_VAE_NET(object):
    def __init__(self,
                 batch_size, time_steps,
                 dim, z_dim,
                 enc_size, dec_size,
                 enc_layers, dec_layers,
                 keep_prob, l2scale,
                 conditional=False, num_labels=0):

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.dim = dim
        self.z_dim = z_dim
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.keep_prob = keep_prob
        self.l2scale = l2scale
        self.conditional = conditional
        self.num_labels = num_labels

    def build_vae(self, input_data, conditions=None):
        if self.conditional:
            # cVAE
            assert not self.num_labels == 0
            repeated_encoding = tf.stack([conditions] * self.time_steps, axis=1)
            input_data_cond = tf.concat([input_data, repeated_encoding], axis=-1)
            input_enc = tf.unstack(input_data_cond, axis=1)
        else:
            input_enc = tf.unstack(input_data, axis=1)

        # # multicell RNN -----------------------------------------------------
        self.cell_enc = self.buildEncoder()
        self.cell_dec = self.buildDecoder()
        enc_state = self.cell_enc.zero_state(self.batch_size, tf.float32)
        dec_state = self.cell_dec.zero_state(self.batch_size, tf.float32)

        self.e = tf.random_normal((self.batch_size, self.z_dim)) 
        self.c, mu, logsigma, sigma, z = [None] * self.time_steps, [None] * self.time_steps, \
                                         [None] * self.time_steps, [None] * self.time_steps, \
                                         [None] * self.time_steps

        w_mu, b_mu, w_sigma, b_sigma, self.w_h_dec, self.b_h_dec = self.buildSampling()

        for t in range(self.time_steps):
            if t == 0:
                c_prev = tf.zeros((self.batch_size, self.dim))
            else:
                c_prev = self.c[t - 1]

            c_sigmoid = tf.sigmoid(c_prev)

            if self.conditional:
                x_hat = tf.unstack(input_data, axis=1)[t] - c_sigmoid
            else:
                x_hat = input_enc[t] - c_sigmoid

            with tf.variable_scope('Encoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.AUTO_REUSE):
                h_enc, enc_state = self.cell_enc(tf.concat([input_enc[t], x_hat], 1), enc_state)

            # sampling layer
            mu[t] = tf.matmul(h_enc, w_mu) + b_mu  # [z_size]
            logsigma[t] = tf.matmul(h_enc, w_sigma) + b_sigma  # [z_size]
            sigma[t] = tf.exp(logsigma[t])

            # cVAE
            if self.conditional:
                z[t] = mu[t] + sigma[t] * self.e
                # conditional information
                z[t] = tf.concat([z[t], conditions], axis=-1)
            else:
                z[t] = mu[t] + sigma[t] * self.e

            with tf.variable_scope('Decoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.AUTO_REUSE):
                h_dec, dec_state = self.cell_dec(z[t], dec_state)
                
            self.c[t] =  tf.nn.sigmoid( tf.matmul(h_dec, self.w_h_dec) + self.b_h_dec )

        self.decoded = tf.stack(self.c, axis=1)

        return self.decoded, sigma, mu, logsigma, z

    def reconstruct_decoder(self, dec_input, conditions=None):
        rec_decoded = [0] * self.time_steps
        rec_dec_state = self.cell_dec.zero_state(self.batch_size, dtype=tf.float32)
        for t in range(self.time_steps):
            if self.conditional:
                dec_input_with_c = tf.concat([dec_input[t], conditions], axis=-1)
                rec_h_dec, rec_dec_state = self.cell_dec(dec_input_with_c, rec_dec_state)
            else:
                rec_h_dec, rec_dec_state = self.cell_dec(dec_input[t], rec_dec_state)

            rec_decoded[t] =  tf.nn.sigmoid( tf.matmul(rec_h_dec, self.w_h_dec) + self.b_h_dec )

        return tf.stack(rec_decoded, axis=1)

    def buildEncoder(self):
        cell_units = []

        for num_units in range(self.enc_layers-1):
            cell = tf.nn.rnn_cell.LSTMCell(self.enc_size, name="Discrete_VAE", reuse=tf.AUTO_REUSE)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            cell_units.append(cell)

        # weight sharing in the last layer of encoder
        cell = tf.nn.rnn_cell.LSTMCell(self.enc_size, name="Shared_VAE", reuse=tf.AUTO_REUSE)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        cell_units.append(cell)

        cell_enc = tf.nn.rnn_cell.MultiRNNCell(cell_units)
        return cell_enc

    def buildDecoder(self):
        cell_units = []

        # weight sharing in the first layer of decoder
        cell = tf.nn.rnn_cell.LSTMCell(self.dec_size, name="Shared_VAE", reuse=tf.AUTO_REUSE)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        cell_units.append(cell)

        for num_units in range(self.dec_layers-1):
            cell = tf.nn.rnn_cell.LSTMCell(self.dec_size, name="Discrete_VAE", reuse=tf.AUTO_REUSE)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            cell_units.append(cell)

        cell_dec = tf.nn.rnn_cell.MultiRNNCell(cell_units)

        return cell_dec

    def buildSampling(self):
        # sampling layer
        w_mu = self.weight_variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='w_mu')
        b_mu = self.bias_variable([self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='b_mu')
        w_sigma = self.weight_variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='w_sigma')
        b_sigma = self.bias_variable([self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='b_sigma')
        # linear layer of decoder
        w_h_dec = self.weight_variable([self.dec_size, self.dim], scope_name='Decoder/Linear/Discrete_VAE', name='w_h_dec')
        b_h_dec = self.bias_variable([self.dim], scope_name='Decoder/Linear/Discrete_VAE', name='b_h_dec')

        return w_mu, b_mu, w_sigma, b_sigma, w_h_dec, b_h_dec

    def weight_variable(self, shape, scope_name, name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # initial = tf.truncated_normal(shape, stddev=0.1)
            wv = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        return wv

    def bias_variable(self, shape, scope_name, name=None):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # initial = tf.constant(0.1, shape=shape)
            bv = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        return bv

class C_GAN_NET(object):
    def __init__(self, batch_size, noise_dim, dim, gen_dim, time_steps,
                 gen_num_units, gen_num_layers,
                 dis_num_units, dis_num_layers,
                 keep_prob, l2_scale,
                 conditional=False, num_labels=0):

        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.dim = dim
        self.gen_dim = gen_dim
        self.time_steps = time_steps
        self.gen_num_units = gen_num_units
        self.gen_num_layers = gen_num_layers
        self.dis_num_units = dis_num_units
        self.dis_num_layers = dis_num_layers
        self.keep_prob = keep_prob
        self.l2_scale = l2_scale
        self.conditional = conditional
        self.num_labels = num_labels

    def build_GenRNN(self, input_noise, conditions=None):
        if self.conditional:
            # create input noise for generator
            repeated_encoding = tf.stack([conditions]*self.time_steps, axis=1)
            noise_with_c = tf.concat([input_noise, repeated_encoding], axis=2)
            self.g_input = tf.unstack(noise_with_c, axis=1)

            # create multi-cell lstm layers
            cells_list = []
            for i in range(self.gen_num_layers):
                g_input_dim = (self.noise_dim + self.num_labels) if i == 0 else self.gen_num_units
                bilstmcell = Bilateral_LSTM_cell(input_dim=g_input_dim, hidden_dim=self.gen_num_units, scope_name="Continuous_generator/RNNCell_%d" % i)
                cells_list.append(bilstmcell)
            self.g_rnn_network = MultilayerCells(cells=cells_list)

        else:
            # create input noise for generator
            self.g_input = tf.unstack(input_noise, axis=1)

            # create multi-cell lstm layers
            cells_list = []
            for i in range(self.gen_num_layers):
                g_input_dim = self.noise_dim if i == 0 else self.gen_num_units
                bilstmcell = Bilateral_LSTM_cell(input_dim=g_input_dim, hidden_dim=self.gen_num_units, scope_name="Continuous_generator/RNNCell_%d" % i)
                cells_list.append(bilstmcell)
            self.g_rnn_network = MultilayerCells(cells=cells_list)

        # create initial states for multi-cell lstm
        initial_state = []
        for i in range(self.gen_num_layers):
            state_ = tf.stack([tf.zeros([self.batch_size, self.gen_num_units]),
                               tf.zeros([self.batch_size, self.gen_num_units])])
            initial_state.append(state_)

        return initial_state

    def gen_Onestep(self, t, state):
        with tf.variable_scope("Continuous_generator", regularizer=l2_regularizer(self.l2_scale)):
            cell_new_output_, new_state = self.g_rnn_network(input=self.g_input[t], state=state[0], state_=state[1])
            new_output_linear = tf.layers.dense(inputs=cell_new_output_, units=self.gen_dim, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE)
        return new_output_linear, new_state

    def build_Discriminator(self, input_discriminator):
        with tf.variable_scope("Continuous_discriminator", regularizer=l2_regularizer(self.l2_scale)):
            cell_units = []
            for num_units in range(self.dis_num_layers):
                cell = tf.nn.rnn_cell.LSTMCell(self.dis_num_units, reuse=tf.AUTO_REUSE)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                cell_units.append(cell)
            d_rnn_network = tf.nn.rnn_cell.MultiRNNCell(cell_units)
            # initial_state = d_rnn_network.zero_state(self.batch_size, dtype=tf.float32)
            initial_state = rnn_init_state(init_="variable", batch_size=self.batch_size,
                                           num_layers=self.dis_num_layers, num_units=self.dis_num_units)
            outputs, _ = tf.nn.static_rnn(cell=d_rnn_network, inputs=input_discriminator, initial_state=initial_state)
            outputs = tf.stack(outputs, axis=1) 

            result = tf.layers.dense(tf.layers.flatten(outputs), 1, reuse=tf.AUTO_REUSE, activation=None)
        return result, outputs

class D_GAN_NET(object):
    def __init__(self, batch_size, noise_dim, gen_dim, dim, time_steps,
                 gen_num_units, gen_num_layers,
                 dis_num_units, dis_num_layers, keep_prob, l2_scale,
                 conditional=False, num_labels=0):
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.gen_dim = gen_dim
        self.dim = dim
        self.time_steps = time_steps
        self.gen_num_units = gen_num_units
        self.gen_num_layers = gen_num_layers
        self.dis_num_units = dis_num_units
        self.dis_num_layers = dis_num_layers
        self.keep_prob = keep_prob
        self.l2_scale = l2_scale
        self.conditional = conditional
        self.num_labels = num_labels

    def build_GenRNN(self, input_noise, conditions=None):
        if self.conditional:
            # create input noise for generator
            repeated_encoding = tf.stack([conditions]*self.time_steps, axis=1)
            noise_with_c = tf.concat([input_noise, repeated_encoding], axis=2)
            self.g_input = tf.unstack(noise_with_c, axis=1)

            # create multi-cell lstm for generator
            cells_list = []
            for i in range(self.gen_num_layers):
                g_input_dim = (self.noise_dim + self.num_labels) if i == 0 else self.gen_num_units
                bilstmcell = Bilateral_LSTM_cell(input_dim=g_input_dim, hidden_dim=self.gen_num_units, scope_name="Discrete_generator/RNNCell_%d" % i)
                cells_list.append(bilstmcell)
            self.g_rnn_network = MultilayerCells(cells=cells_list)

        else:
            # create input noise for generator
            self.g_input = tf.unstack(input_noise, axis=1)

            # create multi-cell lstm for generator
            cells_list = []
            for i in range(self.gen_num_layers):
                g_input_dim = self.noise_dim if i == 0 else self.gen_num_units
                bilstmcell = Bilateral_LSTM_cell(input_dim=g_input_dim, hidden_dim=self.gen_num_units, scope_name="Discrete_generator/RNNCell_%d" % i)
                cells_list.append(bilstmcell)
            self.g_rnn_network = MultilayerCells(cells=cells_list)

        # initial state for multi-cell lstm
        initial_state = []
        for i in range(self.gen_num_layers):
            state_ = tf.stack([tf.zeros([self.batch_size, self.gen_num_units]),
                               tf.zeros([self.batch_size, self.gen_num_units])])
            initial_state.append(state_)

        return initial_state

    def gen_Onestep(self, t, state):
        with tf.variable_scope("Discrete_generator", regularizer=l2_regularizer(self.l2_scale)):
            # output is a tuple of (h_i, (h_i, c_i))
            cell_new_output_, new_state = self.g_rnn_network(input=self.g_input[t], state=state[0], state_=state[1])
            new_output_linear = tf.layers.dense(inputs=cell_new_output_, units=self.gen_dim, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE)
            return new_output_linear, new_state

    def build_Discriminator(self, input_discriminator):
        with tf.variable_scope("Discrete_discriminator", regularizer=l2_regularizer(self.l2_scale)):
            cell_units = []
            for num_units in range(self.dis_num_layers):
                cell = tf.nn.rnn_cell.LSTMCell(self.dis_num_units, reuse=tf.AUTO_REUSE)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                cell_units.append(cell)
            d_rnn_network  = tf.nn.rnn_cell.MultiRNNCell(cell_units)
            # initial_state = d_rnn_network.zero_state(self.batch_size, dtype=tf.float32)
            initial_state = rnn_init_state(init_="variable", batch_size=self.batch_size,
                                           num_layers=self.dis_num_layers, num_units=self.dis_num_units)
            outputs, _ = tf.nn.static_rnn(cell=d_rnn_network, inputs=input_discriminator, initial_state=initial_state)
            outputs = tf.stack(outputs, axis=1) 
            result = tf.layers.dense(tf.layers.flatten(outputs), 1, reuse=tf.AUTO_REUSE, activation=None) 
        return result