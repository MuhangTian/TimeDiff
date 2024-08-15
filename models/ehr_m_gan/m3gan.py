import tensorflow as tf
import numpy as np
import os
from models.ehr_m_gan.Contrastivelosslayer import nt_xent_loss
from models.ehr_m_gan.utils import ones_target, zeros_target, np_sigmoid, np_rounding
# from visualise import visualise_gan, visualise_vae

def reverse_normalize(data, min, max) -> np.ndarray:
    return (data * (max - min) + min)          # to turn back into real scale

class m3gan(object):
    def __init__(self, 
                 args, 
                 COLUMNS_DICT,
                 sess,
                 # -- shared params:
                 batch_size, time_steps,
                 num_pre_epochs, num_epochs,
                 checkpoint_dir, epoch_ckpt_freq, epoch_loss_freq,
                 # -- params for c
                 c_dim, c_noise_dim,
                 c_z_size, c_data_sample,
                 c_gan, c_vae,
                 # -- params for d
                 d_dim, d_noise_dim,
                 d_z_size, d_data_sample,
                 d_gan, d_vae,
                 # -- params for training
                 d_rounds, g_rounds, v_rounds,
                 v_lr_pre, v_lr, g_lr, d_lr,
                 alpha_re, alpha_kl, alpha_mt, 
                 alpha_ct, alpha_sm,
                 c_beta_adv, c_beta_fm, 
                 d_beta_adv, d_beta_fm, 
                 # -- label information
                 conditional=False, num_labels=0, statics_label=None):

        self.args = args
        self.COLUMNS_DICT = COLUMNS_DICT
        self.min_val_con = np.load(f"data/datasets/{self.args.data_name}/min.npy")
        self.max_val_con = np.load(f"data/datasets/{self.args.data_name}/max.npy")
        
        self.sess = sess
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.num_pre_epochs = num_pre_epochs
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.epoch_ckpt_freq = epoch_ckpt_freq
        self.epoch_loss_freq = epoch_loss_freq
        self.statics_label = statics_label

        # params for continuous
        self.c_dim = c_dim
        self.c_noise_dim = c_noise_dim
        self.c_z_size = c_z_size
        self.c_data_sample = c_data_sample
        self.c_rnn_vae_net = c_vae
        self.cgan = c_gan

        # params for discrete
        self.d_dim = d_dim
        self.d_noise_dim = d_noise_dim
        self.d_z_size = d_z_size
        self.d_data_sample = d_data_sample
        self.d_rnn_vae_net = d_vae
        self.dgan = d_gan

        # params for training
        self.d_rounds = d_rounds
        self.g_rounds = g_rounds
        self.v_rounds = v_rounds

        # params for learning rate
        self.v_lr_pre = v_lr_pre
        self.v_lr = v_lr
        self.g_lr = g_lr
        self.d_lr = d_lr
        
        # params for loss scalar
        self.alpha_re = alpha_re
        self.alpha_kl = alpha_kl
        self.alpha_mt = alpha_mt
        self.alpha_ct = alpha_ct
        self.alpha_sm = alpha_sm
        self.c_beta_adv = c_beta_adv
        self.c_beta_fm = c_beta_fm
        self.d_beta_adv = d_beta_adv
        self.d_beta_fm = d_beta_fm

        # params for label information
        self.num_labels = num_labels
        self.conditional = conditional

    def build(self):
        self.build_tf_graph()
        self.build_loss()
        self.build_summary()
        self.saver = tf.train.Saver()

    def save(self, global_id, model_name=None, checkpoint_dir=None):
        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, model_name), global_step=global_id)

    def load(self, model_name=None, checkpoint_dir=None):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        global_id = int(ckpt_name[len(model_name) + 1:])
        return global_id

    def build_tf_graph(self):
        # Step 1: VAE training -------------------------------------------------------------------------------------
        # Pretrain vae for c
        if self.conditional:
            self.real_data_label_pl = tf.compat.v1.placeholder(
                dtype=float, shape=[self.batch_size, self.num_labels], name="real_data_label")

        self.c_real_data_pl = tf.compat.v1.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.c_dim], name="continuous_real_data")

        if self.conditional:
            self.c_decoded_output, self.c_vae_sigma, self.c_vae_mu, self.c_vae_logsigma, self.c_enc_z = \
                self.c_rnn_vae_net.build_vae(self.c_real_data_pl, self.real_data_label_pl)
        else:
            self.c_decoded_output, self.c_vae_sigma, self.c_vae_mu, self.c_vae_logsigma, self.c_enc_z = \
                self.c_rnn_vae_net.build_vae(self.c_real_data_pl)

        # add validation set here -------
        self.c_vae_test_data_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.c_dim], name="vae_validation_c_data")
        if self.conditional:
            self.c_vae_test_decoded, _, _, _, _ = \
                self.c_rnn_vae_net.build_vae(self.c_vae_test_data_pl, self.real_data_label_pl)
        else:
            self.c_vae_test_decoded, _, _, _, _ = \
                self.c_rnn_vae_net.build_vae(self.c_vae_test_data_pl)

        # Pretrain vae for d
        self.d_real_data_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.d_dim], name="discrete_real_data")
        if self.conditional:
            self.d_decoded_output, self.d_vae_sigma, self.d_vae_mu, self.d_vae_logsigma, self.d_enc_z = \
                self.d_rnn_vae_net.build_vae(self.d_real_data_pl, self.real_data_label_pl)
        else:
            self.d_decoded_output, self.d_vae_sigma, self.d_vae_mu, self.d_vae_logsigma, self.d_enc_z = \
                self.d_rnn_vae_net.build_vae(self.d_real_data_pl)

        # add validation set here -------
        self.d_vae_test_data_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.d_dim], name="vae_validation_d_data")
        if self.conditional:
            self.d_vae_test_decoded, _, _, _, _ = \
                self.d_rnn_vae_net.build_vae(self.d_vae_test_data_pl, self.real_data_label_pl)
        else:
            self.d_vae_test_decoded, _, _, _, _ = \
                self.d_rnn_vae_net.build_vae(self.d_vae_test_data_pl)


        # Step2: Generator training --------------------------------------------------------------------------------
        # cgan - initialisation for continuous gan
        self.c_gen_input_noise_pl = tf.placeholder(
            tf.float32, [None, self.time_steps, self.c_noise_dim], name="continuous_generator_input_noise")
        if self.conditional:
            c_initial_state = self.cgan.build_GenRNN(self.c_gen_input_noise_pl, self.real_data_label_pl)
        else:
            c_initial_state = self.cgan.build_GenRNN(self.c_gen_input_noise_pl)

        # dgan - initialisation for discrete gan
        self.d_gen_input_noise_pl = tf.placeholder(
            tf.float32, [None, self.time_steps, self.d_noise_dim], name="discrete_generator_input_noise")
        if self.conditional:
            d_initial_state = self.dgan.build_GenRNN(self.d_gen_input_noise_pl, self.real_data_label_pl)
        else:
            d_initial_state = self.dgan.build_GenRNN(self.d_gen_input_noise_pl)

        ### sequentially coupled training steps
        self.c_gen_output_latent = []
        self.d_gen_output_latent = []

        for t in range(self.time_steps):
            d_state = d_new_state if t > 0 else d_initial_state
            c_state = c_new_state if t > 0 else c_initial_state

            # _new_state is a tuple of (h_i, c_i)
            d_new_linear, d_new_state = self.dgan.gen_Onestep(t, [d_state, c_state])
            c_new_linear, c_new_state = self.cgan.gen_Onestep(t, [c_state, d_state])

            self.d_gen_output_latent.append(d_new_linear)
            self.c_gen_output_latent.append(c_new_linear)

        # Step3: Decoder -------------------------------------------------------------------------------------------
        # dgan - decoding
        if self.conditional:
            self.d_gen_decoded = self.d_rnn_vae_net.reconstruct_decoder(dec_input=self.d_gen_output_latent,
                                                                        conditions=self.real_data_label_pl)
        else:
            self.d_gen_decoded = self.d_rnn_vae_net.reconstruct_decoder(dec_input=self.d_gen_output_latent)
        self.d_gen_decoded = tf.unstack(self.d_gen_decoded, axis=1)

        # cgan - decoding
        if self.conditional:
            self.c_gen_decoded = self.c_rnn_vae_net.reconstruct_decoder(dec_input=self.c_gen_output_latent,
                                                                        conditions=self.real_data_label_pl)
        else:
            self.c_gen_decoded = self.c_rnn_vae_net.reconstruct_decoder(dec_input=self.c_gen_output_latent)
        self.c_gen_decoded = tf.unstack(self.c_gen_decoded, axis=1)

        # Step4: Discriminator -------------------------------------------------------------------------------------
        # cgan - discriminator
        self.c_fake, self.c_fake_fm = self.cgan.build_Discriminator(self.c_gen_decoded)
        self.c_real, self.c_real_fm = self.cgan.build_Discriminator(tf.unstack(self.c_real_data_pl, axis=1))

        # dgan - discriminator
        self.d_fake = self.dgan.build_Discriminator(self.d_gen_decoded)
        self.d_real = self.dgan.build_Discriminator(tf.unstack(self.d_real_data_pl, axis=1))

    def build_loss(self):

        #################
        # (1) VAE loss  #
        #################
        alpha_re = self.alpha_re
        alpha_kl = self.alpha_kl
        alpha_mt = self.alpha_mt
        alpha_ct = self.alpha_ct
        alpha_sm = self.alpha_sm

        # 1. VAE loss for c
        self.c_re_loss = tf.compat.v1.losses.mean_squared_error(self.c_real_data_pl, self.c_decoded_output) # reconstruction loss for x(-[0,1]
        c_kl_loss = [0] * self.time_steps # KL divergence
        for t in range(self.time_steps):
            c_kl_loss[t] = 0.5 * (tf.reduce_sum(self.c_vae_sigma[t], 1) + tf.reduce_sum(
                tf.square(self.c_vae_mu[t]), 1) - tf.reduce_sum(self.c_vae_logsigma[t] + 1, 1))
        self.c_kl_loss = tf.reduce_mean(tf.add_n(c_kl_loss))

        # 2. Euclidean distance between latent representations from d and c
        x_latent_1 = tf.stack(self.c_enc_z, axis=1)
        x_latent_2 = tf.stack(self.d_enc_z, axis=1)
        self.vae_matching_loss = tf.compat.v1.losses.mean_squared_error(x_latent_1, x_latent_2)

        # 3. Contrastive loss
        self.vae_contra_loss = nt_xent_loss(tf.reshape(x_latent_1, [x_latent_1.shape[0], -1]),
                                            tf.reshape(x_latent_2, [x_latent_2.shape[0], -1]), self.batch_size)

        # 4. If label: conditional VAE and classification cross entropy loss
        if self.conditional:
            # exclude the label information from the latent vector
            x_latent_1_ = x_latent_1[:, :, :-1]
            x_latent_2_ = x_latent_2[:, :, :-1]
            with tf.variable_scope("Shared_VAE/semantic_classifier"):
                vae_flatten_input = tf.compat.v1.layers.flatten(tf.concat([x_latent_1_, x_latent_2_], axis=-1))
                vae_hidden_layer = tf.layers.dense(vae_flatten_input, units=24, activation=tf.nn.relu)
                vae_logits = tf.layers.dense(vae_hidden_layer, units=4, activation=tf.nn.tanh)
            self.vae_semantics_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.squeeze(tf.cast(self.real_data_label_pl, dtype=tf.int32)), logits=vae_logits))

        if self.conditional:
            self.c_vae_loss = alpha_re * self.c_re_loss + \
                              alpha_kl * self.c_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss + \
                              alpha_sm * self.vae_semantics_loss
        else:
            self.c_vae_loss = alpha_re * self.c_re_loss + \
                              alpha_kl * self.c_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss

        # vae validation loss
        self.c_vae_valid_loss = tf.compat.v1.losses.mean_squared_error(self.c_vae_test_data_pl, self.c_vae_test_decoded)

        # VAE loss for d (BCE loss)  
        # self.d_re_loss = tf.losses.mean_squared_error(self.d_real_data_pl, self.d_decoded_output) # reconstruction loss
        self.d_re_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(y_true=self.d_real_data_pl, y_pred=self.d_decoded_output, from_logits=False))

        d_kl_loss = [0] * self.time_steps # KL divergence
        for t in range(self.time_steps):
            d_kl_loss[t] = 0.5 * (tf.reduce_sum(self.d_vae_sigma[t], 1) + tf.reduce_sum(
                tf.square(self.d_vae_mu[t]), 1) - tf.reduce_sum(self.d_vae_logsigma[t] + 1, 1))
        self.d_kl_loss = 0.1 * tf.reduce_mean(tf.add_n(d_kl_loss))

        if self.conditional:
            self.d_vae_loss = alpha_re * self.d_re_loss + \
                              alpha_kl * self.d_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss + \
                              alpha_sm * self.vae_semantics_loss
        else:
            self.d_vae_loss = alpha_re * self.d_re_loss + \
                              alpha_kl * self.d_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss

        # vae validation loss
        self.d_vae_valid_loss = tf.compat.v1.losses.mean_squared_error(self.d_vae_test_data_pl, self.d_vae_test_decoded)

        ###########################
        # (2) Discriminator loss  #
        ###########################

        # cgan - discriminator loss (no activation function for discriminator therefore from logits)
        self.continuous_d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.c_real, labels=ones_target(self.batch_size, min=0.7, max=1.2)))
        self.continuous_d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.c_fake, labels=zeros_target(self.batch_size, min=0.1, max=0.3)))
        self.continuous_d_loss = self.continuous_d_loss_real + self.continuous_d_loss_fake

        # dgan - discriminator loss
        self.dicrete_d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_real, labels=ones_target(self.batch_size, min=0.8, max=0.9)))
        self.dicrete_d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_fake, labels=zeros_target(self.batch_size, min=0.1, max=0.1)))
        self.dicrete_d_loss = self.dicrete_d_loss_real + self.dicrete_d_loss_fake

        ###########################
        # (3) Generator loss      #
        ###########################
        # cgan - adversarial loss
        self.c_gen_loss_adv = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.c_fake, labels=tf.ones_like(self.c_fake)))

        # cgan - feature matching (statistical comparison between discriminative result from D)
        self.c_g_loss_v1 = tf.reduce_mean(
            tf.abs(tf.sqrt(tf.nn.moments(self.c_fake_fm, [0])[1] + 1e-6)
                   - tf.sqrt(tf.nn.moments(self.c_real_fm, [0])[1] + 1e-6)))

        self.c_g_loss_v2 = tf.reduce_mean(
            tf.abs(tf.sqrt(tf.abs(tf.nn.moments(self.c_fake_fm, [0])[0]))
                   - tf.sqrt(tf.abs(tf.nn.moments(self.c_real_fm, [0])[0]))))

        self.c_gen_loss_fm = self.c_g_loss_v1 + self.c_g_loss_v2

        # cgan - add two losses for generator
        c_beta_adv = self.c_beta_adv
        c_beta_fm = self.c_beta_fm
        self.c_gen_loss = c_beta_adv * self.c_gen_loss_adv + c_beta_fm * self.c_gen_loss_fm

        # dgan - adversarial loss
        self.d_gen_loss_adv = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.ones_like(self.d_fake)))

        # dgan - feature matching (statistical comparison between generated and real data)
        self.d_g_loss_v1 = tf.reduce_mean(
            tf.abs(tf.sqrt(tf.nn.moments(tf.stack(self.d_gen_decoded, axis=1), [0])[1] + 1e-6) -
                   tf.sqrt(tf.nn.moments(self.d_real_data_pl, [0])[1] + 1e-6)))
        self.d_g_loss_v2 = tf.reduce_mean(
            tf.abs((tf.nn.moments(tf.stack(self.d_gen_decoded, axis=1), [0])[0]) -
                   (tf.nn.moments(self.d_real_data_pl, [0])[0])))
        self.d_gen_loss_fm = self.d_g_loss_v1 + self.d_g_loss_v2

        # dgan - add two losses for generator
        d_beta_adv = self.d_beta_adv
        d_beta_fm = self.d_beta_fm
        self.d_gen_loss = d_beta_adv * self.d_gen_loss_adv + d_beta_fm * self.d_gen_loss_fm

        #######################
        # Optimizer           #
        #######################
        t_vars = tf.trainable_variables()
        c_vae_vars = [var for var in t_vars if 'Continuous_VAE' in var.name]
        d_vae_vars = [var for var in t_vars if 'Discrete_VAE' in var.name]
        s_vae_vars = [var for var in t_vars if 'Shared_VAE' in var.name]
        c_g_vars = [var for var in t_vars if 'Continuous_generator' in var.name]
        c_d_vars = [var for var in t_vars if 'Continuous_discriminator' in var.name]
        d_g_vars = [var for var in t_vars if 'Discrete_generator' in var.name]
        d_d_vars = [var for var in t_vars if 'Discrete_discriminator' in var.name]

        # Optimizer for c of vae
        self.c_v_op_pre = tf.compat.v1.train.AdamOptimizer(learning_rate=self.v_lr_pre)\
            .minimize(self.c_vae_loss, var_list=c_vae_vars+s_vae_vars)

        # Optimizer for d of vae
        self.d_v_op_pre = tf.compat.v1.train.AdamOptimizer(learning_rate=self.v_lr_pre)\
            .minimize(self.d_vae_loss, var_list=d_vae_vars+s_vae_vars)

        # Optimizer for c of vae
        self.c_v_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.v_lr) \
            .minimize(self.c_vae_loss, var_list=c_vae_vars + s_vae_vars)

        # Optimizer for d of vae
        self.d_v_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.v_lr) \
            .minimize(self.d_vae_loss, var_list=d_vae_vars + s_vae_vars)

        # Optimizer for c of generator
        self.c_g_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.g_lr)\
            .minimize(self.c_gen_loss, var_list=c_g_vars)

        # Optimizer for d of generator
        self.d_g_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.g_lr)\
            .minimize(self.d_gen_loss, var_list=d_g_vars)

        # Optimizer for c of discriminator
        self.c_d_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.d_lr)\
            .minimize(self.continuous_d_loss, var_list=c_d_vars)

        # Optimizer for d of discriminator
        self.d_d_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.d_lr)\
            .minimize(self.dicrete_d_loss, var_list=d_d_vars)

    def build_summary(self):

        # loss summary of variational autoencoder for c
        self.c_vae_summary = []
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/reconstruction_loss", self.c_re_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/kl_divergence_loss", self.c_kl_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/matching_loss", self.vae_matching_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/contrastive_loss", self.vae_contra_loss))
        if self.conditional:
            self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/semantic_loss", self.vae_semantics_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/vae_loss", self.c_vae_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/validation_loss", self.c_vae_valid_loss))
        self.c_vae_summary = tf.summary.merge(self.c_vae_summary)

        # loss summary of variational autoencoder for d
        self.d_vae_summary = []
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/reconstruction_loss", self.d_re_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/kl_divergence_loss", self.d_kl_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/matching_loss", self.vae_matching_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/contrastive_loss", self.vae_contra_loss))
        if self.conditional:
            self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/semantic_loss", self.vae_semantics_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/vae_loss", self.d_vae_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/validation_loss", self.d_vae_valid_loss))
        self.d_vae_summary = tf.summary.merge(self.d_vae_summary)

        # loss summary of discriminator for c
        self.c_discriminator_summary = []
        self.c_discriminator_summary.append(
            tf.summary.scalar("c_discriminator_loss/d_real", self.continuous_d_loss_real))
        self.c_discriminator_summary.append(
            tf.summary.scalar("c_discriminator_loss/d_fake", self.continuous_d_loss_fake))
        self.c_discriminator_summary.append(
            tf.summary.scalar("c_discriminator_loss/discriminator_loss", self.continuous_d_loss))
        self.c_discriminator_summary = tf.summary.merge(self.c_discriminator_summary)

        # loss summary of generator for c
        self.c_generator_summary = []
        self.c_generator_summary.append(
            tf.summary.scalar("c_generator_loss/adversarial_loss", self.c_gen_loss_adv))
        self.c_generator_summary.append(
            tf.summary.scalar("c_generator_loss/feature_matching_loss_v1", self.c_g_loss_v1))
        self.c_generator_summary.append(
            tf.summary.scalar("c_generator_loss/feature_matching_loss_v2", self.c_g_loss_v2))
        self.c_generator_summary.append(
            tf.summary.scalar("c_generator_loss/feature_matching_loss", self.c_gen_loss_fm))
        self.c_generator_summary.append(
            tf.summary.scalar("c_generator_loss/generator_loss", self.c_gen_loss))
        self.c_generator_summary = tf.summary.merge(self.c_generator_summary)

        # loss summary of discriminator for d
        self.d_discriminator_summary = []
        self.d_discriminator_summary.append(
            tf.summary.scalar("d_discriminator_loss/dicrete_d_loss_real", self.dicrete_d_loss_real))
        self.d_discriminator_summary.append(
            tf.summary.scalar("d_discriminator_loss/dicrete_d_loss_fake", self.dicrete_d_loss_fake))
        self.d_discriminator_summary.append(
            tf.summary.scalar("d_discriminator_loss/d_discriminator", self.dicrete_d_loss))
        self.d_discriminator_summary = tf.summary.merge(self.d_discriminator_summary)

        # loss summary of discriminator for d
        self.d_generator_summary = []
        self.d_generator_summary.append(
            tf.summary.scalar("d_generator_loss/g_loss_v1", self.d_g_loss_v1))
        self.d_generator_summary.append(
            tf.summary.scalar("d_generator_loss/g_loss_v2", self.d_g_loss_v2))
        self.d_generator_summary.append(
            tf.summary.scalar("d_generator_loss/d_gen_loss_fm", self.d_gen_loss_fm))
        self.d_generator_summary.append(
            tf.summary.scalar("d_generator_loss/d_gen_loss_adv", self.d_gen_loss_adv))
        self.d_generator_summary.append(
            tf.summary.scalar("d_generator_loss/d_generator", self.d_gen_loss))
        self.d_generator_summary = tf.summary.merge(self.d_generator_summary)

    def gen_input_noise(self, num_sample, T, noise_dim):
        return np.random.uniform(size=[num_sample, T, noise_dim])

    def train(self):
        # self.summary_writer = tf.summary.FileWriter("logs/tf_summary", self.sess.graph)

        #  prepare training data for c
        continuous_x = self.c_data_sample[: int(0.9 * self.c_data_sample.shape[0]), :, :]
        continuous_x_test = self.c_data_sample[int(0.9 * self.c_data_sample.shape[0]) : , :, :]

        # prepare training data for d
        discrete_x = self.d_data_sample[: int(0.9 * self.d_data_sample.shape[0]), :, :]
        discrete_x_test = self.d_data_sample[int(0.9 * self.d_data_sample.shape[0]):, :, :]

        # prepare training data for label
        if self.conditional:
            label_data = self.statics_label[: int(0.9 * self.d_data_sample.shape[0]), :]

        # num of batches
        data_size = continuous_x.shape[0]
        num_batches = data_size // self.batch_size

        tf.global_variables_initializer().run()

        # pretrain step
        print('start pretraining')
        global_id = 0

        for pre in range(self.num_pre_epochs):

            # prepare data for training dataset (same index)
            random_idx = np.random.permutation(data_size)
            continuous_x_random = continuous_x[random_idx]
            discrete_x_random = discrete_x[random_idx]
            if self.conditional:
                label_data_random = label_data[random_idx]

            # validation data
            random_idx_ = np.random.permutation(continuous_x_test.shape[0])
            continuous_x_test_batch = continuous_x_test[random_idx_][:self.batch_size, :, :]
            discrete_x_test_batch = discrete_x_test[random_idx_][:self.batch_size, :, :]

            print("pretraining epoch %d" % pre)

            c_real_data_lst = []
            c_rec_data_lst = []
            d_real_data_lst = []
            d_rec_data_lst = []

            for b in range(num_batches):

                feed_dict = {}
                # feed d data
                feed_dict[self.c_real_data_pl] = continuous_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.c_vae_test_data_pl] = continuous_x_test_batch
                # feed c data
                feed_dict[self.d_real_data_pl] = discrete_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.d_vae_test_data_pl] = discrete_x_test_batch
                # feed label
                if self.conditional:
                    feed_dict[self.real_data_label_pl] = label_data_random[b * self.batch_size: (b + 1) * self.batch_size]

                # Pretrain the discrete and continuous vae loss
                _ = self.sess.run(self.c_v_op_pre, feed_dict=feed_dict)
                if ((pre + 1) % self.epoch_loss_freq == 0 or pre == self.num_pre_epochs - 1):
                    summary_result = self.sess.run(self.c_vae_summary, feed_dict=feed_dict)
                    # self.summary_writer.add_summary(summary_result, global_id)

                _ = self.sess.run(self.d_v_op_pre, feed_dict=feed_dict)
                if ((pre + 1) % self.epoch_loss_freq == 0 or pre == self.num_pre_epochs - 1):
                    summary_result = self.sess.run(self.d_vae_summary, feed_dict=feed_dict)
                    # self.summary_writer.add_summary(summary_result, global_id)

                global_id += 1
            
                # if ((pre + 1) % self.epoch_ckpt_freq == 0 or pre == self.num_pre_epochs - 1):
                #     # real data vs. reconstructed data 
                #     real_data, rec_data = self.sess.run([self.c_real_data_pl, self.c_decoded_output], feed_dict=feed_dict)
                #     c_real_data_lst.append(real_data)
                #     c_rec_data_lst.append(rec_data)

                #     # real data vs. reconstructed data (rounding to 0 or 1)
                #     real_data, rec_data = self.sess.run([self.d_real_data_pl, self.d_decoded_output], feed_dict=feed_dict)
                #     d_real_data_lst.append(real_data)
                #     d_rec_data_lst.append(np_rounding( rec_data) )
                
            # visualize
            # if ((pre + 1) % self.epoch_ckpt_freq == 0 or pre == self.num_pre_epochs - 1):
                # visualise_vae(continuous_x_random, np.vstack(c_rec_data_lst), discrete_x_random, np.vstack(d_rec_data_lst), inx=(pre+1))
                # print('finish vae reconstructed data saving in pre-epoch ' + str(pre))

        # np.savez('data/fake/vae.npz', c_real=np.vstack(c_real_data_lst), c_rec=np.vstack(c_rec_data_lst),
        #                               d_real=np.vstack(d_real_data_lst), d_rec=np.vstack(d_rec_data_lst))

        # saving the pre-trained model
        # save_path = os.path.join(self.checkpoint_dir, "pretrain_vae_{}".format(global_id))
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # self.save(global_id=global_id - 1, model_name='m3gan', checkpoint_dir=save_path)
        print('finish the pretraining')

        # start jointly training ------
        print('start joint training')

        for e in range(self.num_epochs):
            # prepare data for training dataset (same index)
            random_idx = np.random.permutation(data_size)
            continuous_x_random = continuous_x[random_idx]
            discrete_x_random = discrete_x[random_idx]
            if self.conditional:
                label_data_random = label_data[random_idx]

            # validation data
            random_idx_ = np.random.permutation(continuous_x_test.shape[0])
            continuous_x_test_batch = continuous_x_test[random_idx_][:self.batch_size, :, :]
            discrete_x_test_batch = discrete_x_test[random_idx_][:self.batch_size, :, :]

            print("training epoch %d" % e)

            for b in range(num_batches):
                feed_dict = {}
                # feed c
                feed_dict[self.c_real_data_pl] = continuous_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.c_gen_input_noise_pl] = self.gen_input_noise(self.batch_size, self.time_steps, noise_dim=self.c_noise_dim)
                feed_dict[self.c_vae_test_data_pl] = continuous_x_test_batch
                # feed d
                feed_dict[self.d_real_data_pl] = discrete_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.d_gen_input_noise_pl] = self.gen_input_noise(self.batch_size, self.time_steps, noise_dim=self.d_noise_dim)
                feed_dict[self.d_vae_test_data_pl] = discrete_x_test_batch
                # if conditional, feed label
                if self.conditional:
                    feed_dict[self.real_data_label_pl] = label_data_random[b * self.batch_size: (b + 1) * self.batch_size]

                # training d
                for _ in range(self.d_rounds):
                    #_, d_summary_result = self.sess.run([self.d_d_op, self.d_discriminator_summary], feed_dict=feed_dict)
                    _ = self.sess.run(self.d_d_op, feed_dict=feed_dict)
                    # if ((e + 1) % self.epoch_loss_freq == 0 or e == self.num_epochs - 1):
                        # d_summary_result = self.sess.run(self.d_discriminator_summary, feed_dict=feed_dict)
                        # self.summary_writer.add_summary(d_summary_result, global_id)

                    _ = self.sess.run(self.c_d_op, feed_dict=feed_dict)
                    # if ((e + 1) % self.epoch_loss_freq == 0 or e == self.num_epochs - 1):
                        # c_summary_result = self.sess.run(self.c_discriminator_summary, feed_dict=feed_dict)
                        # self.summary_writer.add_summary(c_summary_result, global_id)

                # training g
                for _ in range(self.g_rounds):
                    _ = self.sess.run(self.d_g_op, feed_dict=feed_dict)
                    # if ((e + 1) % self.epoch_loss_freq == 0 or e == self.num_epochs - 1):
                        # d_summary_result = self.sess.run(self.d_generator_summary, feed_dict=feed_dict)
                        # self.summary_writer.add_summary(d_summary_result, global_id)

                    _ = self.sess.run(self.c_g_op, feed_dict=feed_dict)
                    # if ((e + 1) % self.epoch_loss_freq == 0 or e == self.num_epochs - 1):
                        # c_summary_result = self.sess.run(self.c_generator_summary, feed_dict=feed_dict)
                        # self.summary_writer.add_summary(c_summary_result, global_id)

                # training v
                for _ in range(self.v_rounds):
                    _ = self.sess.run(self.d_v_op, feed_dict=feed_dict)
                    # if ((e + 1) % self.epoch_loss_freq == 0 or e == self.num_epochs - 1):
                        # summary_result = self.sess.run(self.d_vae_summary, feed_dict=feed_dict)
                        # self.summary_writer.add_summary(summary_result, global_id)

                    _ = self.sess.run(self.c_v_op, feed_dict=feed_dict)
                    # if ((e + 1) % self.epoch_loss_freq == 0 or e == self.num_epochs - 1):
                        # summary_result = self.sess.run(self.d_vae_summary, feed_dict=feed_dict)
                        # self.summary_writer.add_summary(summary_result, global_id)

                global_id += 1

            if ((e + 1) % self.epoch_ckpt_freq == 0 or e == self.num_epochs - 1):
            #     data_gen_path = os.path.join("data/fake/", "epoch{}".format(e))
            #     if not os.path.exists(data_gen_path):
            #         os.makedirs(data_gen_path)
            #     if self.conditional:
            #         d_gen_data, c_gen_data = self.generate_data(num_sample=self.c_data_sample.shape[0], labels=self.statics_label)
            #     else:
                d_gen_data, c_gen_data = self.generate_data(num_sample=self.c_data_sample.shape[0])
                num_samples = len(d_gen_data)
                d_gen_data, c_gen_data = np.transpose(d_gen_data, (0, 2, 1)), np.transpose(c_gen_data, (0, 2, 1))
                
                if self.args.data_name in ["mimiciv", "eicu", "mimiciii", "hirid"]:
                    gen_data_whole = np.zeros((num_samples, self.c_dim+self.d_dim, self.time_steps))
                    gen_data_whole[:, self.COLUMNS_DICT[self.args.data_name]["numerical"], :] = c_gen_data
                    gen_data_whole[:, self.COLUMNS_DICT[self.args.data_name]["categorical"], :] = d_gen_data
                else:
                    gen_data_whole = c_gen_data
            
                gen_data_whole = reverse_normalize(gen_data_whole, self.min_val_con, self.max_val_con)
                np.save(self.args.gen_path, gen_data_whole)
                # np.savez(os.path.join(data_gen_path, "gen_data.npz"), c_gen_data=c_gen_data, d_gen_data=d_gen_data)
                # visualise_gan(continuous_x_random, c_gen_data, discrete_x_random, d_gen_data, inx=(e+1))
                # print('finish generated data saving in epoch ' + str(e))

    def generate_data(self, num_sample, labels=None):
        d_gen_data = []
        c_gen_data = []
        round_ = num_sample // self.batch_size

        for i in range(round_):
            d_gen_input_noise = self.gen_input_noise(
                num_sample=self.batch_size, T=self.time_steps, noise_dim=self.d_noise_dim)
            c_gen_input_noise = self.gen_input_noise(
                num_sample=self.batch_size, T=self.time_steps, noise_dim=self.c_noise_dim)

            feed_dict = {}
            feed_dict[self.d_gen_input_noise_pl] = d_gen_input_noise
            feed_dict[self.c_gen_input_noise_pl] = c_gen_input_noise
            if self.conditional:
                feed_dict[self.real_data_label_pl] = \
                    labels[i * self.batch_size: (i + 1) * self.batch_size]

            d_gen_data_, c_gen_data_ = self.sess.run([self.d_gen_decoded, self.c_gen_decoded], feed_dict=feed_dict)
            d_gen_data.append(np.stack(d_gen_data_, axis=1))
            c_gen_data.append(np.stack(c_gen_data_, axis=1))

        d_gen_data = np.concatenate(d_gen_data, axis=0)
        c_gen_data = np.concatenate(c_gen_data, axis=0)

        return np_rounding(d_gen_data), c_gen_data