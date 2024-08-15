import argparse
import os
import pickle
import timeit

import numpy as np
import tensorflow as tf

from helpers.utils import (create_id, exists, is_file_not_on_disk,
                           is_file_on_disk, is_positive_float,
                           is_positive_integer, reverse_normalize,
                           seed_everything)
from models.ehr_m_gan.m3gan import m3gan
from models.ehr_m_gan.networks import (C_GAN_NET, C_VAE_NET, D_GAN_NET,
                                       D_VAE_NET)
from models.ehr_m_gan.utils import renormlizer

COLUMNS_DICT = {
    "mimiciv": {"numerical": [0, 2, 4, 6, 8], "categorical": [1, 3, 5, 7, 9, 10], "categorical_num_classes": [2, 2, 2, 2, 2, 2]},
    "mimiciii": {"numerical": [0, 2, 4, 6, 8, 10, 12], "categorical": [1, 3, 5, 7, 9, 11, 13, 14], "categorical_num_classes": [2, 2, 2, 2, 2, 2, 2, 2]},
    "eicu": {"numerical": [0, 2, 4, 6], "categorical": [1, 3, 5, 7, 8], "categorical_num_classes": [2, 2, 2, 2, 2]},
    "hirid": {"numerical": [0, 1, 2, 3, 4, 5, 6], "categorical": [7], "categorical_num_classes": [2]},
}

def main(args):
    dataset = np.load(args.load_path)
    
    if args.data_name in ["mimiciv", "eicu", "mimiciii", "hirid"]:
        continuous_x = dataset[:, COLUMNS_DICT[args.data_name]["numerical"], :]
        discrete_x = dataset[:, COLUMNS_DICT[args.data_name]["categorical"], :]
    else:           # all other datasets are real-valued (stocks and energy)
        continuous_x = dataset
        discrete_x = np.ones_like(continuous_x) # dummy discrete data
        
    continuous_x = np.transpose(continuous_x, (0, 2, 1))
    discrete_x = np.transpose(discrete_x, (0, 2, 1))
   
    # prepare data for training GAN
    # with open(os.path.join('data/real/', args.dataset, 'vital_sign_24hrs.pkl'), 'rb') as f:
    #     continuous_x = pickle.load(f)

    # with open(os.path.join('data/real/', args.dataset, 'med_interv_24hrs.pkl'), 'rb') as f:
    #     discrete_x = pickle.load(f)

    # with open(os.path.join('data/real/', args.dataset, 'statics.pkl'), 'rb') as f:
    #     statics_label = pickle.load(f)
    # statics_label = np.asarray(statics_label)[:, 0].reshape([-1, 1])
    
    statics_label = None

    # required shape (batch_size, time_steps, dim)
    time_steps = continuous_x.shape[1]
    c_dim = continuous_x.shape[2]
    d_dim = discrete_x.shape[2]
    no_gen = continuous_x.shape[0]

    shared_latent_dim = 25
    c_z_size = shared_latent_dim
    c_noise_dim = int(c_dim/2)
    d_z_size = shared_latent_dim
    d_noise_dim = int(d_dim/2)

    # networks for continuousGAN
    c_vae = C_VAE_NET(batch_size=args.batch_size, time_steps=time_steps, dim=c_dim, z_dim=c_z_size,
                    enc_size=args.enc_size, dec_size=args.dec_size, 
                    enc_layers=args.enc_layers, dec_layers=args.dec_layers, 
                    keep_prob=args.keep_prob, l2scale=args.l2_scale,
                    conditional=args.conditional, num_labels=args.num_labels)

    c_gan = C_GAN_NET(batch_size=args.batch_size, noise_dim=c_noise_dim, dim=c_dim,
                    gen_num_units=args.gen_num_units, gen_num_layers=args.gen_num_layers,
                    dis_num_units=args.dis_num_units, dis_num_layers=args.dis_num_layers,
                    keep_prob=args.keep_prob, l2_scale=args.l2_scale,
                    gen_dim=c_z_size, time_steps=time_steps,
                    conditional=args.conditional, num_labels=args.num_labels)

    # networks for discreteGAN
    d_vae = D_VAE_NET(batch_size=args.batch_size, time_steps=time_steps, dim=d_dim, z_dim=d_z_size,
                    enc_size=args.enc_size, dec_size=args.dec_size, 
                    enc_layers=args.enc_layers, dec_layers=args.dec_layers, 
                    keep_prob=args.keep_prob, l2scale=args.l2_scale,
                    conditional=args.conditional, num_labels=args.num_labels)

    d_gan = D_GAN_NET(batch_size=args.batch_size, noise_dim=d_noise_dim, dim=d_dim,
                    gen_num_units=args.gen_num_units, gen_num_layers=args.gen_num_layers,
                    dis_num_units=args.dis_num_units, dis_num_layers=args.dis_num_layers,
                    keep_prob=args.keep_prob, l2_scale=args.l2_scale,
                    gen_dim=d_z_size, time_steps=time_steps,
                    conditional=args.conditional, num_labels=args.num_labels)

    # create data directory for saving
    checkpoint_dir = args.check_point_path
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    start = timeit.default_timer()
    tf.reset_default_graph()
    run_config = tf.ConfigProto()
    with tf.Session(config=run_config) as sess:
        model = m3gan(args=args,
                    COLUMNS_DICT=COLUMNS_DICT,
                    sess=sess,
                    batch_size=args.batch_size,
                    time_steps=time_steps,
                    num_pre_epochs=args.num_pre_epochs,
                    num_epochs=args.num_epochs,
                    checkpoint_dir=checkpoint_dir,
                    epoch_ckpt_freq=args.epoch_ckpt_freq,
                    epoch_loss_freq=args.epoch_loss_freq,
                    # params for c
                    c_dim=c_dim, c_noise_dim=c_noise_dim,
                    c_z_size=c_z_size, c_data_sample=continuous_x,
                    c_vae=c_vae, c_gan=c_gan,
                    # params for d
                    d_dim=d_dim, d_noise_dim=d_noise_dim,
                    d_z_size=d_z_size, d_data_sample=discrete_x,
                    d_vae=d_vae, d_gan=d_gan,
                    # params for training
                    d_rounds=args.d_rounds, g_rounds=args.g_rounds, v_rounds=args.v_rounds,
                    v_lr_pre=args.v_lr_pre, v_lr=args.v_lr, g_lr=args.g_lr, d_lr=args.d_lr,
                    alpha_re=args.alpha_re, alpha_kl=args.alpha_kl, alpha_mt=args.alpha_mt, 
                    alpha_ct=args.alpha_ct, alpha_sm=args.alpha_sm,
                    c_beta_adv=args.c_beta_adv, c_beta_fm=args.c_beta_fm, 
                    d_beta_adv=args.d_beta_adv, d_beta_fm=args.d_beta_fm, 
                    # input label
                    conditional=args.conditional, num_labels=args.num_labels,
                    statics_label=statics_label)
        model.build()
        model.train()

        # evaluation
        d_gen_data, c_gen_data = model.generate_data(num_sample=no_gen)
    
    # renormalize
    assert len(d_gen_data) == len(c_gen_data)
    num_samples = len(d_gen_data)
    d_gen_data, c_gen_data = np.transpose(d_gen_data, (0, 2, 1)), np.transpose(c_gen_data, (0, 2, 1))
    
    if args.data_name in ["mimiciv", "eicu", "mimiciii", "hirid"]:
        gen_data_whole = np.zeros((num_samples, c_dim+d_dim, time_steps))
        gen_data_whole[:, COLUMNS_DICT[args.data_name]["numerical"], :] = c_gen_data
        gen_data_whole[:, COLUMNS_DICT[args.data_name]["categorical"], :] = d_gen_data
    else:
        gen_data_whole = c_gen_data
        
    min_val_con = np.load(f"data/datasets/{args.data_name}/min.npy")
    max_val_con = np.load(f"data/datasets/{args.data_name}/max.npy")
    gen_data_whole = reverse_normalize(gen_data_whole, min_val_con, max_val_con)
    np.save(args.gen_path, gen_data_whole)

    stop = timeit.default_timer()
    # print('Time: ', stop - start)


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='The batch size for training the model.')
    parser.add_argument('--num_pre_epochs', type=int, default=500, help='The number of epoches in pretraining the VAEs.')
    parser.add_argument('--num_epochs', type=int, default=800, help='The number of epoches in training the GANs.')
    parser.add_argument('--epoch_ckpt_freq', type=int, default=100, help='The frequency of epoches for saving models and synthetic data.')
    parser.add_argument('--epoch_loss_freq', type=int, default=100, help='The frequency of epoches for display the losses.')
    parser.add_argument('--d_rounds', type=int, default=1, help='The number of rounds for training discriminators per epoch.')
    parser.add_argument('--g_rounds', type=int, default=3, help='The number of rounds for training generators per epoch.')
    parser.add_argument('--v_rounds', type=int, default=1, help='The number of rounds for updating VAEs per epoch.')
    
    parser.add_argument('--v_lr_pre', type=float, default=0.0005, help='The learning rate for pretraining the VAEs.')
    parser.add_argument('--v_lr', type=float, default=0.0001, help='The learning rate for updating the VAEs during the adversarial training steps.')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='The learning rate for training generators during the adversarial training steps.')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='The learning rate for training discriminators during the adversarial training steps.')

    parser.add_argument('--alpha_re', type=float, default=1, help='The weight scalar of reconstruction loss in training VAEs.')
    parser.add_argument('--alpha_kl', type=float, default=0.5, help='The weight scalar of KL divergence in training VAEs.')
    parser.add_argument('--alpha_mt', type=float, default=0.1, help='The weight scalar of matching loss in training VAEs.')
    parser.add_argument('--alpha_ct', type=float, default=0.1, help='The weight scalar of contrastive loss in training VAEs.')
    parser.add_argument('--alpha_sm', type=float, default=1, help='The weight scalar of semantic loss in training VAEs.')

    parser.add_argument('--c_beta_adv', type=float, default=1, help='The weight scalar of adversarial loss in training continuous-GANs.')
    parser.add_argument('--c_beta_fm', type=float, default=20, help='The weight scalar of feature matching loss in training continuous-GANs.')
    parser.add_argument('--d_beta_adv', type=float, default=1, help='The weight scalar of adversarial loss in training discrete-GANs.')
    parser.add_argument('--d_beta_fm', type=float, default=20, help='The weight scalar of feature matching loss in training discrete-GANs.')

    parser.add_argument('--enc_size', type=int, default=128, help='The size of encoders.')
    parser.add_argument('--dec_size', type=int, default=128, help='The size of decoders.')
    parser.add_argument('--enc_layers', type=int, default=3, help='The layer numbers of encoders.')
    parser.add_argument('--dec_layers', type=int, default=3, help='The layer numbers of decoders')

    parser.add_argument('--gen_num_units', type=int, default=512, help='The size of genrators.')
    parser.add_argument('--gen_num_layers', type=int, default=3, help='The layer numbers of generators.')
    parser.add_argument('--dis_num_units', type=int, default=256, help='The size of discriminators.')
    parser.add_argument('--dis_num_layers', type=int, default=3, help='The layer numbers of discriminators.')

    parser.add_argument('--keep_prob', type=float, default=0.8, help='The dropout rate in the LSTM networks.')
    parser.add_argument('--l2_scale', type=float, default=0.001, help='The l2 scale regularizer in the LSTM networks.')

    parser.add_argument('--conditional', type=bool, default=False, help='Whether using the extension of Conditional-GAN.')
    parser.add_argument('--num_labels', type=int, default=1, help='The number of conditional labels in Conditional-GAN.')
    
    parser.add_argument("--data_name", type=str, required=True, help="The name of the dataset.")
    parser.add_argument("--load_path", type=is_file_on_disk, required=True, help="Path to load the dataset.")
    parser.add_argument("--check_point_path", type=str, required=True, help="Path to checkpoint models.")
    parser.add_argument("--gen_path", type=is_file_not_on_disk, required=True, help="Path to save generated data")

    args = parser.parse_args() 
  
    main(args)