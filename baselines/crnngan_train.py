"""
Code implementation for C-RNN-GAN, we referenced the following
source (please see below).
-----------------------------------------------------------------------------

Copyright 2019 Christopher John Bayron

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
    
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file has been created by Christopher John Bayron based on "rnn_gan.py"
by Olof Mogren. The referenced code is available in:

    https://github.com/olofmogren/c-rnn-gan
"""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import wandb
from models.crnn_gan.c_rnn_gan import Discriminator, Generator

# from models.crnn_gan import music_data_utils

# DATA_DIR = 'data'
# CKPT_DIR = 'models'
# COMPOSER = 'sonata-ish'

# G_FN = 'c_rnn_gan_g.pth'
# D_FN = 'c_rnn_gan_d.pth'

# G_LRN_RATE = 0.001
# D_LRN_RATE = 0.001
MAX_GRAD_NORM = 5.0
# following values are modified at runtime
# MAX_SEQ_LEN = 200
# BATCH_SIZE = 32

EPSILON = 1e-40 # value to use to approximate zero (to prevent undefined results)

class GLoss(nn.Module):
    ''' C-RNN-GAN generator loss
    '''
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, logits_gen):
        logits_gen = torch.clamp(logits_gen, EPSILON, 1.0)
        batch_loss = -torch.log(logits_gen)

        return torch.mean(batch_loss)


class DLoss(nn.Module):
    ''' C-RNN-GAN discriminator loss
    '''
    def __init__(self, label_smoothing=False):
        super(DLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits_real, logits_gen):
        ''' Discriminator loss

        logits_real: logits from D, when input is real
        logits_gen: logits from D, when input is from Generator

        loss = -(ylog(p) + (1-y)log(1-p))

        '''
        logits_real = torch.clamp(logits_real, EPSILON, 1.0)
        d_loss_real = -torch.log(logits_real)

        if self.label_smoothing:
            p_fake = torch.clamp((1 - logits_real), EPSILON, 1.0)
            d_loss_fake = -torch.log(p_fake)
            d_loss_real = 0.9*d_loss_real + 0.1*d_loss_fake

        logits_gen = torch.clamp((1 - logits_gen), EPSILON, 1.0)
        d_loss_gen = -torch.log(logits_gen)

        batch_loss = d_loss_real + d_loss_gen
        return torch.mean(batch_loss)

class TimeSeriesLoader:
    def __init__(self, data, cut_length, batch_size):
        """Assumes data is of shape (num_samples, seq_length, num_features)"""
        self.data = data
        self.cut_length = cut_length
        self.batch_size = batch_size
        self.num_samples, self.seq_length, self.num_features = data.shape
        self.num_batches = self.num_samples // self.batch_size
        self.indices = np.arange(self.num_samples)
        self.shuffle()
        self.iter_count = 0
    
    def shuffle(self):
        np.random.shuffle(self.indices)
    
    def get_num_song_features(self):
        return self.num_features

    def get_batch(self):
        """Returns a batch of shape (batch_size, seq_length, num_features)"""
        batch_indices = self.indices[:self.batch_size]
        self.indices = np.roll(self.indices, -self.batch_size)
        batch = self.data[batch_indices]
        self.iter_count += 1
        if self.iter_count >= int(self.num_samples // self.batch_size):
            self.iter_count = 0
            self.shuffle()
        return batch


def run_training(model, optimizer, criterion, dataloader, freeze_g=False, freeze_d=False):
    ''' Run single training epoch
    '''
    
    num_feats = dataloader.get_num_song_features()
    # dataloader.rewind(part='train')
    batch_song = dataloader.get_batch()

    model['g'].train()
    model['d'].train()

    loss = {}
    g_loss_total = 0.0
    d_loss_total = 0.0
    num_corrects = 0
    num_sample = 0

    while dataloader.iter_count != 0:

        real_batch_sz = batch_song.shape[0]

        # get initial states
        # each batch is independent i.e. not a continuation of previous batch
        # so we reset states for each batch
        # POSSIBLE IMPROVEMENT: next batch is continuation of previous batch
        g_states = model['g'].init_hidden(real_batch_sz)
        d_state = model['d'].init_hidden(real_batch_sz)

        #### GENERATOR ####
        if not freeze_g:
            optimizer['g'].zero_grad()
        # prepare inputs
        z = torch.empty([real_batch_sz, MAX_SEQ_LEN, num_feats]).uniform_() # random vector
        batch_song = torch.Tensor(batch_song)

        # feed inputs to generator
        g_feats, _ = model['g'](z, g_states)

        # calculate loss, backprop, and update weights of G
        if isinstance(criterion['g'], GLoss):
            d_logits_gen, _, _ = model['d'](g_feats, d_state)
            loss['g'] = criterion['g'](d_logits_gen)
        else: # feature matching
            # feed real and generated input to discriminator
            _, d_feats_real, _ = model['d'](batch_song, d_state)
            _, d_feats_gen, _ = model['d'](g_feats, d_state)
            loss['g'] = criterion['g'](d_feats_real, d_feats_gen)

        if not freeze_g:
            loss['g'].backward()
            nn.utils.clip_grad_norm_(model['g'].parameters(), max_norm=MAX_GRAD_NORM)
            optimizer['g'].step()

        #### DISCRIMINATOR ####
        if not freeze_d:
            optimizer['d'].zero_grad()
        # feed real and generated input to discriminator
        d_logits_real, _, _ = model['d'](batch_song, d_state)
        # need to detach from operation history to prevent backpropagating to generator
        d_logits_gen, _, _ = model['d'](g_feats.detach(), d_state)
        # calculate loss, backprop, and update weights of D
        loss['d'] = criterion['d'](d_logits_real, d_logits_gen)
        if not freeze_d:
            loss['d'].backward()
            nn.utils.clip_grad_norm_(model['d'].parameters(), max_norm=MAX_GRAD_NORM)
            optimizer['d'].step()

        g_loss_total += loss['g'].item()
        d_loss_total += loss['d'].item()
        num_corrects += (d_logits_real > 0.5).sum().item() + (d_logits_gen < 0.5).sum().item()
        num_sample += real_batch_sz

        # fetch next batch
        batch_song = dataloader.get_batch()

    g_loss_avg, d_loss_avg = 0.0, 0.0
    d_acc = 0.0
    if num_sample > 0:
        g_loss_avg = g_loss_total / num_sample
        d_loss_avg = d_loss_total / num_sample
        d_acc = 100 * num_corrects / (2 * num_sample) # 2 because (real + generated)

    return model, g_loss_avg, d_loss_avg, d_acc


def run_validation(model, criterion, dataloader):
    ''' Run single validation epoch
    '''
    num_feats = dataloader.get_num_song_features()
    # dataloader.rewind(part='validation')
    batch_song = dataloader.get_batch()

    model['g'].eval()
    model['d'].eval()

    g_loss_total = 0.0
    d_loss_total = 0.0
    num_corrects = 0
    num_sample = 0

    while batch_meta is not None and batch_song is not None:

        real_batch_sz = batch_song.shape[0]

        # initial states
        g_states = model['g'].init_hidden(real_batch_sz)
        d_state = model['d'].init_hidden(real_batch_sz)

        #### GENERATOR ####
        # prepare inputs
        z = torch.empty([real_batch_sz, MAX_SEQ_LEN, num_feats]).uniform_() # random vector
        batch_song = torch.Tensor(batch_song)

        # feed inputs to generator
        g_feats, _ = model['g'](z, g_states)
        # feed real and generated input to discriminator
        d_logits_real, d_feats_real, _ = model['d'](batch_song, d_state)
        d_logits_gen, d_feats_gen, _ = model['d'](g_feats, d_state)
        # calculate loss
        if isinstance(criterion['g'], GLoss):
            g_loss = criterion['g'](d_logits_gen)
        else: # feature matching
            g_loss = criterion['g'](d_feats_real, d_feats_gen)

        d_loss = criterion['d'](d_logits_real, d_logits_gen)

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()
        num_corrects += (d_logits_real > 0.5).sum().item() + (d_logits_gen < 0.5).sum().item()
        num_sample += real_batch_sz

        # fetch next batch
        batch_song = dataloader.get_batch()

    g_loss_avg, d_loss_avg = 0.0, 0.0
    d_acc = 0.0
    if num_sample > 0:
        g_loss_avg = g_loss_total / num_sample
        d_loss_avg = d_loss_total / num_sample
        d_acc = 100 * num_corrects / (2 * num_sample) # 2 because (real + generated)

    return g_loss_avg, d_loss_avg, d_acc


def run_epoch(model, optimizer, criterion, dataloader, ep, num_ep,
              freeze_g=False, freeze_d=False, pretraining=False, wandb=None):
    ''' Run a single epoch
    '''
    model, trn_g_loss, trn_d_loss, trn_acc = \
        run_training(model, optimizer, criterion, dataloader, freeze_g=freeze_g, freeze_d=freeze_d)

    # val_g_loss, val_d_loss, val_acc = run_validation(model, criterion, dataloader)

    if pretraining:
        print("Pretraining Epoch %d/%d " % (ep+1, num_ep), "[Freeze G: ", freeze_g, ", Freeze D: ", freeze_d, "]")
    else:
        print("Epoch %d/%d " % (ep+1, num_ep), "[Freeze G: ", freeze_g, ", Freeze D: ", freeze_d, "]")

    print("\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n" %
          (trn_g_loss, trn_d_loss, trn_acc))
    if wandb is not None:
        wandb.log({"g_loss": trn_g_loss, "d_loss": trn_d_loss, "d_acc": trn_acc})

    # -- DEBUG --
    # This is for monitoring the current output from generator
    # generate from model then save to MIDI file
    g_states = model['g'].init_hidden(1)
    num_feats = dataloader.get_num_song_features()
    z = torch.empty([1, MAX_SEQ_LEN, num_feats]).uniform_() # random vector
    if torch.cuda.is_available():
        z = z.cuda()
        model['g'].cuda()

    model['g'].eval()
    g_feats, _ = model['g'](z, g_states)
    song_data = g_feats.squeeze().cpu()
    song_data = song_data.detach().numpy()

    # if (ep+1) == num_ep:
    #     midi_data = dataloader.save_data('sample.mid', song_data)
    # else:
    #     midi_data = dataloader.save_data(None, song_data)
    #     print(midi_data[0][:16])
    # -- DEBUG --

    return model, trn_acc

def normalize_data(data, indicator_cols: list=None, eps=0):
    """
    Parameters
    ----------
    data : torch.Tensor
        assumes in dimension (num_samples, num_channels, seq_length)
    indicator_cols : list
        columns for missingness indicators, if provided, do not noramlize these, if None, normalize all
    eps : int, optional
        used when want to let values to be (0,1), by default 0

    Returns
    -------
    tuple
        normalized data, min, max
    """
    data = data.numpy()
    
    if indicator_cols is not None:
        indicators = data[:, indicator_cols, :].copy()
        assert not np.any(np.isnan(indicators)), 'indicator columns contain NaNs'
    
    min, max = np.nanmin(data, axis=(0, 2)), np.nanmax(data, axis=(0, 2))
    min, max = min[None, :, None], max[None, :, None]
    data = (data - min + eps) / (max - min + 2*eps)
    assert np.all((data[~np.isnan(data)] >= 0) & (data[~np.isnan(data)] <= 1)), 'failed to normalize data'
    
    if indicator_cols is not None:
        data[:, indicator_cols, :] = indicators         # if passed, do not normalize these columns
            
    return torch.tensor(data).float(), torch.tensor(min).float(), torch.tensor(max).float()

def reverse_normalize(data, min, max) -> np.ndarray:
    return (data * (max - min) + min)          # to turn back into real scale

def replace_nan_with_mean(data: torch.Tensor):
    mean = data.nanmean(dim=(0, 2))[None, :, None]
    data = torch.where(torch.isnan(data), mean, data)
    assert torch.all(~torch.isnan(data)), 'failed to replace NaNs!'
    return data

def smart_load(load_path):
    if load_path.split(".")[-1] == "pt":
        return torch.load(load_path)
    elif load_path.split(".")[-1] == "npy":
        return np.load(load_path)
    elif load_path.split(".")[-1] == "csv":
        return pd.read_csv(load_path)
    else:
        raise ValueError("Unsupported file type: %s" % load_path.split(".")[0])

def smart_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    else:
        raise ValueError("Unsupported type: %s" % type(data))

def main(args, data, train: bool):
    ''' Training sequence'''
    # dataloader = music_data_utils.MusicDataLoader(DATA_DIR, single_composer=COMPOSER)
    dataloader = TimeSeriesLoader(data, cut_length=args.cut_length, batch_size=args.batch_size)
    num_feats = dataloader.get_num_song_features()

    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    model = {
        'g': Generator(num_feats, use_cuda=train_on_gpu, hidden_units=args.hidden),
        'd': Discriminator(num_feats, use_cuda=train_on_gpu, hidden_units=args.hidden),
    }
    if train:
        if args.use_sgd:
            optimizer = {
                'g': optim.SGD(model['g'].parameters(), lr=args.g_lrn_rate, momentum=0.9),
                'd': optim.SGD(model['d'].parameters(), lr=args.d_lrn_rate, momentum=0.9)
            }
        else:
            optimizer = {
                'g': optim.Adam(model['g'].parameters(), args.g_lrn_rate),
                'd': optim.Adam(model['d'].parameters(), args.d_lrn_rate)
            }

        criterion = {
            'g': nn.MSELoss(reduction='sum') if args.feature_matching else GLoss(),
            'd': DLoss(args.label_smoothing)
        }

        if args.load_g:
            ckpt = torch.load(os.path.join(CKPT_DIR, G_FN))
            model['g'].load_state_dict(ckpt)
            print("Continue training of %s" % os.path.join(CKPT_DIR, G_FN))

        if args.load_d:
            ckpt = torch.load(os.path.join(CKPT_DIR, D_FN))
            model['d'].load_state_dict(ckpt)
            print("Continue training of %s" % os.path.join(CKPT_DIR, D_FN))

        if train_on_gpu:
            model['g'].cuda()
            model['d'].cuda()

        if not args.no_pretraining:
            for ep in range(args.d_pretraining_epochs):
                model, _ = run_epoch(model, optimizer, criterion, dataloader,
                                ep, args.d_pretraining_epochs, freeze_g=True, pretraining=True)

            for ep in range(args.g_pretraining_epochs):
                model, _ = run_epoch(model, optimizer, criterion, dataloader,
                                ep, args.g_pretraining_epochs, freeze_d=True, pretraining=True)

        freeze_d = False
        for ep in range(args.num_epochs):
            # if ep % args.freeze_d_every == 0:
            #     freeze_d = not freeze_d

            model, trn_acc = run_epoch(model, optimizer, criterion, dataloader, ep, args.num_epochs, freeze_d=freeze_d)
            if args.conditional_freezing:
                # conditional freezing
                freeze_d = False
                if trn_acc >= 95.0:
                    freeze_d = True

        if not args.no_save_g:
            torch.save(model['g'].state_dict(), args.save_path.split(".")[0] + "_g.pt")
            print("Saved generator: %s" % args.save_path)

        if not args.no_save_d:
            torch.save(model['d'].state_dict(), args.save_path.split(".")[0] + "_d.pt")
            print("Saved discriminator: %s" % args.save_path)
    else:
        model["g"].use_cuda = False
        model["g"].load_state_dict(torch.load(args.save_path.split(".")[0] + "_g.pt", map_location=torch.device('cpu')))
        model["d"].load_state_dict(torch.load(args.save_path.split(".")[0] + "_d.pt", map_location=torch.device('cpu')))
    
    # NOTE: very memory intensive, very likely need to do it separately on a CPU only machine with lots of memory
    model["g"].eval()

    full_song_data = []
    for _ in tqdm(range(20000), desc="Sampling..."):
        z = torch.empty([1, MAX_SEQ_LEN, num_feats]).uniform_() # random vector
        z = z.cpu()
        g_states = model["g"].init_hidden(1)
        g_feats, g_states = model["g"](z, g_states)
        song_data = g_feats.cpu()
        song_data = song_data.detach().numpy() 
        full_song_data.append(song_data)

    full_song_data = np.concatenate(full_song_data, axis=0)
    full_song_data = np.transpose(full_song_data, (0,2,1))
    print('Generated data shape: ', full_song_data.shape)
    np.save(args.sample_save_path, full_song_data)
    print(f"Saved to {args.sample_save_path}")


if __name__ == "__main__":
    prs = ArgumentParser()
    prs.add_argument('--load_g', action='store_true')
    prs.add_argument('--load_d', action='store_true')
    prs.add_argument('--no_save_g', action='store_true')
    prs.add_argument('--no_save_d', action='store_true')
    prs.add_argument("--hidden", type=int, default=256)

    prs.add_argument('--num_epochs', default=300, type=int)
    prs.add_argument('--batch_size', default=16, type=int)
    prs.add_argument('--g_lrn_rate', default=0.001, type=float)
    prs.add_argument('--d_lrn_rate', default=0.001, type=float)

    prs.add_argument('--no_pretraining', action='store_true')
    prs.add_argument('--g_pretraining_epochs', default=5, type=int)
    prs.add_argument('--d_pretraining_epochs', default=5, type=int)
    # prs.add_argument('--freeze_d_every', default=5, type=int)
    prs.add_argument('--use_sgd', action='store_true')
    prs.add_argument('--conditional_freezing', action='store_true')
    prs.add_argument('--label_smoothing', action='store_true')
    prs.add_argument('--feature_matching', action='store_true')
    
    prs.add_argument("--cut_length", type=int, default=None)        # just in case need to adjust seq_length
    prs.add_argument("--load_path", type=str, required=True)
    prs.add_argument("--save_path", type=str, required=True)
    prs.add_argument("--sample_save_path", type=str, required=True)
    prs.add_argument("--wandb", action="store_true")
    prs.add_argument("--seed", type=int, default=2023)
    
    args = prs.parse_args()
    BATCH_SIZE = args.batch_size
    if args.wandb:
        wandb = wandb.init(project="Tony-results", entity="gen-ehr", name=args.save_path.split("/")[-1], config=args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = smart_load(args.load_path)
    data = replace_nan_with_mean(data)
    data, SAMPLE_MIN, SAMPLE_MAX = normalize_data(data)
    data = data.permute(0,2,1)
    MAX_SEQ_LEN = data.shape[1]
    
    main(args, data, train=True)
    # main(args, data, train=False)
