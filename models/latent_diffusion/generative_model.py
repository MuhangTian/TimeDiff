import logging
import timeit
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from ema_pytorch import EMA
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import models.latent_diffusion.utils as utils
from models.latent_diffusion.gaussian_diffusion import GaussianDiffusion

logging.basicConfig(filename = None, format=("[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)

class LatentDiffusion:
    def __init__(
        self,
        diffusion_model: GaussianDiffusion,
        encoder: nn.Module,
        decoder: nn.Module,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        autoencoder_num_steps = 50000,
        gradient_accumulate_every = 1,
        diff_lr = 1e-4,
        diff_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        sample_every = 1000,
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        wandb = None,
        check_point_path = "results/checkpoint.pt",
        run_id = 2023,
    ):
        super().__init__()
        
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )
        
        # timing and logging
        self.timer = timeit.default_timer
        self.wandb = wandb
        self.check_point_path = check_point_path
        self.sample_min = dataset.min.numpy()
        self.sample_max = dataset.max.numpy()
        
        # model
        device = self.accelerator.device
        print(f"*** Using device: {device} ***")
        self.model = diffusion_model
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.channels = diffusion_model.channels
        self.autoencoder_num_steps = autoencoder_num_steps

        # sampling and training hyperparameters
        self.sample_every = sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.diff_num_steps = diff_num_steps
        self.run_id = run_id

        # dataset and dataloader
        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 1)

        dl = self.accelerator.prepare(dl)
        self.dl = utils.cycle(dl)           # to make the dataset infinitely long, useful for iterations

        # optimizer
        self.opt = torch.optim.Adam(diffusion_model.parameters(), lr = diff_lr, betas = adam_betas)
        self.ae_opt = torch.optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()))

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        # step counter state
        self.step = 0
        self.ae_step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def load(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(path, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if utils.exists(self.accelerator.scaler) and utils.exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    
    def check_point(self):
        if self.wandb is not None:
            self.ema.ema_model.eval()

            with torch.no_grad():
                latent_vecs = self.ema.ema_model.sample(self.batch_size)
                latent_vecs = latent_vecs.permute(0, 2, 1)
                generated = self.decoder(latent_vecs, self.decoder_times).permute(0, 2, 1).detach().cpu().numpy()
                generated = utils.reverse_normalize(data = generated, min = self.sample_min, max = self.sample_max)
                
                rows = 8
                cols = 5
                fig, axs = plt.subplots(rows, cols, figsize=(cols*11, rows*8))
                data_iterator = iter(generated)
                x = self.obs_time.cpu().numpy().astype(int)
                
                plot_dict = {0: 'Hemoglobin', 1: 'Creatinine', 2: 'Sodium', 3: 'BUN', 8: 'Mortality (1/0)'}
                idx_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 8}               # map between plot column indices and feature indices
                nan_idx_map = {0: 4, 1: 5, 2: 6, 3: 7}                   # map between feature indices and nan indicator indices
                
                for i in range(rows):
                    timeseries = next(data_iterator)
                    color_choice = iter(sns.color_palette())
                    for j in range(cols):
                        idx = idx_map[j]
                        if idx == 8:
                            sns.lineplot(x=x, y=np.round(timeseries[idx]), ax=axs[i, j], linewidth=5, label=plot_dict[idx], color=next(color_choice))
                            axs[i, j].set_ylim(-0.3, 1.3)
                        else:
                            nan_idx = nan_idx_map[idx]
                            nonan_indicator = np.round(utils.reverse_to_nonan_indicator(timeseries[nan_idx]))
                            data = np.where(nonan_indicator == 1, timeseries[idx], np.nan)
                            if np.all(np.isnan(data)):
                                next(color_choice)
                                continue
                            else:
                                sns.scatterplot(x=x, y=data, ax=axs[i, j], label=plot_dict[idx], color=next(color_choice), s=400)
                        axs[i, j].set_xlabel("Time (Hours)", fontsize=30)
                        axs[i, j].set_ylabel("Measurement Value", fontsize=30)
                        axs[i, j].set_xlim(0, 1500)
                        axs[i, j].legend(fontsize=30)
                        axs[i, j].tick_params(axis='both', which='major', labelsize=30)

                
                plt.tight_layout()
                plt.legend()
                self.wandb.log({"diffusion_check_point": self.wandb.Image(plt)})
                plt.close()
                
                torch.save(torch.tensor(generated).permute(0, 2, 1), f"tracking/diffusion_generated_{self.run_id}.pt")
                self.ema.ema_model.train()
        self.save_weights()

    def save_weights(self):
        weights = {
            "diff_step": self.step,
            "diff_model": self.accelerator.get_state_dict(self.model),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "ae_opt": self.ae_opt.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if utils.exists(self.accelerator.scaler) else None,
        }
        torch.save(weights, self.check_point_path)
    
    def check_point_autoencoder(self, generated):
        with torch.no_grad():
            rows = 8
            cols = 5
            fig, axs = plt.subplots(rows, cols, figsize=(cols*11, rows*8))
            
            generated = generated.permute(0, 2, 1).detach().cpu().numpy()
            generated = utils.reverse_normalize(data = generated, min = self.sample_min, max = self.sample_max)
            data_iterator = iter(generated)
            x = self.obs_time.cpu().numpy().astype(int)
            
            plot_dict = {0: 'Hemoglobin', 1: 'Creatinine', 2: 'Sodium', 3: 'BUN', 8: 'Mortality (1/0)'}
            idx_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 8}               # map between plot column indices and feature indices
            nan_idx_map = {0: 4, 1: 5, 2: 6, 3: 7}                   # map between feature indices and nan indicator indices
            
            for i in range(rows):
                timeseries = next(data_iterator)
                color_choice = iter(sns.color_palette())
                for j in range(cols):
                    idx = idx_map[j]
                    if idx == 8:
                        sns.lineplot(x=x, y=np.round(timeseries[idx]), ax=axs[i, j], linewidth=5, label=plot_dict[idx], color=next(color_choice))
                        axs[i, j].set_ylim(-0.3, 1.3)
                    else:
                        nan_idx = nan_idx_map[idx]
                        nonan_indicator = np.round(timeseries[nan_idx])
                        nonan_indicator = utils.reverse_to_nonan_indicator(nonan_indicator)
                        data = np.where(nonan_indicator == 1, timeseries[idx], np.nan)
                        if np.all(np.isnan(data)):
                            next(color_choice)
                            continue
                        else:
                            sns.scatterplot(x=x, y=data, ax=axs[i, j], linewidth=2, label=plot_dict[idx], color=next(color_choice), s=400)
                    axs[i, j].set_xlabel("Time (Hours)", fontsize=30)
                    axs[i, j].set_ylabel("Measurement Value", fontsize=30)
                    axs[i, j].set_xlim(0, 1500)
                    axs[i, j].legend(fontsize=30)
                    axs[i, j].tick_params(axis='both', which='major', labelsize=30)
            
            plt.tight_layout()
            plt.legend()
            self.wandb.log({"autoencoder_real_data": self.wandb.Image(plt)})
            plt.close()
            
            torch.save(torch.tensor(generated).permute(0, 2, 1), f"tracking/autoencoder_generated_{self.run_id}.pt")
            self.save_weights()

    def train(self):
        logging.info("********** START TRAINING AUTOENCODER **********")
        self.encoder.train()
        self.decoder.train()
        
        accelerator = self.accelerator
        last_loss_log_time = self.timer()
        self.obs_time, self.decoder_times = None, None
        
        with tqdm(initial = self.ae_step, total = self.autoencoder_num_steps) as pbar:
        
            while self.ae_step < self.autoencoder_num_steps:
                data, coeffs = next(self.dl)

                if self.decoder_times is None:
                    self.decoder_times = data[:,:,0]                             # all the same, so just keep track of one is enough
                if self.obs_time is None:
                    self.obs_time = data[0,:,0].clone()                          # first dimension is time, select one is enough since they are all the same, and keep track
                data = data[:,:,1:]                                              # exclude time

                latent_vectors = self.encoder(self.obs_time, coeffs)
                generated = self.decoder(latent_vectors, self.decoder_times)           # indices serves the purpose for decoder

                nonan_generated, nonan_data = generated[~torch.isnan(data)], data[~torch.isnan(data)]       # still makes sense, since as long as values and NaN indicators are correct, series is correct
                loss = torch.sqrt(F.mse_loss(nonan_generated, nonan_data)) * 10
                pbar.set_description(f"reconstruction loss: {loss.item():.4f}")

                if self.timer() - last_loss_log_time > 60 and self.wandb is not None:               # log every 1 minute to reduce overhead
                    self.wandb.log({"autoencoder/reconstruction_loss": loss.item()})
                    last_loss_log_time = self.timer()
                
                if self.ae_step != 0 and self.ae_step % 100 == 0:
                    self.check_point_autoencoder(generated)

                self.ae_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(chain(self.encoder.parameters(), self.decoder.parameters()), 1)
                self.ae_opt.step()
                
                self.ae_step += 1
                pbar.update(1)

        logging.info("********** START TRAINING LATENT DIFFUSION MODEL **********")
        self.decoder.eval()
        self.encoder.eval()
        
        with tqdm(initial = self.step, total = self.diff_num_steps, disable = not accelerator.is_main_process) as pbar:
            
            while self.step < self.diff_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data, coeffs = next(self.dl)
                    
                    with torch.no_grad():
                        latent_vectors = self.encoder(self.obs_time, coeffs).permute(0, 2, 1)                     # latent vectors has dimension (batch_size, length, channels), permute it to (batch_size, channels, length)

                    with self.accelerator.autocast():
                        loss = self.model(latent_vectors)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                if self.wandb is not None and self.timer() - last_loss_log_time > 60:
                    self.wandb.log({"total_loss": total_loss})
                    last_loss_log_time = self.timer()

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.sample_every == 0:
                        self.check_point()                  
                
                pbar.update(1)

        self.check_point()
        accelerator.print('========================= TRAINING COMPLETE =========================')