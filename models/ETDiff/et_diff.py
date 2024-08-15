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

import models.ETDiff.utils as utils

logging.basicConfig(filename = None, format=("[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)

class ETDiff(nn.Module):
    def __init__(
        self,
        diffusion_model: nn.Module,
        dataset: Dataset,
        *,
        train_batch_size = 16,
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
        self.channels = diffusion_model.channels

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

        # EMA
        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
        self.ema.to(self.device)

        # step counter state
        self.step = 0

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
        model.load_state_dict(data['diff_model'])

        self.step = data['diff_step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data["ema"])

        if utils.exists(self.accelerator.scaler) and utils.exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    
    # @torch.no_grad()
    # def check_point_irregular(self):
    #     if self.wandb is not None:
    #         self.ema.ema_model.eval()

    #         generated = self.ema.ema_model.sample(self.batch_size)
    #         generated = utils.reverse_normalize(data = generated, min = self.sample_min, max = self.sample_max)
            
    #         rows = 8
    #         cols = 5
    #         fig, axs = plt.subplots(rows, cols, figsize=(cols*11, rows*8))
    #         data_iterator = iter(generated)
    #         x = self.obs_time.cpu().numpy().astype(int)
            
    #         plot_dict = {0: 'Hemoglobin', 1: 'Creatinine', 2: 'Sodium', 3: 'BUN', 8: 'Mortality (1/0)'}
    #         idx_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 8}               # map between plot column indices and feature indices
    #         nan_idx_map = {0: 4, 1: 5, 2: 6, 3: 7}                   # map between feature indices and nan indicator indices
            
    #         for i in range(rows):
    #             timeseries = next(data_iterator)
    #             color_choice = iter(sns.color_palette())
    #             for j in range(cols):
    #                 idx = idx_map[j]
    #                 if idx == 8:
    #                     sns.lineplot(x=x, y=np.round(timeseries[idx]), ax=axs[i, j], linewidth=5, label=plot_dict[idx], color=next(color_choice))
    #                     axs[i, j].set_ylim(-0.3, 1.3)
    #                 else:
    #                     nan_idx = nan_idx_map[idx]
    #                     nonan_indicator = np.round(utils.reverse_to_nonan_indicator(timeseries[nan_idx]))
    #                     data = np.where(nonan_indicator == 1, timeseries[idx], np.nan)
    #                     if np.all(np.isnan(data)):
    #                         next(color_choice)
    #                         continue
    #                     else:
    #                         sns.scatterplot(x=x, y=data, ax=axs[i, j], label=plot_dict[idx], color=next(color_choice), s=400)
    #                 axs[i, j].set_xlabel("Time (Hours)", fontsize=30)
    #                 axs[i, j].set_ylabel("Measurement Value", fontsize=30)
    #                 axs[i, j].set_xlim(0, 1500)
    #                 axs[i, j].legend(fontsize=30)
    #                 axs[i, j].tick_params(axis='both', which='major', labelsize=30)

            
    #         plt.tight_layout()
    #         plt.legend()
    #         self.wandb.log({"diffusion_check_point": self.wandb.Image(plt)})
    #         plt.close()
            
    #         torch.save(torch.tensor(generated).permute(0, 2, 1), f"tracking/diffusion_generated_{self.run_id}.pt")
    #         self.ema.ema_model.train()
    #     self.save_weights()

    @torch.no_grad()
    def check_point_regular(self):
        self.ema.ema_model.eval()

        generated = self.ema.ema_model.sample(self.batch_size)
        generated = utils.reverse_normalize(data = generated.cpu().numpy(), min = self.sample_min, max = self.sample_max)
        
        rows = 8
        cols = 5
        fig, axs = plt.subplots(rows, cols, figsize=(cols*11, rows*8))
        data_iterator = iter(generated)
        x = np.arange(60, 1420, 5)
        
        plot_dict = {0: 'Heart Rate', 2: 'Respiratory Rate', 4: 'SPO2', 6: 'Mean Arterial Pressure', 8: 'Mortality (1/0)'}
        idx_map = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}               # map between plot column indices and feature indices
        nan_idx_map = {0: 1, 1: 3, 2: 5, 3: 7}                   # map between feature indices and nan indicator indices
        
        for i in range(rows):
            timeseries = next(data_iterator)
            color_choice = iter(sns.color_palette())
            for j in range(cols):
                idx = idx_map[j]
                if idx == 8:
                    sns.lineplot(x=x, y=np.round(timeseries[idx]), ax=axs[i, j], linewidth=5, label=plot_dict[idx], color=next(color_choice))
                    axs[i, j].set_ylim(-0.3, 1.3)
                else:
                    nan_idx = nan_idx_map[j]
                    nonan_indicator = np.round(utils.reverse_to_nonan_indicator(timeseries[nan_idx]))
                    data = np.where(nonan_indicator == 1, timeseries[idx], np.nan)
                    if np.all(np.isnan(data)):
                        next(color_choice)
                        continue
                    else:
                        sns.lineplot(x=x, y=data, ax=axs[i, j], label=plot_dict[idx], color=next(color_choice), linewidth=2)
                axs[i, j].set_xlabel("Time (Hours)", fontsize=30)
                axs[i, j].set_ylabel("Measurement Value", fontsize=30)
                axs[i, j].set_xlim(0, 1500)
                axs[i, j].legend(fontsize=30)
                axs[i, j].tick_params(axis='both', which='major', labelsize=30)

        if utils.exists(self.wandb):
            plt.tight_layout()
            plt.legend()
            self.wandb.log({"diffusion_check_point": self.wandb.Image(plt)})
            plt.close()
        
        torch.save(torch.tensor(generated), f"tracking/diffusion_generated_{self.run_id}.pt")
        self.ema.ema_model.train()
        self.save_weights()

    def save_weights(self):
        weights = {
            "diff_step": self.step,
            "diff_model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if utils.exists(self.accelerator.scaler) else None,
        }
        torch.save(weights, self.check_point_path)

    def train(self):
        accelerator = self.accelerator
        last_loss_log_time = self.timer()

        logging.info("********** START TRAINING DIFFUSION MODEL **********")
        
        with tqdm(initial = self.step, total = self.diff_num_steps, disable = not accelerator.is_main_process) as pbar:
            
            while self.step < self.diff_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    
                    with self.accelerator.autocast():
                        loss = self.model(data)
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
                        # self.check_point_regular()              
                        # self.check_point_irregular()
                        self.save_weights()
                
                pbar.update(1)

        self.save_weights()
        accelerator.print('========================= TRAINING COMPLETE =========================')