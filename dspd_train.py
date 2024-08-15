import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import wandb
from models.tsdiff.synthetic.data import DataModule
from models.tsdiff.synthetic.diffusion_model import DiffusionModule
from models.tsdiff.synthetic.nf_model import NFModule
from models.tsdiff.synthetic.ode_model import ODEModule
from models.tsdiff.synthetic.sde_model import SDEModule
from helpers.utils import reverse_normalize

warnings.simplefilter(action='ignore', category=(np.VisibleDeprecationWarning))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
    *,
    sample_save_path: str,
    load_path: str,
    seed: int,
    diffusion: str,
    model: str,
    gp_sigma: float = None,
    ou_theta: float = None,
    beta_start: float = None,
    beta_end: float = None,
    batch_size: int = 256,
    hidden_dim: int = 128,
    predict_gaussian_noise: bool = True,
    beta_fn: str = 'linear',
    discrete_num_steps: int = 100,
    continuous_t1: float = 1,
    loss_weighting: str = 'exponential',
    learning_rate: float = 1e-3,
    weight_decay: float = 0,
    epochs: int = 100,
    patience: int = 20,
    wandb: bool = False,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    datamodule = DataModule(load_path=load_path, batch_size=batch_size, include_time=False)

    if diffusion is not None:
        if 'Continuous' in diffusion:
            beta_start, beta_end = 0.1, 20
        else:
            beta_start, beta_end = 1e-4, 20 / discrete_num_steps

    if model == 'ode':
        Module = ODEModule
    elif model == 'nf':
        Module = NFModule
    elif model == 'sde':
        Module = SDEModule
    else:
        Module = DiffusionModule

    # Load model
    module = Module(
        dim=datamodule.dim,
        data_mean=datamodule.x_mean,
        data_std=datamodule.x_std,
        max_t=datamodule.t_max,
        diffusion=diffusion,
        model=model,
        predict_gaussian_noise=predict_gaussian_noise,
        gp_sigma=gp_sigma,
        ou_theta=ou_theta,
        beta_fn=beta_fn,
        discrete_num_steps=discrete_num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        continuous_t1=continuous_t1,
        loss_weighting=loss_weighting,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # Train
    checkpointing = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best-checkpoint')
    # early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
    trainer = Trainer(
        # gpus=1,
        # auto_select_gpus=True,
        max_epochs=epochs,
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[checkpointing],
    )

    trainer.fit(module, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())

    # Load best model
    module = Module.load_from_checkpoint(checkpointing.best_model_path)

    t = datamodule.trainset[:1000][1].to(device)        # NOTE: assuming time is regular intervaled
    samples_arr = []
    for _ in range(20):
        samples = module.sample(t=t, use_ode=True)
        samples_arr.append(samples.detach().cpu().numpy())
    samples = np.concatenate(samples_arr, axis=0)
    samples = np.transpose(samples, (0,2,1))            # back to original shape
    samples = reverse_normalize(samples, min = datamodule.min.numpy(), max = datamodule.max.numpy())
    np.save(sample_save_path, samples)
    print(f"Saved to {sample_save_path}")

    print("=== COMPLETE ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train forecasting model.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--diffusion', type=str, choices=[
        'GaussianDiffusion', 'OUDiffusion', 'GPDiffusion',
        'ContinuousGaussianDiffusion', 'ContinuousOUDiffusion', 'ContinuousGPDiffusion',
    ])
    parser.add_argument('--model', type=str, choices=['feedforward', 'rnn', 'cnn', 'ode', 'transformer'])
    parser.add_argument('--gp_sigma', type=float, default=0.1)
    parser.add_argument('--ou_theta', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument("--sample_save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--discrete_num_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    
    if args.wandb:
         wandb.init(project="Tony-DSPD", entity="gen-ehr", name=args.sample_save_path.split("/")[2], config=args)

    train(**args.__dict__)