
# import torchtext
import argparse
import datetime
import logging
import time
from logging.config import dictConfig

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

import models.p_or_t_forcing.cfg as cfg
import models.p_or_t_forcing.opts as opts
from models.p_or_t_forcing.dataset import TimeSeriesDataset
from models.p_or_t_forcing.model import LMGan
from models.p_or_t_forcing.trainer import Trainer
from models.p_or_t_forcing.utils import time_since

# Instantiate parser and parse args
parser = argparse.ArgumentParser(description='train.py')
opts.model_opts(parser)
opts.training_opts(parser)
opts.model_io_opts(parser)
opts.data_opts(parser)
opt = parser.parse_args()

dictConfig(cfg.logging_cfg)
print('Arguments:')
print(opt)

# check cuda
if opt.cuda and not torch.cuda.is_available():
    raise RuntimeError('Cannot train on GPU because cuda is not available')

device = 'cuda' if opt.cuda else 'cpu'
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.enabled = False

dataset = TimeSeriesDataset(opt.data_path)
print(dataset.data.shape)
opt.vocab_size = len(dataset)
opt.device = device
opt.channels = dataset.channels
SAMPLE_MIN, SAMPLE_MAX = dataset.min, dataset.max
lmloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

start_time = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')[:-10]

# Initialize model
model = LMGan(opt)
print('Initialized new models')
model.device = device
model.to(device)


# Configuring training
plot_every = opt.plot_every
print_every = opt.print_every

# Begin!
trainer = Trainer(opt, model)


def main():
    # Keep track of time elapsed and running averages
    start = time.time()
    losses = []
    print_nll_loss_total = 0  # Reset every print_every
    print_g_loss_total = 0
    print_d_loss_total = 0
    plot_loss_total = 0  # Reset every plot_every

    for epoch in range(1, opt.n_epochs + 1):
        for idx, batch in enumerate(lmloader):
            nll_loss, gen_loss, disc_loss = trainer.train(opt, batch.to(device))

            if idx % print_every == 0:
                losses.append(nll_loss)
            # Keep track of loss
            print_nll_loss_total += nll_loss
            if opt.adversarial:
                print_g_loss_total += gen_loss
                print_d_loss_total += disc_loss
                plot_loss_total += nll_loss

            if epoch == 0: continue

            if idx % print_every == 0:
                print_summary = '%s (%d %d%%) nll %.4f generator %.4f discriminator %.4f' % (
                    time_since(start, epoch / opt.n_epochs),
                    epoch,
                    epoch / opt.n_epochs * 100,
                    print_nll_loss_total / print_every,
                    print_g_loss_total / print_every,
                    print_d_loss_total / print_every,
                )
                logging.info(print_summary)
                print_nll_loss_total = 0
                print_g_loss_total = 0
                print_d_loss_total = 0

        torch.save(model, opt.save_path)
    
    logging.info("***** COMPLETE TRAINING *****")
    sample_bsz = 200
    generated_samples = []
    for _ in range(100):        # do 20000 samples
        samples = dataset.data[torch.randperm(len(dataset))[:sample_bsz]]
        bsz_sample = trainer.sample(samples.to(device), sample_bsz)
        generated_samples.append(bsz_sample)
    generated_samples = torch.cat(generated_samples, dim=0)
    generated_samples = generated_samples.permute(0,2,1).cpu().numpy()
    generated_samples = reverse_normalize(generated_samples, SAMPLE_MIN.numpy(), SAMPLE_MAX.numpy())
    np.save(opt.sample_save_path, generated_samples)
    print(f"Generated samples saved to {opt.sample_save_path}")

def reverse_normalize(data, min, max) -> np.ndarray:
    return (data * (max - min) + min)          # to turn back into real scale
    
if __name__ == '__main__':
    main()
