import argparse
import pickle

import torch

import models.p_or_t_forcing.cfg as cfg
import models.p_or_t_forcing.opts as opts
from models.p_or_t_forcing.dataset import LMDataset

multinomial = cfg.sample_methods.multinomial


class Sampler:

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def sample(self, batch_size=1, input=None, start_hidden=None,
               strategy=multinomial, temperature=3.0, n_sampled=100):
        """
        Main sampling function.
        :param batch_size:
        :param start_hidden:
        :return:
        """
        self.model.eval()
        start_hidden = start_hidden or self.model.init_hidden(batch_size)

        # run model and get hidden states and sampled indices
        input = input or self.input(batch_size)
        _, hidden_states, words_indices = self.model.consume(
            input, start_hidden, sampling=True, method=strategy,
            temperature=temperature, n_sampled=n_sampled)

        words_indices = torch.stack(words_indices[1:], dim=1).cpu()

        for idx in range(batch_size):
            tokens = [self.dataset.vocab.itos(idx[0].item()) for idx in words_indices[idx, :]]
            print(''.join(tokens))

    def input(self, batch_size):
        l = [0] * len(self.dataset.vocab)
        l[20] = 1
        return torch.LongTensor(l).to(device).repeat((batch_size, 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sampler.py')
    opts.model_opts(parser)
    opts.model_io_opts(parser)
    opts.data_opts(parser)
    opts.sample_opts(parser)
    opt = parser.parse_args()

    if opt.cuda and not torch.cuda.is_available():
        raise RuntimeError('Cannot sample on GPU because cuda is not available')

    device = 'cuda' if opt.cuda else 'cpu'
    model = torch.load(opt.checkpoint)
    model.device = device
    model.to(device)

    lmdataset = LMDataset(
        vocab_path=opt.vocab_path,
        corpus_path=opt.data_path,
        bptt=opt.length,
        device=device,
        min_counts=0  # TODO: make sure it works
    )
    sampler = Sampler(model, lmdataset)

    sampler.sample(
        opt.batch_size,
        strategy=opt.sampling_strategy,
        temperature=opt.temperature,
        n_sampled=opt.length
    )
