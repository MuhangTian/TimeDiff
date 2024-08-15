from collections import namedtuple
from functools import partial
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from torch.cuda.amp import autocast
from tqdm import tqdm

import models.latent_diffusion.utils as utils

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class GaussianDiffusion(nn.Module):
    def __init__(self, model, seq_length, timesteps = 1000, beta_schedule = 'cosine', auto_normalize = True):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.seq_length = seq_length

        if beta_schedule == 'linear':
            betas = utils.linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = utils.cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)
        register_buffer('loss_weight', torch.ones_like(snr))

        # whether to autonormalize, assume input is in [0, 1]
        self.normalize = utils.normalize_to_neg_one_to_one if auto_normalize else utils.identity
        self.unnormalize = utils.unnormalize_to_zero_to_one if auto_normalize else utils.identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            utils.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            utils.extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            utils.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            utils.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = utils.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = utils.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1, max = 1) if clip_x_start else utils.identity

        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)       # x_0
        x_start = maybe_clip(x_start)

        if clip_x_start and rederive_pred_noise:
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1, 1)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_sample = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_sample, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, clip_denoised = True):
        batch, device = shape[0], self.betas.device

        sample = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            sample, x_start = self.p_sample(sample, t, self_cond, clip_denoised = clip_denoised)

        sample = self.unnormalize(sample)
        return sample

    @torch.no_grad()
    def sample(self, batch_size = 16):
        seq_length, channels = self.seq_length, self.channels
        sample = self.p_sample_loop((batch_size, channels, seq_length))
        return sample

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = utils.default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        sample = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            sample, x_start = self.p_sample(sample, i, self_cond)

        return sample

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = utils.default(noise, lambda: torch.randn_like(x_start))

        return (
            utils.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        b, c, n = x_start.shape
        noise = utils.default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times        # DEBUG:
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)
        loss = F.mse_loss(model_out, noise, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * utils.extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, sample, *args, **kwargs):
        b, c, n, device, seq_length, = *sample.shape, sample.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        sample = self.normalize(sample)
        return self.p_losses(sample, t, *args, **kwargs)