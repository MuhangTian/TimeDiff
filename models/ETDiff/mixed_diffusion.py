from collections import namedtuple
from functools import partial
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
import numpy as np
from torch.cuda.amp import autocast
from tqdm import tqdm

import models.ETDiff.utils as utils

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

@torch.jit.script
def log_sub_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m

@torch.jit.script
def sliced_logsumexp(x, slices):
    lse = torch.logcumsumexp(torch.nn.functional.pad(x, [0, 0, 1, 0], value=-float('inf')), dim=1)

    slice_starts = slices[:-1]
    slice_ends = slices[1:]
    diff = slice_ends - slice_starts
    
    slice_lse = log_sub_exp(lse[:, slice_ends], lse[:, slice_starts])
    slice_lse_repeated = torch.repeat_interleave(slice_lse, diff, dim=1)
    return slice_lse_repeated

def index_to_log_onehot(x, num_classes):
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i, :], num_classes[i]).permute(0, 2, 1))
 
    x_onehot = torch.cat(onehots, dim=1)
    log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_onehot

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def ohe_to_categories(ohe, K):
    "turn one hot encoded (ohe) data into category"
    K = torch.from_numpy(K)
    indices = torch.cat([torch.zeros((1,)), K.cumsum(dim=0)], dim=0).int().tolist()
    res = []
    for i in range(len(indices) - 1):
        res.append(ohe[:, indices[i]:indices[i+1]].argmax(dim=1))
    return torch.stack(res, dim=1)

class MixedDiffusion(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        seq_length: int, 
        channels: int, 
        numerical_features_indices: list,
        categorical_features_indices: list,
        categorical_num_classes: list,
        timesteps: int = 1000, 
        beta_schedule: str = 'cosine', 
        auto_normalize: bool = True,
        loss_lambda: float = 0.5,
        parametrization: str = 'x0',
        ):
        super().__init__()
        
        self.model = model
        self.channels = channels
        self.self_condition = self.model.self_condition
        self.seq_length = seq_length
        self.loss_lambda = loss_lambda
        self.numerical_features_indices = numerical_features_indices
        self.num_numerical_features = len(numerical_features_indices)
        self.categorical_features_indices = categorical_features_indices        
        self.categorical_num_classes = np.asarray(categorical_num_classes)
        self.parametrization = parametrization

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

        # for calculations of forward and reverse diffusion process
        log_alpha = torch.log(alphas)
        log_cumprod_alpha = torch.cumsum(log_alpha, dim=0)
        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)
        categorical_num_classes_expanded = torch.from_numpy(
            np.concatenate([self.categorical_num_classes[i].repeat(self.categorical_num_classes[i]) for i in range(len(self.categorical_num_classes))])
        )
        categorical_num_classes_expanded = categorical_num_classes_expanded.unsqueeze(1).expand(-1, seq_length)
        self.slices_for_classes = [np.arange(self.categorical_num_classes[0])]
        offsets = np.cumsum(self.categorical_num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        offsets = torch.from_numpy(np.append([0], offsets))
        
        register_buffer('log_alpha', log_alpha)         # for multinomial diffusion
        register_buffer('log_1_min_alpha', log_1_min_alpha)
        register_buffer('log_cumprod_alpha', log_cumprod_alpha)
        register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha)
        register_buffer('Lt_history', torch.zeros(timesteps))
        register_buffer('Lt_count', torch.zeros(timesteps))
        register_buffer('categorical_num_classes_expanded', categorical_num_classes_expanded)
        self.register_buffer('offsets', offsets.long())
        
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))      # for gaussian diffusion
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for variance of q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)             # \beta_t * (1 - \bar{\alpha}_{t-1}) / 1 - \bar{\alpha}_t
        register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # loss weight, use ones here, but could be modified
        snr = alphas_cumprod / (1 - alphas_cumprod)
        register_buffer('loss_weight', torch.ones_like(snr))

        # whether to normalize to [-1, 1], this noramlization assumes input is in [0, 1] (already normalized)
        self.normalize = utils.normalize_to_neg_one_to_one if auto_normalize else utils.identity
        self.unnormalize = utils.unnormalize_to_zero_to_one if auto_normalize else utils.identity
        
    def extract_features(self, x, mode):
        """extract features, assuming x is in dimension of (batch_size, features, seq_length)"""
        if mode == 'numerical':
            return x[:, self.numerical_features_indices, :]
        elif mode == 'categorical':
            return x[:, self.categorical_features_indices, :]
        else:
            raise ValueError(f'unknown mode {mode}')
        
    def create_samples(self, numerical_samples, categorical_samples):
        """create samples from numerical and categorical samples"""
        assert numerical_samples.shape[0] == categorical_samples.shape[0], "numerical and categorical samples must have the same batch size"
        samples = torch.zeros((numerical_samples.shape[0], self.channels - len(self.categorical_features_indices), self.seq_length))
        samples[:, self.numerical_features_indices, :] = numerical_samples
        samples[:, self.categorical_features_indices, :] = categorical_samples
        return samples
    
    def extract_modeloutput(self, x, mode):
        if mode == 'numerical':
            return x[:, :len(self.numerical_features_indices), :]
        elif mode == 'categorical':
            return x[:, len(self.numerical_features_indices):, :]
        else:
            raise ValueError(f'unknown mode {mode}')
        
    def normalize_numericals(self, x):
        """normalize numerical features only"""
        x_num = self.extract_features(x, "numerical")
        x_num = self.normalize(x_num)
        x[:, self.numerical_features_indices, :] = x_num
        return x

    def gauss_predict_start_from_noise(self, x_t, t, noise):
        """Returns x_0 using noisy sample x_t and \epsilon"""
        return (
            utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def gauss_predict_noise_from_start(self, x_t, t, x0):
        """Re-derive \epsilon from x_t and x_0, often not used, but could be an option"""
        return (
            (utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def gauss_q_posterior(self, x_start, x_t, t):
        """obtain mean and variance for posterior"""
        posterior_mean = (      # \tilde{\mu}
            utils.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            utils.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = utils.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = utils.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gauss_model_predictions(self, model_output, x, t, x_self_cond = None, clip_x_start = False):
        """Obtain model predictions and get x_0"""
        maybe_clip = partial(torch.clamp, min = -1, max = 1) if clip_x_start else utils.identity

        pred_noise = model_output
        x_start = self.gauss_predict_start_from_noise(x, t, pred_noise)       # x_0
        x_start = maybe_clip(x_start)

        return ModelPrediction(pred_noise, x_start)

    def gauss_p_mean_variance(self, model_output, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.gauss_model_predictions(model_output, x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1, 1)

        model_mean, posterior_variance, posterior_log_variance = self.gauss_q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
        
    @autocast(enabled = False)
    def gauss_q_sample(self, x_start, t, noise=None):
        """obtain noisy sample (forward diffusion process)"""
        noise = utils.default(noise, lambda: torch.randn_like(x_start))

        return (
            utils.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    # ------------------- multinomial diffusion ------------------- #
    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl
    
    def cat_q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = utils.extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = utils.extract(self.log_1_min_alpha, t, log_x_t.shape)
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.categorical_num_classes_expanded)
        )
        return log_probs
    
    def cat_q_pred(self, log_x_start, t):
        """forward multinomial diffusion process: create noisy sample, q(x_t | x_0)"""
        log_cumprod_alpha_t = utils.extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = utils.extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.categorical_num_classes_expanded)
        )
        return log_probs

    def cat_predict_start(self, model_out, log_x_t, t, out_dict):
        assert model_out.size(0) == log_x_t.size(0)
        assert model_out.size(1) == self.categorical_num_classes.sum(), f'{model_out.size()}'

        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred
    
    def cat_q_posterior(self, log_x_start, log_x_t, t):
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.cat_q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0.to(torch.float32))
        unnormed_logprobs = log_EV_qxtmin_x0 + self.cat_q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = unnormed_logprobs - sliced_logsumexp(unnormed_logprobs, self.offsets)

        return log_EV_xtmin_given_xt_given_xstart
    
    def cat_p_pred(self, model_out, log_x, t, out_dict):
        if self.parametrization == 'x0':
            log_x_recon = self.cat_predict_start(model_out, log_x, t=t, out_dict=out_dict)
            log_model_pred = self.cat_q_posterior(log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.cat_predict_start(model_out, log_x, t=t, out_dict=out_dict)
        else:
            raise ValueError("Unknown parametrization: {}".format(self.parametrization))
        return log_model_pred
    
    @torch.no_grad()
    def cat_p_sample(self, model_out, log_x, t, out_dict):
        model_log_prob = self.cat_p_pred(model_out, log_x=log_x, t=t, out_dict=out_dict)
        out = self.cat_log_sample_categorical(model_log_prob)
        return out
    
    def cat_q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.cat_q_pred(log_x_start, t)
        log_sample = self.cat_log_sample_categorical(log_EV_qxt_x0)
        return log_sample
    
    def cat_log_sample_categorical(self, logits):
        full_sample = []
        for i in range(len(self.categorical_num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_onehot(full_sample, self.categorical_num_classes)
        return log_sample
    
    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.cat_q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.categorical_num_classes_expanded * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)
    
    def compute_Lt(self, model_out, log_x_start, log_x_t, t, out_dict=None, detach_mean=False):
        log_true_prob = self.cat_q_posterior(log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob = self.cat_p_pred(model_out, log_x=log_x_t, t=t, out_dict=out_dict)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl
        return loss
    
    def cat_loss(self, model_out, log_x_start, log_x_t, t, pt, out_dict):
        kl = self.compute_Lt(model_out, log_x_start, log_x_t, t, out_dict)
        kl_prior = self.kl_prior(log_x_start)
        vb_loss = kl / pt + kl_prior
        return vb_loss
    
    def gauss_loss(self, model_out, x_start, x_t, t, noise, model_kwargs=None):
        loss = F.mse_loss(model_out, noise, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * utils.extract(self.loss_weight, t, loss.shape)
        return loss.mean()
    
    def mixed_loss(self, x_start, t, noise = None):
        b = x_start.shape[0]
        device = x_start.device
        pt = torch.ones_like(t).float() / self.num_timesteps

        x_num = self.extract_features(x_start, "numerical")
        x_cat = self.extract_features(x_start, "categorical")
        
        x_num_t = x_num
        log_x_cat_t = x_cat
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gauss_q_sample(x_num, t, noise=noise)
        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.categorical_num_classes)
            log_x_cat_t = self.cat_q_sample(log_x_start=log_x_cat, t=t)
        
        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)
        model_out = self.model(x_in, t, None)               # this is our BRNN

        model_out_num = self.extract_modeloutput(model_out, "numerical")
        model_out_cat = self.extract_modeloutput(model_out, "categorical")

        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()
        if x_cat.shape[1] > 0:
            loss_multi = self.cat_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt, None) / len(self.categorical_num_classes)
        
        if x_num.shape[1] > 0:
            loss_gauss = self.gauss_loss(model_out_num, x_num, x_num_t, t, noise)

        return (self.loss_lambda * loss_multi.mean()) + loss_gauss

    def forward(self, sample, *args, **kwargs):
        b, c, n, device, seq_length, = *sample.shape, sample.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        sample = self.normalize_numericals(sample)
        return self.mixed_loss(sample, t, *args, **kwargs)
    
    # ------------------- sampling ------------------- #
    @torch.no_grad()
    def gauss_p_sample(self, model_output, x, batched_times: torch.Tensor, t, x_self_cond = None, clip_denoised = True):
        """sample from p_{\theta}"""
        b, *_, device = *model_output.shape, model_output.device
        model_mean, _, model_log_variance, x_start = self.gauss_p_mean_variance(
            model_output = model_output, 
            x = x, t = batched_times, 
            x_self_cond = x_self_cond, clip_denoised = clip_denoised
        )
        noise = torch.randn_like(model_output) if t > 0 else 0. # no noise if t == 0
        pred_sample = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_sample, x_start
    
    @torch.no_grad()
    def sample(self, batch_size):
        b = batch_size
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features, self.seq_length), device=device)

        has_cat = self.categorical_num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.categorical_num_classes_expanded), self.seq_length), device=device)
            log_z = self.cat_log_sample_categorical(uniform_logits)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="Sampling loop time step", total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = torch.cat([z_norm, log_z], dim=1).float()
            model_out = self.model(x, t, None)
            model_out_num = self.extract_modeloutput(model_out, "numerical")
            model_out_cat = self.extract_modeloutput(model_out, "categorical")
            
            z_norm, _ = self.gauss_p_sample(model_out_num, z_norm, t, i, clip_denoised=True)
            if has_cat:
                log_z = self.cat_p_sample(model_out_cat, log_z, t, None)

        z_ohe = torch.exp(log_z).round()
        z_norm = self.unnormalize(z_norm)
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.categorical_num_classes)
        sample = self.create_samples(numerical_samples = z_norm.cpu(), categorical_samples = z_cat.cpu().float())
        return sample
