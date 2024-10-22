# Copyright (c) 2019-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path as osp

import models.gt_gan.lib.layers as layers
import numpy as np
import torch
import torch.nn.functional as F
from models.gt_gan.lib.utils import sample_standard_gaussian
import math
from models.gt_gan.train_misc import set_cnf_options

SOLVERS = ['euler','Euler', 'RK2', 'RK4', 'RK23', 'Sym12Async', 'RK12','Dopri5','dopri5','rk4','sym12async','adalf', 'fixedstep_sym12async','fixedstep_adalf']
def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2
def parse_arguments():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser("Continuous Time Flow Process")
    # parser.add_argument("--data_path", type=str, default="data/gbm_2.pkl")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dims", type=str, default="32-64-64-32")
    parser.add_argument(
        "--aug_hidden_dims",
        type=str,
        default=None,
        help="The hiddden dimension of the odenet taking care of augmented dimensions",
    )
    parser.add_argument(
        "--aug_dim",
        type=int,
        default=0,
        help="The dimension along which input is augmented. 0 for 1-d input",
    )
    parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
    parser.add_argument(
        "--num_blocks", type=int, default=1, help="Number of stacked CNFs."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="ode_rnn",
        choices=["ode_rnn", "rnn", "np", "attentive_np"],
    )

    parser.add_argument("--conv", type=eval, default=False, choices=[True, False])
    parser.add_argument(
        "--layer_type",
        type=str,
        default="concat",
        choices=[
            "ignore",
            "concat",
            "concat_v2",
            "squash",
            "concatsquash",
            "concatcoord",
            "hyper",
            "blend",
        ],
    )
    parser.add_argument(
        "--divergence_fn",
        type=str,
        default="approximate",
        choices=["brute_force", "approximate"],
    )
    parser.add_argument(
        "--nonlinearity",
        type=str,
        default="softplus",
        choices=["tanh", "relu", "softplus", "elu", "swish","sigmoid"],
    )
    parser.add_argument("--solver", type=str, default='sym12async', choices=SOLVERS)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    #parser.add_argument("--solver", type=str, default='rk4', choices=SOLVERS)
    #parser.add_argument("--atol", type=float, default=1e-5)
    #parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--step_size", type=float, default=0.1, help="Optional fixed step size."
    )
    parser.add_argument('--first_step', type=float, default=0.166667, help='only for adaptive solvers')
    parser.add_argument(
        "--test_solver", type=str, default=None, choices=SOLVERS + [None]
    )
    parser.add_argument('--test_step_size', type=float, default=None)
    parser.add_argument("--test_atol", type=float, default=0.1)
    parser.add_argument("--test_rtol", type=float, default=0.1)
    parser.add_argument('--test_first_step', type=float, default=None)
    parser.add_argument("--input_size", type=int, default=24)
    parser.add_argument("--aug_size", type=int, default=0, help="size of time")
    parser.add_argument(
        "--latent_size", type=int, default=10, help="size of latent dimension"
    )
    parser.add_argument(
        "--rec_size", type=int, default=20, help="size of the recognition network"
    )
    parser.add_argument(
        "--rec_layers",
        type=int,
        default=1,
        help="number of layers in recognition network(ODE)",
    )
    parser.add_argument(
        "-u",
        "--units",
        type=int,
        default=100,
        help="Number of units per layer in encoder ODE func",
    )
    parser.add_argument(
        "-g",
        "--gru-units",
        type=int,
        default=100,
        help="Number of units per layer in each of GRU update networks in encoder",
    )
    parser.add_argument(
        "-n",
        "--num_iwae_samples",
        type=int,
        default=1,
        help="Number of samples to train IWAE encoder",
    )
    parser.add_argument(
        "--niwae_test", type=int, default=25, help="Numver of IWAE samples during test"
    )
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--time_length", type=float, default=1.0)
    parser.add_argument("--train_T", type=eval, default=True)
    parser.add_argument("--aug_mapping", action="store_true")
    parser.add_argument(
        "--activation", type=str, default="exp", choices=["exp", "softplus", "identity"]
    )

    parser.add_argument("--num_epochs", type=int, default=1000)
    # parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--test_batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument(
        "--amsgrad", action="store_true", help="use amsgrad for adam optimizer"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum value for sgd optimizer"
    )

    parser.add_argument("--decoder_frequency", type=int, default=3)
    parser.add_argument("--aggressive", action="store_true")

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
    parser.add_argument("--residual", type=eval, default=False, choices=[True, False])
    parser.add_argument("--autoencode", type=eval, default=False, choices=[True, False])
    parser.add_argument("--rademacher", type=eval, default=True, choices=[True, False])
    parser.add_argument("--multiscale", type=eval, default=False, choices=[True, False])
    parser.add_argument("--parallel", type=eval, default=False, choices=[True, False])

    # Regularizations
    parser.add_argument('--reconstruction', type=float, default=0.01, help="|| x - decode(encode(x)) ||")
    parser.add_argument('--kinetic-energy', type=float, default=0.05, help="int_t ||f||_2^2")
    parser.add_argument('--jacobian-norm2', type=float, default=0.01, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total-deriv', type=float, default=None, help="int_t ||df/dt||^2")
    parser.add_argument('--directional-penalty', type=float, default=0.01, help="int_t ||(df/dx)^T f||^2")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1e10,
        help="Max norm of graidents (default is just stupidly high to avoid any clipping)",
    )

    parser.add_argument("--begin_epoch", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save", type=str, default="ctfp")
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument(
        "--no_tb_log", action="store_true", help="Do not use tensorboard logging"
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        choices=["train", "test", "val"],
        help="The split of dataset to evaluate the model on",
    )
    # args = parser.parse_args()
    # args.save = osp.join("experiments", args.save)

    # args.effective_shape = args.input_size
    return parser


def build_augmented_model_tabular(args, dims, regularization_fns=None):
    """
    The function used for creating conditional Continuous Normlizing Flow
    with augmented neural ODE

    Parameters:
        args: arguments used to create conditional CNF. Check args parser for details.
        dims: dimension of the input. Currently only allow 1-d input.
        regularization_fns: regularizations applied to the ODE function

    Returns:
        a ctfp model based on augmened neural ode
    """
    # import pdb
    # pdb.set_trace()
    hidden_dims = tuple(map(int, args.dims.split(",")))
    if args.aug_hidden_dims is not None:
        aug_hidden_dims = tuple(map(int, args.aug_hidden_dims.split(",")))
    else:
        aug_hidden_dims = None

    def build_cnf():

        diffeq = layers.ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            #effective_shape=args.effective_shape,# no in original
            strides=None,
            conv=False,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
            #aug_dim=args.aug_dim, # no in original
            #aug_mapping=args.aug_mapping,# no in original
            #aug_hidden_dims=args.aug_hidden_dims,# no in original
        )
        odefunc = layers.ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
            #effective_shape=args.effective_shape, # no in original
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
            rtol=args.rtol,
            atol=args.atol,
        )
        return cnf
    #chain = [layers.LogitTransform(alpha=args.alpha)] if args.alpha > 0 else [layers.ZeroMeanTransform()]
    #chain = chain + [build_cnf() for _ in range(args.num_blocks)]
    chain = [build_cnf() for _ in range(args.num_blocks)]
    model = layers.SequentialFlow(chain)
    print('build cnf')
    return model

def build_augmented_model_tabular2(args, dims, regularization_fns=None):
    """
    The function used for creating conditional Continuous Normlizing Flow
    with augmented neural ODE

    Parameters:
        args: arguments used to create conditional CNF. Check args parser for details.
        dims: dimension of the input. Currently only allow 1-d input.
        regularization_fns: regularizations applied to the ODE function

    Returns:
        a ctfp model based on augmened neural ode
    """
    # import pdb
    # pdb.set_trace()
    hidden_dims = tuple(map(int, args.dims.split(",")))
    if args.aug_hidden_dims is not None:
        aug_hidden_dims = tuple(map(int, args.aug_hidden_dims.split(",")))
    else:
        aug_hidden_dims = None

    def build_cnf():
        diffeq = layers.AugODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            effective_shape=args.effective_shape,# no in original
            strides=None,
            conv=False,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
            aug_dim=args.aug_dim, # no in original
            aug_mapping=args.aug_mapping,# no in original
            aug_hidden_dims=args.aug_hidden_dims,# no in original
        )
        odefunc = layers.AugODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
            effective_shape=args.effective_shape, # no in original
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
            rtol=args.rtol,
            atol=args.atol,
        )
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]
    if args.batch_norm:
        bn_layers = [
            layers.MovingBatchNorm1d(
                dims, bn_lag=args.bn_lag, effective_shape=args.effective_shape
            )
            for _ in range(args.num_blocks)
        ]
        bn_chain = [
            layers.MovingBatchNorm1d(
                dims, bn_lag=args.bn_lag, effective_shape=args.effective_shape
            )
        ]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = layers.SequentialFlow(chain)
    set_cnf_options(args, model)
    print('build cnf')
    return model

def log_jaco(values, reverse=False):
    """
    compute log transformation and log determinant of jacobian

    Parameters:
        values: tensor to be transformed
        reverse (bool): If reverse is False, given z_1 return z_0 = log(z_1) and
                        log det of d z_1/d z_0. If reverse is True, given z_0
                        return z_1 = exp(z_0) and log det of d z_1/d z_0

    Returns:
        transformed tesnors and log determinant of the transformation
    """
    if not reverse:
        log_values = torch.log(values)
        return log_values, torch.sum(log_values, dim=2)
    else:
        return torch.exp(values), torch.sum(values, dim=2)


def inversoft_jaco(values, reverse=False):
    """
    compute softplus  transformation and log determinant of jacobian

    Parameters:
        values: tensor to be transformed
        reverse (bool): If reverse is False, given z_1 return
                        z_0 = inverse_softplus(z_1) and log det of d z_1/d z_0.
                        If reverse is True, given z_0 return z_1 = softplus(z_0)
                        and log det of d z_1/d z_0

    Returns:
        transformed tesnors and log determinant of the transformation
    """
    if not reverse:
        inverse_values = torch.log(1 - torch.exp(-values)) + values
        log_det = torch.sum(
            inverse_values - torch.nn.functional.softplus(inverse_values), dim=2
        )
        return inverse_values, log_det
    else:
        log_det = torch.sum(values - torch.nn.functional.softplus(values), dim=2)
        return torch.nn.functional.softplus(values)


def compute_loss(log_det, base_variables, args):
    """
    This function computes the loss of observations with respect to base wiener
    process.

    Parameters:
        log_det: log determinant of transformation 1-D vectors of size
                 batch_size*length
        base_variables: Tensor after mapping observations back to the space of
                        base Wiener process. 2-D tensor of size batch_size*length
                        x input_shape
        vars: Difference between consequtive observation time stampes.
              2-D tensor of size batch_size*length x input_shape
        masks: Binary tensor showing whether a place is actual observation or
               padded dummy variable. 1-D binary vectors of size
               batch_size*length

    Returns:
        the step-wise mean of observations' negative log likelihood
    """
    mean_martingale = base_variables.clone()
    mean_martingale[:, 1:] = base_variables.clone()[:, :-1]
    mean_martingale[:, 0:1] = 0
    mean_martingale = mean_martingale.view(-1, mean_martingale.shape[2])
    base_variables = base_variables.view(-1, base_variables.shape[2])
    # import pdb
    # pdb.set_trace()
    vars = torch.ones(base_variables.shape[0], base_variables.shape[-1]).to(base_variables)
    masks = torch.ones(base_variables.shape[0])
    # non_zero_idx = masks.nonzero()[:, 0]
    mean_martingale_masked = mean_martingale#[non_zero_idx]
    vars_masked = vars#[non_zero_idx]
    log_det_masked = log_det#[non_zero_idx]
    base_variables_masked = base_variables#[non_zero_idx]
    #num_samples = non_zero_idx.shape[0]
    normal_distri = torch.distributions.Normal(
        mean_martingale_masked, torch.sqrt(vars_masked)
    )
    LL = normal_distri.log_prob(base_variables_masked).view(base_variables_masked.shape[0], -1).sum(1, keepdim=True) - log_det_masked.flatten()
    return -torch.mean(LL)


def compute_ll(log_det, base_variables, vars, masks):
    """
    This function computes the log likelihood of observations with respect to base wiener
    process used for latent_CTFP.

    Parameters:
        log_det: log determinant of transformation 2-D vectors of size
                 batch_size x length
        base_variables: Tensor after mapping observations back to the space of
                        base Wiener process. 3-D tensor of size batch_size x
                        length x input_shape
        vars: Difference between consequtive observation time stampes.
              3-D tensor of size batch_size x length x 1
        masks: Binary tensor showing whether a place is actual observation or
               padded dummy variable. 2-D binary vectors of size
               batch_size x length

    Returns:
        the sum of log likelihood of all observations
    """
    # import pdb;pdb.set_trace()
    mean_martingale = base_variables.clone()
    mean_martingale[:, 1:] = base_variables.clone()[:, :-1]
    mean_martingale[:, 0:1] = 0
    normal_distri = torch.distributions.Normal(mean_martingale, torch.sqrt(vars))
    LL = normal_distri.log_prob(base_variables)
    LL = (torch.sum(LL, -1) - log_det) * masks
    return torch.sum(LL, -1)

def compute_ll2(log_det, base_variables, vars, masks):
    """
    This function computes the log likelihood of observations with respect to base wiener
    process used for latent_CTFP.

    Parameters:
        log_det: log determinant of transformation 2-D vectors of size
                 batch_size x length
        base_variables: Tensor after mapping observations back to the space of
                        base Wiener process. 3-D tensor of size batch_size x
                        length x input_shape
        vars: Difference between consequtive observation time stampes.
              3-D tensor of size batch_size x length x 1
        masks: Binary tensor showing whether a place is actual observation or
               padded dummy variable. 2-D binary vectors of size
               batch_size x length

    Returns:
        the sum of log likelihood of all observations
    """

    #mean_martingale = base_variables.clone()
    #mean_martingale[:, 1:] = base_variables.clone()[:, :-1]
    #mean_martingale[:, 0:1] = 0

    logpz = standard_normal_logprob(base_variables).view(base_variables.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - log_det
    loss = -torch.mean(logpx)

    #normal_distri = torch.distributions.Normal(mean_martingale, torch.sqrt(vars))
    #LL = normal_distri.log_prob(base_variables)
    #LL = (torch.sum(LL, -1) - log_det) * masks
    return loss


def run_ctfp_model(args, aug_model, values, times, z=True, z_values = None):
    """
    Functions for running the ctfp model

    Parameters:
        args: arguments returned from parse_arguments
        aug_model: ctfp model as decoder
        values: observations, a 3-D tensor of shape batchsize x max_length x input_size
        times: observation time stampes, a 3-D tensor of shape batchsize x max_length x 1
        vars: Difference between consequtive observation time stampes.
              2-D tensor of size batch_size x length
        masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
               position is observation or padded dummy variables

    Returns:
    """
    if not z:
        aux = torch.cat([torch.zeros_like(values), times], dim=2)
        #aux : [100,89,2]
        aux = aux.view(-1, aux.shape[2])
        #aux : [8900,2]
        aux, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        #aux : [8900,2]
        aux = aux[:, args.effective_shape:]
        ## run flow backward
        if args.activation == "exp":
            transform_values, transform_logdet = log_jaco(values)
        elif args.activation == "softplus":
            transform_values, transform_logdet = inversoft_jaco(values)
        elif args.activation == "identity":
            transform_values = values
            transform_logdet = torch.sum(torch.zeros_like(values), dim=2)
        else:
            raise NotImplementedError

        aug_values = torch.cat(
            [transform_values.view(-1, transform_values.shape[2]), aux], dim=1
        )
        # import pdb
        # pdb.set_trace()
        base_values, flow_logdet = aug_model(
            aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
        )

        base_values = base_values[:, : args.effective_shape]
        base_values = base_values.view(values.shape[0], -1, args.effective_shape)

        ## flow_logdet and transform_logdet are both of size length*batch_size

        loss = compute_loss(
            flow_logdet.view(-1, base_values.shape[1])
            + transform_logdet.view(-1, base_values.shape[1]),
            base_values,
            args
        )
        return loss, base_values
    else:
        out = torch.zeros((values.shape[0],values.shape[1],values.shape[2]))
        # Z = torch.randn(values.shape[0], values.shape[2])
        # out[:,0,:] = Z
        for i in range(values.shape[1]-1):
            normal_distri = torch.distributions.Normal(out[:,0,:],torch.ones_like(out[:,0,:]))
            tmp = normal_distri.sample()
            out[:,i+1,:] = tmp
        Z = out.to(values)
        Z_sequence = Z

        if not z_values is None:
            Z = z_values
            # Z_sequence = Z.view(-1, Z.shape[2]).unsqueeze(1)
            # Z_sequence = Z_sequence.repeat(1, max_length, 1)

        time_to_cat = times.repeat(1, 1, 1)
        # time_to_cat = torch.zeros_like(time_to_cat)
        # batch_size, 24, 25
        times = torch.cat([Z_sequence, time_to_cat], -1)

        # batch_size, 24, 24
        values = values.repeat(1, 1, 1)
        # batch_size, 24, 25
        aux = times
        # batch_size*24, 25
        aux = aux.view(-1, aux.shape[2])
        # batch_size*24, 25
        aux, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        # batch_size*24, 24
        aux = aux[:, :args.effective_shape]
        # batch_size, 24, 24
        aux = aux.view(values.shape[0], -1, args.effective_shape)
        return aux


def create_separate_batches(data, times, masks):
    """
    Separate a batch of data with unequal length into smaller batch of size 1
    the length of each smaller batch is different and contains no padded dummy
    variables

    Parameters:
       data: observations, a 3-D tensor of shape batchsize x max_length x input_size
       times: observation time stamps, a 2-D tensor of shape batchsize x max_length
       masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
              position is observation or padded dummy variables

    Returns:
        a list of tuples containing the data, time, masks
    """
    batch_size = data.shape[0]
    data_size = data.shape[-1]
    ## only repeat the last dimension to concatenate with data
    repeat_times = tuple([1] * (len(data.shape) - 1) + [data_size])
    separate_batches = []
    for i in range(batch_size):
        length = int(torch.sum(masks[i]))
        data_item = data[i: i + 1, :length]
        time_item = times[i, :length].squeeze(-1)
        mask_item = masks[i: i + 1, :length].unsqueeze(-1).repeat(*repeat_times)
        separate_batches.append((torch.cat([data_item, mask_item], -1), time_item))
    return separate_batches


def run_latent_ctfp_model(
        args, encoder, aug_model, values, times, vars, masks, evaluation=False
):
    """
    Functions for running the latent ctfp model

    Parameters:
        args: arguments returned from parse_arguments
        encoder: ode_rnn model as encoder
        aug_model: ctfp model as decoder
        values: observations, a 3-D tensor of shape batchsize x max_length x input_size
        times: observation time stampes, a 3-D tensor of shape batchsize x max_length x 1
        vars: Difference between consequtive observation time stampes.
              2-D tensor of size batch_size x length
        masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
               position is observation or padded dummy variables
        evluation (bool): whether to run the latent ctfp model in the evaluation
                          mode. Return IWAE if set to true. Return both IWAE and
                          training loss if set to false

    Returns:
        Return IWAE if evaluation set to true.
        Return both IWAE and training loss if evaluation set to false.
    """
    if evaluation:
        num_iwae_samples = args.niwae_test
        batch_size = args.test_batch_size
    else:
        num_iwae_samples = args.num_iwae_samples
        batch_size = args.batch_size
    data_batches = create_separate_batches(values, times, masks)
    mean_list, stdv_list = [], []
    for item in data_batches:
        z_mean, z_stdv = encoder(item[0], item[1])
        mean_list.append(z_mean)
        stdv_list.append(z_stdv)
    means = torch.cat(mean_list, dim=1)
    stdvs = torch.cat(stdv_list, dim=1)
    # Sample latent variables
    repeat_times = [1] * len(means.shape)
    repeat_times[0] = num_iwae_samples
    means = means.repeat(*repeat_times)
    stdvs = stdvs.repeat(*repeat_times)
    latent = sample_standard_gaussian(means, stdvs)
    #latent : [3,100,10]


    ## Decode latent

    latent_sequence = latent.view(-1, args.latent_size).unsqueeze(1)
    # latent_sequence : [300,1,10]
    max_length = times.shape[1]
    latent_sequence = latent_sequence.repeat(1, max_length, 1)
    # latent_sequence : [300,89,10] 복사 한거임
    time_to_cat = times.repeat(num_iwae_samples, 1, 1)
    times = torch.cat([latent_sequence, time_to_cat], -1)
    #times : [300,89,11]
    #times = latent_sequence[300,89,10] + 시간[300,89,1]

    ## run flow forward to get augmented dimensions
    #values : [100,89,1]
    values = values.repeat(num_iwae_samples, 1, 1)
    #values : [300,89,1]
    #times : [300,89,11]
    aux = torch.cat([torch.zeros_like(values), times], dim=2).to(values)
    # aux : [300,89,12]
    aux = aux.view(-1, aux.shape[2])
    # aux : [26700 = 300 * 89, 12]
    # aug_model -> forward(z, logpz)
    # 여기서 aug_model은 X를 W로 분포 변경
    # 그러면 aux는 W_hat 일듯.
    aux, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
    # aux : [26700,12]
    aux = aux[:, args.effective_shape:]
    # aux : [26700,11]
    ## run flow backward
    if args.activation == "exp":
        transform_values, transform_logdet = log_jaco(values)
    elif args.activation == "softplus":
        transform_values, transform_logdet = inversoft_jaco(values)
    elif args.activation == "identity":
        transform_values = values
        transform_logdet = torch.sum(torch.zeros_like(values), dim=2)
    else:
        raise NotImplementedError
    #transform_values : [300,89,1]
    aug_values = torch.cat(
        [transform_values.view(-1, transform_values.shape[2]), aux], dim=1
    )
    base_values, flow_logdet = aug_model(
        aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
    )

    base_values = base_values[:, : args.effective_shape]
    base_values = base_values.view(values.shape[0], -1, args.effective_shape)

    ## flow_logdet and transform_logdet are both of size length*batch_size x length
    flow_logdet = flow_logdet.sum(-1).view(num_iwae_samples * batch_size, -1)
    transform_logdet = transform_logdet.view(num_iwae_samples * batch_size, -1)
    if len(vars.shape) == 2:
        vars_unsqueed = vars.unsqueeze(-1)
    else:
        vars_unsqueed = vars
    ll = compute_ll(
        flow_logdet + transform_logdet,
        base_values,
        vars_unsqueed.repeat(num_iwae_samples, 1, 1),
        masks.repeat(num_iwae_samples, 1),
    )
    ll = ll.view(num_iwae_samples, batch_size)
    ## Reconstruction log likelihood
    ## Compute KL divergence and compute IWAE
    posterior = torch.distributions.Normal(means[:1], stdvs[:1])
    prior = torch.distributions.Normal(
        torch.zeros_like(means[:1]), torch.ones_like(stdvs[:1])
    )
    # kl_latent = kl_divergence(posterior, prior).sum(-1)

    prior_z = prior.log_prob(latent).sum(-1)
    posterior_z = posterior.log_prob(latent).sum(-1)

    weights = ll + prior_z - posterior_z
    loss = -torch.logsumexp(weights, 0) + np.log(num_iwae_samples)
    if evaluation:
        return torch.sum(loss) / torch.sum(masks)
    loss = torch.sum(loss) / (batch_size * max_length)
    loss_training = -torch.sum(F.softmax(weights, 0).detach() * weights) / (
            batch_size * max_length
    )
    return loss, loss_training

def run_latent_ctfp_model2(
        args, aug_model, values, times, device, z=True
):
    """
    Functions for running the latent ctfp model

    Parameters:
        args: arguments returned from parse_arguments
        encoder: ode_rnn model as encoder
        aug_model: ctfp model as decoder
        values: observations, a 3-D tensor of shape batchsize x max_length x input_size
        times: observation time stampes, a 3-D tensor of shape batchsize x max_length x 1
        vars: Difference between consequtive observation time stampes.
              2-D tensor of size batch_size x length
        masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
               position is observation or padded dummy variables
        evluation (bool): whether to run the latent ctfp model in the evaluation
                          mode. Return IWAE if set to true. Return both IWAE and
                          training loss if set to false

    Returns:
        Return IWAE if evaluation set to true.
        Return both IWAE and training loss if evaluation set to false.
    """
    '''
    if evaluation:
        num_iwae_samples = args.niwae_test
        batch_size = args.test_batch_size
    else:
        num_iwae_samples = args.num_iwae_samples
        batch_size = args.batch_size
    data_batches = create_separate_batches(values, times, masks)
    mean_list, stdv_list = [], []
    # item[0] : 1 46 2 item[1] : 46
    # every iter different seq  -> output is same z_mean (1,1,10), z_stdv(1,1,10)
    for item in data_batches:
        z_mean, z_stdv = encoder(item[0], item[1])
        mean_list.append(z_mean)
        stdv_list.append(z_stdv)
    pdb.set_trace()

    means = torch.cat(mean_list, dim=1)
    stdvs = torch.cat(stdv_list, dim=1)
    # Sample latent variables means.shape = 3
    repeat_times = [1] * len(means.shape)
    repeat_times[0] = num_iwae_samples
    means = means.repeat(*repeat_times)
    stdvs = stdvs.repeat(*repeat_times)
    # mean, stdvs : 3 50 10 -> latent 3 50 10 
    '''
    if z==True:
        mu = torch.zeros(1, values.shape[0], values.shape[1]).to(device)
        stdvs = torch.ones(1, values.shape[0], values.shape[1]).to(device)
        latent = sample_standard_gaussian(mu, stdvs)
        latent_sequence = latent.view(-1, latent.shape[2]).unsqueeze(1)
        max_length = times.shape[1]
        # 150 89 10
        latent_sequence = latent_sequence.repeat(1, max_length, 1)
        #time_to_cat = times.repeat(1, 1, 1)
        time_to_cat = times.repeat(1, 1, 1)
        times = torch.cat([latent_sequence, time_to_cat], -1)

        ## run flow forward to get augmented dimensions
        # values 50 89 10 -> 150 89 10 
        values = values.repeat(1, 1, 1)
        # values 150,89,1 times: 150,89,11 -> aux 150,89,12
        #aux = torch.cat([torch.zeros_like(values), times], dim=2)
        aux = times
        # 13350, 12
        aux = aux.view(-1, aux.shape[2])
        # 13350, 12 , torch.zeros(aux.shape[0], 1) : 13350, 12  => 13350 12 

        aux, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        aux = aux[:, :args.effective_shape]
        aux = aux.view(values.shape[0], -1, args.effective_shape)
        return aux

    else:
        mu = torch.zeros(1, values.shape[0], values.shape[1]).to(device)
        stdvs = torch.ones(1, values.shape[0], values.shape[1]).to(device)
        latent = sample_standard_gaussian(mu, stdvs)

        vars = torch.ones_like(stdvs).squeeze(0)
        masks = torch.ones_like(stdvs).squeeze(0)
        ## Decode latent
    # 150 1 10 
        latent_sequence = latent.view(-1, latent.shape[2]).unsqueeze(1)
        # max_length 89
        max_length = times.shape[1]
        # 150 89 10
        latent_sequence = latent_sequence.repeat(1, max_length, 1)
        time_to_cat = times.repeat(1, 1, 1)
        times = torch.cat([latent_sequence, time_to_cat], -1)

        ## run flow forward to get augmented dimensions
        # values 50 89 10 -> 150 89 10 
        values = values.repeat(1, 1, 1)
        # values 150,89,1 times: 150,89,11 -> aux 150,89,12
        #aux = torch.cat([torch.zeros_like(values), times], dim=2)
        aux = times
        # 13350, 12
        aux = aux.view(-1, aux.shape[2])

        #out = torch.zeros_like(aux)

        #for idx in range(aux.shape[1]):
        #    h,_ = aug_model(aux[:,idx,:],torch.zeros(aux.shape[0], 1).to(aux), reverse=True)

        # 13350, 12 , torch.zeros(aux.shape[0], 1) : 13350, 12  => 13350 12 
        aux, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        # 13350 12 => 13350 11
        aux = aux[:, args.effective_shape:]
        ## run flow backward
        if args.activation == "exp":
            # values 150 89 1 => transform_valeus 150 89 1 transform_lodget 150 89 
            transform_values, transform_logdet = log_jaco(values)
        elif args.activation == "softplus":
            transform_values, transform_logdet = inversoft_jaco(values)
        elif args.activation == "identity":
            transform_values = values
            transform_logdet = torch.sum(torch.zeros_like(values), dim=2)
        else:
            raise NotImplementedError
        # transform + aux = aug _values  13350 1 + 13350 11 => 13350 12
        aug_values = torch.cat(
            [transform_values.view(-1, transform_values.shape[1]), aux], dim=1
        )

        # input : 13350 12, 13350 1 => base_values: 13350 12 flow_lodget 13350 1 
        base_values, flow_logdet = aug_model(
            aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
        )

        # base_values -> 150 89 1
        base_values = base_values[:, : args.effective_shape]
        base_values = base_values.view(values.shape[0], -1, args.effective_shape)

        ## flow_logdet and transform_logdet are both of size length*batch_size x length
        # flow_logdet -> 150 89 transform_logdet -> 150 89
        flow_logdet = flow_logdet.sum(-1).view(1 * base_values.shape[0], -1)
        transform_logdet = transform_logdet.view(1 * base_values.shape[0], -1)

        if len(vars.shape) == 2:
            vars_unsqueed = vars.unsqueeze(-1)
        else:
            vars_unsqueed = vars

        ll = compute_ll(
            flow_logdet + transform_logdet,
            base_values,
            vars_unsqueed.repeat(1, 1, 1),
            masks.repeat(1, 1),
        )
        ll = ll.view(1, base_values.shape[0])
        ## Reconstruction log likelihood
        ## Compute KL divergence and compute IWAE
        #posterior = torch.distributions.Normal(means[:1], stdvs[:1])
        #prior = torch.distributions.Normal(
        #    torch.zeros_like(means[:1]), torch.ones_like(stdvs[:1])
        #)
        # kl_latent = kl_divergence(posterior, prior).sum(-1)

        #prior_z = prior.log_prob(latent).sum(-1)
        #posterior_z = posterior.log_prob(latent).sum(-1)

        weights = ll #+ prior_z - posterior_z
        loss = -torch.logsumexp(weights, 0) + np.log(1)
        loss = torch.sum(loss) / (base_values.shape[0] * base_values.shape[1])
        loss_training = -torch.sum(F.softmax(weights, 0).detach() * weights) / (
                base_values.shape[0] *  base_values.shape[1]
        )
        return loss, loss_training

def run_latent_ctfp_model3(
        args, aug_model, values, times, device, z=True
):
    """
    Functions for running the latent ctfp model

    Parameters:
        args: arguments returned from parse_arguments
        encoder: ode_rnn model as encoder
        aug_model: ctfp model as decoder
        values: observations, a 3-D tensor of shape batchsize x max_length x input_size
        times: observation time stampes, a 3-D tensor of shape batchsize x max_length x 1
        vars: Difference between consequtive observation time stampes.
              2-D tensor of size batch_size x length
        masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
               position is observation or padded dummy variables
        evluation (bool): whether to run the latent ctfp model in the evaluation
                          mode. Return IWAE if set to true. Return both IWAE and
                          training loss if set to false

    Returns:
        Return IWAE if evaluation set to true.
        Return both IWAE and training loss if evaluation set to false.
    """
    '''
    if evaluation:
        num_iwae_samples = args.niwae_test
        batch_size = args.test_batch_size
    else:
        num_iwae_samples = args.num_iwae_samples
        batch_size = args.batch_size
    data_batches = create_separate_batches(values, times, masks)
    mean_list, stdv_list = [], []
    # item[0] : 1 46 2 item[1] : 46
    # every iter different seq  -> output is same z_mean (1,1,10), z_stdv(1,1,10)
    for item in data_batches:
        z_mean, z_stdv = encoder(item[0], item[1])
        mean_list.append(z_mean)
        stdv_list.append(z_stdv)
    pdb.set_trace()

    means = torch.cat(mean_list, dim=1)
    stdvs = torch.cat(stdv_list, dim=1)
    # Sample latent variables means.shape = 3
    repeat_times = [1] * len(means.shape)
    repeat_times[0] = num_iwae_samples
    means = means.repeat(*repeat_times)
    stdvs = stdvs.repeat(*repeat_times)
    # mean, stdvs : 3 50 10 -> latent 3 50 10 
    '''
    if z==True:
        mu = torch.zeros(1, values.shape[0], values.shape[1]).to(device)
        stdvs = torch.ones(1, values.shape[0], values.shape[1]).to(device)
        latent = sample_standard_gaussian(mu, stdvs)
        latent_sequence = latent.view(-1, latent.shape[2]).unsqueeze(1)
        max_length = times.shape[1]
        # 150 89 10
        latent_sequence = latent_sequence.repeat(1, max_length, 1)
        #time_to_cat = times.repeat(1, 1, 1)
        #times = torch.cat([latent_sequence, time_to_cat], -1)

        ## run flow forward to get augmented dimensions
        # values 50 89 10 -> 150 89 10 
        #values = values.repeat(1, 1, 1)
        # values 150,89,1 times: 150,89,11 -> aux 150,89,12
        #aux = torch.cat([torch.zeros_like(values), times], dim=2)
        aux = latent_sequence
        # 13350, 12
        
        aux = aux.view(-1, aux.shape[2])
        # 13350, 12 , torch.zeros(aux.shape[0], 1) : 13350, 12  => 13350 12 
        aux, _, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        #aux = aux[:, :args.effective_shape]
        aux = aux.view(values.shape[0], -1, args.effective_shape)
        if args.activation == "exp":
            # values 150 89 1 => transform_valeus 150 89 1 transform_lodget 150 89 
            aux, _ = log_jaco(aux, reverse=True)
        elif args.activation == "softplus":
            aux, _ = inversoft_jaco(aux,reverse=True)
        elif args.activation == "identity":
            pass 
        else:
            raise NotImplementedError
        return aux

    else:
        stdvs = torch.ones(1, values.shape[0], values.shape[1]).to(device)

        vars = torch.ones_like(stdvs).squeeze(0)
        masks = torch.ones_like(stdvs).squeeze(0)
        if args.activation == "exp":
            # values 150 89 1 => transform_valeus 150 89 1 transform_lodget 150 89 
            transform_values, transform_logdet = log_jaco(values)
        elif args.activation == "softplus":
            transform_values, transform_logdet = inversoft_jaco(values)
        elif args.activation == "identity":
            transform_values = values
            transform_logdet = torch.sum(torch.zeros_like(values), dim=2)
        else:
            raise NotImplementedError
        # transform + aux = aug _values  13350 1 + 13350 11 => 13350 12
        aug_values = transform_values.view(-1, transform_values.shape[1])

        # input : 13350 12, 13350 1 => base_values: 13350 12 flow_lodget 13350 1
        if args.kinetic_energy == None : 
            base_values, flow_logdet, _ = aug_model(
                aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
            )
        else:
            base_values, flow_logdet, reg_states = aug_model(
                aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
            )
            reg_states = tuple(torch.mean(rs) for rs in reg_states)
        # base_values -> 150 89 1
        #base_values = base_values[:, : args.effective_shape]
        base_values = base_values.view(values.shape[0], -1, args.effective_shape)

        ## flow_logdet and transform_logdet are both of size length*batch_size x length
        # flow_logdet -> 150 89 transform_logdet -> 150 89
        flow_logdet = flow_logdet.sum(-1).view(1 * base_values.shape[0], -1)
        transform_logdet = transform_logdet.view(1 * base_values.shape[0], -1)

        if len(vars.shape) == 2:
            vars_unsqueed = vars.unsqueeze(-1)
        else:
            vars_unsqueed = vars

        ll = compute_ll(
            flow_logdet + transform_logdet,
            base_values,
            vars_unsqueed.repeat(1, 1, 1),
            masks.repeat(1, 1),
        )
        ll = ll.view(1, base_values.shape[0])

        weights = ll #+ prior_z - posterior_z
        loss = -torch.logsumexp(weights, 0) + np.log(1)
        loss = torch.sum(loss) / (base_values.shape[0] * base_values.shape[1])
        loss_training = -torch.sum(F.softmax(weights, 0).detach() * weights) / (
                base_values.shape[0] *  base_values.shape[1]
        )
        if args.kinetic_energy == None : 
            return loss, loss_training
        else:
            return loss, loss_training, reg_states[0]

def run_latent_ctfp_model4(
        args, aug_model, values, times, device, z=True
):
    """
    Functions for running the latent ctfp model

    Parameters:
        args: arguments returned from parse_arguments
        encoder: ode_rnn model as encoder
        aug_model: ctfp model as decoder
        values: observations, a 3-D tensor of shape batchsize x max_length x input_size
        times: observation time stampes, a 3-D tensor of shape batchsize x max_length x 1
        vars: Difference between consequtive observation time stampes.
              2-D tensor of size batch_size x length
        masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
               position is observation or padded dummy variables
        evluation (bool): whether to run the latent ctfp model in the evaluation
                          mode. Return IWAE if set to true. Return both IWAE and
                          training loss if set to false

    Returns:
        Return IWAE if evaluation set to true.
        Return both IWAE and training loss if evaluation set to false.
    """
    '''
    if evaluation:
        num_iwae_samples = args.niwae_test
        batch_size = args.test_batch_size
    else:
        num_iwae_samples = args.num_iwae_samples
        batch_size = args.batch_size
    data_batches = create_separate_batches(values, times, masks)
    mean_list, stdv_list = [], []
    # item[0] : 1 46 2 item[1] : 46
    # every iter different seq  -> output is same z_mean (1,1,10), z_stdv(1,1,10)
    for item in data_batches:
        z_mean, z_stdv = encoder(item[0], item[1])
        mean_list.append(z_mean)
        stdv_list.append(z_stdv)
    pdb.set_trace()

    means = torch.cat(mean_list, dim=1)
    stdvs = torch.cat(stdv_list, dim=1)
    # Sample latent variables means.shape = 3
    repeat_times = [1] * len(means.shape)
    repeat_times[0] = num_iwae_samples
    means = means.repeat(*repeat_times)
    stdvs = stdvs.repeat(*repeat_times)
    # mean, stdvs : 3 50 10 -> latent 3 50 10 
    '''
    if z==True:
        
        mu = torch.zeros(1, values.shape[0], values.shape[1]).to(device)
        stdvs = torch.ones(1, values.shape[0], values.shape[1]).to(device)
        latent = sample_standard_gaussian(mu, stdvs)
        latent_sequence = latent.view(-1, latent.shape[2]).unsqueeze(1)
        max_length = times.shape[1]
        latent_sequence = latent_sequence.repeat(1, max_length, 1)
        aux = latent_sequence

        aux = aux.view(-1, aux.shape[2])

        aux, _,_ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        aux = aux.view(values.shape[0], -1, args.effective_shape)
        
        if args.activation == "exp":
            # values 150 89 1 => transform_valeus 150 89 1 transform_lodget 150 89 
            aux, _ = log_jaco(aux, reverse=True)
        elif args.activation == "softplus":
            aux, _ = inversoft_jaco(aux,reverse=True)
        elif args.activation == "identity":
            pass 
        else:
            raise NotImplementedError
        
        return aux

    else:
        stdvs = torch.ones(1, values.shape[0], values.shape[1]).to(device)

        vars = torch.ones_like(stdvs).squeeze(0)
        masks = torch.ones_like(stdvs).squeeze(0)
        if args.activation == "exp":
            # values 150 89 1 => transform_valeus 150 89 1 transform_lodget 150 89 
            transform_values, transform_logdet = log_jaco(values)
        elif args.activation == "softplus":
            transform_values, transform_logdet = inversoft_jaco(values)
        elif args.activation == "identity":
            transform_values = values
            transform_logdet = torch.sum(torch.zeros_like(values), dim=2)
        else:
            raise NotImplementedError
        # transform + aux = aug _values  13350 1 + 13350 11 => 13350 12
        aug_values = transform_values.view(-1, transform_values.shape[1])

        # input : 13350 12, 13350 1 => base_values: 13350 12 flow_lodget 13350 1 
        base_values, flow_logdet, reg_states = aug_model(
            aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
        )
        reg_states = tuple(torch.mean(rs) for rs in reg_states)
        # base_values -> 150 89 1
        #base_values = base_values[:, : args.effective_shape]
        base_values = base_values.view(values.shape[0], -1, args.effective_shape)

        ## flow_logdet and transform_logdet are both of size length*batch_size x length
        # flow_logdet -> 150 89 transform_logdet -> 150 89
        flow_logdet = flow_logdet.sum(-1).view(1 * base_values.shape[0], -1)
        transform_logdet = transform_logdet.view(1 * base_values.shape[0], -1)

        if len(vars.shape) == 2:
            vars_unsqueed = vars.unsqueeze(-1)
        else:
            vars_unsqueed = vars

        ll = compute_ll2(
            flow_logdet + transform_logdet,
            base_values,
            vars_unsqueed.repeat(1, 1, 1),
            masks.repeat(1, 1),
        )
        #ll = ll.view(1, base_values.shape[0])


        weights = ll #+ prior_z - posterior_z
        loss = -torch.logsumexp(weights, 0) + np.log(1)
        loss = torch.sum(loss) / (base_values.shape[0] * base_values.shape[1])
        loss_training = -torch.sum(F.softmax(weights, 0).detach() * weights) / (
                base_values.shape[0] *  base_values.shape[1]
        )
        #weights = ll #+ prior_z - posterior_z
        #loss = -torch.logsumexp(weights, 0) + np.log(1)
        #loss = torch.sum(loss) / (base_values.shape[0] * base_values.shape[1])
        #loss_training = -torch.sum(F.softmax(weights, 0).detach() * weights) / (
        #base_values.shape[0] *  base_values.shape[1]
        #)
        if args.kinetic_energy == None : 
            return loss, loss_training
        else:
            return loss, loss_training, reg_states[0]

def run_latent_ctfp_model5(
        args, aug_model, values, times, device, z=True
):
    """
    Functions for running the latent ctfp model

    Parameters:
        args: arguments returned from parse_arguments
        encoder: ode_rnn model as encoder
        aug_model: ctfp model as decoder
        values: observations, a 3-D tensor of shape batchsize x max_length x input_size
        times: observation time stampes, a 3-D tensor of shape batchsize x max_length x 1
        vars: Difference between consequtive observation time stampes.
              2-D tensor of size batch_size x length
        masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
               position is observation or padded dummy variables
        evluation (bool): whether to run the latent ctfp model in the evaluation
                          mode. Return IWAE if set to true. Return both IWAE and
                          training loss if set to false

    Returns:
        Return IWAE if evaluation set to true.
        Return both IWAE and training loss if evaluation set to false.
    """
    '''
    if evaluation:
        num_iwae_samples = args.niwae_test
        batch_size = args.test_batch_size
    else:
        num_iwae_samples = args.num_iwae_samples
        batch_size = args.batch_size
    data_batches = create_separate_batches(values, times, masks)
    mean_list, stdv_list = [], []
    # item[0] : 1 46 2 item[1] : 46
    # every iter different seq  -> output is same z_mean (1,1,10), z_stdv(1,1,10)
    for item in data_batches:
        z_mean, z_stdv = encoder(item[0], item[1])
        mean_list.append(z_mean)
        stdv_list.append(z_stdv)
    pdb.set_trace()

    means = torch.cat(mean_list, dim=1)
    stdvs = torch.cat(stdv_list, dim=1)
    # Sample latent variables means.shape = 3
    repeat_times = [1] * len(means.shape)
    repeat_times[0] = num_iwae_samples
    means = means.repeat(*repeat_times)
    stdvs = stdvs.repeat(*repeat_times)
    # mean, stdvs : 3 50 10 -> latent 3 50 10 
    '''
    if z==True:
        # mu 1 128 24 -> 1 128 182
        # import pdb;pdb.set_trace()
        mu = torch.zeros(1, values.shape[0], values.shape[2]).to(device)
        #mu = torch.zeros(values.shape[0], values.shape[1], values.shape[2]).to(device)
        # stdvs 1 128 24 -> 1 128 182
        stdvs = torch.ones(1, values.shape[0], values.shape[2]).to(device)
        #stdvs = torch.ones(values.shape[0], values.shape[1], values.shape[2]).to(device)
        # latent 1 128 24 -> 1 128 182
        latent = sample_standard_gaussian(mu, stdvs)
        # latent_sequence 128 1 24 -> 128 1 182
        latent_sequence = latent.view(-1, latent.shape[2]).unsqueeze(1)
        # max_length 24 -> 182
        max_length = times.shape[1]
        # latent_sequence 128 24 24 -> 128 182 182
        latent_sequence = latent_sequence.repeat(1, max_length, 1)    
        # aux 128 24 25
        aux = torch.cat([latent_sequence, times], dim=2)
        # aux 3072 25
        aux = aux.view(-1, aux.shape[2])
        aux, _, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)

        #aux, _, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        aux = aux[:, :-times.shape[2]]
        aux = aux.view(values.shape[0], -1, values.shape[2])
        if args.activation == "exp":
            # values 150 89 1 => transform_valeus 150 89 1 transform_lodget 150 89 
            aux, _ = log_jaco(aux, reverse=True)
        elif args.activation == "softplus":
            aux = inversoft_jaco(aux,reverse=True)
        elif args.activation == "identity":
            pass 
        else:
            raise NotImplementedError

        return aux

    else:
        # import pdb;pdb.set_trace()
        max_length = times.shape[1]
        time_to_cat = times.repeat(args.num_iwae_samples, 1, 1)
        values = values.repeat(args.num_iwae_samples, 1, 1)

        aux = torch.cat([torch.zeros_like(values), time_to_cat], dim=2)
        aux = aux.view(-1, aux.shape[2])
        aux, _,_ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        
        aux = aux[:, -times.shape[2]:]

        #stdvs = torch.ones(1, values.shape[0], values.shape[2]).to(device)
        stdvs = torch.ones(1, values.shape[0], values.shape[1]).to(device)
        # import pdb;pdb.set_trace()
        vars = torch.ones_like(stdvs).squeeze(0)
        masks = torch.ones_like(stdvs).squeeze(0)
        if args.activation == "exp":
            # values 150 89 1 => transform_valeus 150 89 1 transform_lodget 150 89 
            transform_values, transform_logdet = log_jaco(values)
        elif args.activation == "softplus":
            transform_values, transform_logdet = inversoft_jaco(values)
        elif args.activation == "identity":
            transform_values = values
            transform_logdet = torch.sum(torch.zeros_like(values), dim=2)
        else:
            raise NotImplementedError
        # transform + aux = aug _values  13350 1 + 13350 11 => 13350 12

        aug_values = transform_values.view(-1, transform_values.shape[2])
        aug_values = torch.cat([aug_values, aux], dim=1)
        #aug_values = aug_values.view(-1, aug_values.shape[2])

        # input : 13350 12, 13350 1 => base_values: 13350 12 flow_lodget 13350 1
        if args.kinetic_energy == None : 
            base_values, flow_logdet, _ = aug_model(
                aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
            )
        else:
            base_values, flow_logdet, reg_states = aug_model(
                aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
            )
            reg_states = tuple(torch.mean(rs) for rs in reg_states)
        # base_values -> 150 89 1
        base_values = base_values[:, :-times.shape[2]]
        base_values = base_values.view(values.shape[0], -1, values.shape[2])

        ## flow_logdet and transform_logdet are both of size length*batch_size x length
        # flow_logdet -> 150 89 transform_logdet -> 150 89
        flow_logdet = flow_logdet.sum(-1).view(base_values.shape[0], -1)
        transform_logdet = transform_logdet.view(base_values.shape[0], -1)

        if len(vars.shape) == 2:
            vars_unsqueed = vars.unsqueeze(-1)
        else:
            vars_unsqueed = vars

        ll = compute_ll(
            flow_logdet + transform_logdet,
            base_values,
            vars_unsqueed.repeat(1, 1, 1),
            masks.repeat(1, 1),
        )
        ll = ll.view(args.num_iwae_samples, int((base_values.shape[0]/args.num_iwae_samples)))

        weights = ll #+ prior_z - posterior_z
        loss = -torch.logsumexp(weights, 0) + np.log(args.num_iwae_samples)
        loss = torch.sum(loss) / (int((base_values.shape[0]/args.num_iwae_samples))*max_length)
        loss_training = -torch.sum(F.softmax(weights, 0).detach() * weights) / (
                int((base_values.shape[0]/args.num_iwae_samples))*max_length
        )
        if args.kinetic_energy == None : 
            return loss, loss_training
        else:
            return loss, loss_training, reg_states[0]

def run_latent_ctfp_model5_2(
        args, aug_model, values, times, device, z=True
):
    """
    Functions for running the latent ctfp model
    Parameters:
        args: arguments returned from parse_arguments
        encoder: ode_rnn model as encoder
        aug_model: ctfp model as decoder
        values: observations, a 3-D tensor of shape batchsize x max_length x input_size
        times: observation time stampes, a 3-D tensor of shape batchsize x max_length x 1
        vars: Difference between consequtive observation time stampes.
              2-D tensor of size batch_size x length
        masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
               position is observation or padded dummy variables
        evluation (bool): whether to run the latent ctfp model in the evaluation
                          mode. Return IWAE if set to true. Return both IWAE and
                          training loss if set to false
    Returns:
        Return IWAE if evaluation set to true.
        Return both IWAE and training loss if evaluation set to false.
    """
    '''
    if evaluation:
        num_iwae_samples = args.niwae_test
        batch_size = args.test_batch_size
    else:
        num_iwae_samples = args.num_iwae_samples
        batch_size = args.batch_size
    data_batches = create_separate_batches(values, times, masks)
    mean_list, stdv_list = [], []
    # item[0] : 1 46 2 item[1] : 46
    # every iter different seq  -> output is same z_mean (1,1,10), z_stdv(1,1,10)
    for item in data_batches:
        z_mean, z_stdv = encoder(item[0], item[1])
        mean_list.append(z_mean)
        stdv_list.append(z_stdv)
    pdb.set_trace()
    means = torch.cat(mean_list, dim=1)
    stdvs = torch.cat(stdv_list, dim=1)
    # Sample latent variables means.shape = 3
    repeat_times = [1] * len(means.shape)
    repeat_times[0] = num_iwae_samples
    means = means.repeat(*repeat_times)
    stdvs = stdvs.repeat(*repeat_times)
    # mean, stdvs : 3 50 10 -> latent 3 50 10 
    '''
    if z==True:
        mu = torch.zeros(1, values.shape[0], values.shape[2]).to(device)
        stdvs = torch.ones(1, values.shape[0], values.shape[2]).to(device)
        latent = sample_standard_gaussian(mu, stdvs)
        latent_sequence = latent.view(-1, latent.shape[2]).unsqueeze(1)
        max_length = times.shape[1]
        # 150 89 10
        latent_sequence = latent_sequence.repeat(1, max_length, 1)    
        #aux = torch.cat([latent_sequence, times], dim=2)
        aux = torch.cat([latent_sequence, times], dim=2)
        #aux = latent_sequence
        # 13350, 12
        aux = aux.view(-1, aux.shape[2])
        # 13350, 12 , torch.zeros(aux.shape[0], 1) : 13350, 12  => 13350 12
        # import pdb;pdb.set_trace()
        aux, _, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        #aux, _, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        aux = aux[:, :-times.shape[2]]
        aux = aux.view(values.shape[0], -1, values.shape[2])
        if args.activation == "exp":
            # values 150 89 1 => transform_valeus 150 89 1 transform_lodget 150 89 
            aux, _ = log_jaco(aux, reverse=True)
        elif args.activation == "softplus":
            aux = inversoft_jaco(aux,reverse=True)
        elif args.activation == "identity":
            pass 
        else:
            raise NotImplementedError
        return aux
    else:
        max_length = times.shape[1]
        time_to_cat = times.repeat(args.num_iwae_samples, 1, 1)
        values = values.repeat(args.num_iwae_samples, 1, 1)
        aux = torch.cat([torch.zeros_like(values), time_to_cat], dim=2)
        aux = aux.view(-1, aux.shape[2])
        aux, _,_ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        aux = aux[:, -times.shape[2]:]
        stdvs = torch.ones(1, values.shape[0], values.shape[1]).to(device)
        vars = torch.ones_like(stdvs).squeeze(0)
        masks = torch.ones_like(stdvs).squeeze(0)
        if args.activation == "exp":
            # values 150 89 1 => transform_valeus 150 89 1 transform_lodget 150 89 
            transform_values, transform_logdet = log_jaco(values)
        elif args.activation == "softplus":
            transform_values, transform_logdet = inversoft_jaco(values)
        elif args.activation == "identity":
            transform_values = values
            transform_logdet = torch.sum(torch.zeros_like(values), dim=2)
        else:
            raise NotImplementedError
        # transform + aux = aug _values  13350 1 + 13350 11 => 13350 12
        aug_values = transform_values.view(-1, transform_values.shape[2])
        aug_values = torch.cat([aug_values, aux], dim=1)
        #aug_values = aug_values.view(-1, aug_values.shape[2])
        # input : 13350 12, 13350 1 => base_values: 13350 12 flow_lodget 13350 1
        if args.kinetic_energy == None : 
            base_values, flow_logdet, _ = aug_model(
                aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
            )
        else:
            base_values, flow_logdet, reg_states = aug_model(
                aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
            )
            reg_states = tuple(torch.mean(rs) for rs in reg_states)
        # base_values -> 150 89 1
        base_values = base_values[:, :-times.shape[2]]
        base_values = base_values.view(values.shape[0], -1, args.effective_shape)
        ## flow_logdet and transform_logdet are both of size length*batch_size x length
        # flow_logdet -> 150 89 transform_logdet -> 150 89
        flow_logdet = flow_logdet.sum(-1).view(base_values.shape[0], -1)
        transform_logdet = transform_logdet.view(base_values.shape[0], -1)
        if len(vars.shape) == 2:
            vars_unsqueed = vars.unsqueeze(-1)
        else:
            vars_unsqueed = vars
        ll = compute_ll(
            flow_logdet + transform_logdet,
            base_values,
            vars_unsqueed.repeat(1, 1, 1),
            masks.repeat(1, 1),
        )
        ll = ll.view(args.num_iwae_samples, int((base_values.shape[0]/args.num_iwae_samples)))
        weights = ll #+ prior_z - posterior_z
        loss = -torch.logsumexp(weights, 0) + np.log(args.num_iwae_samples)
        loss = torch.sum(loss) / (int((base_values.shape[0]/args.num_iwae_samples))*base_values.shape[1])
        loss_training = -torch.sum(F.softmax(weights, 0).detach() * weights) / (
                int((base_values.shape[0]/args.num_iwae_samples))*(base_values.shape[1])
        )
        if args.kinetic_energy == None : 
            return loss, loss_training
        else:
            return loss, loss_training, reg_states[0]
def run_latent_ctfp_model5_3(
        args, aug_model, values, times, device, z=True
):

    if z==True:      
        mu = torch.zeros(1, values.shape[0], values.shape[1]).to(device)
        stdvs = torch.ones(1, values.shape[0], values.shape[1]).to(device)
        latent = sample_standard_gaussian(mu, stdvs)

        latent_sequence = latent.view(-1, latent.shape[2]).unsqueeze(1)
        max_length = times.shape[1]
        latent_sequence = latent_sequence.repeat(1, max_length, 1)
        aux = torch.cat([latent_sequence, times], dim=2)
        aux = aux.view(-1, aux.shape[2])

        aux, _,_ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        aux = aux[:,: args.effective_shape]
        aux = aux.view(values.shape[0], -1, args.effective_shape)
        
        if args.activation == "exp":
            # values 150 89 1 => transform_valeus 150 89 1 transform_lodget 150 89 
            aux, _ = log_jaco(aux, reverse=True)
        elif args.activation == "softplus":
            aux, _ = inversoft_jaco(aux,reverse=True)
        elif args.activation == "identity":
            pass 
        else:
            raise NotImplementedError
        
        return aux

    else:
        stdvs = torch.ones(1, values.shape[0], values.shape[1]).to(device)

        vars = torch.ones_like(stdvs).squeeze(0)
        masks = torch.ones_like(stdvs).squeeze(0)

        time_to_cat = times.repeat(args.num_iwae_samples, 1, 1)
        values = values.repeat(args.num_iwae_samples, 1, 1)
        aux = torch.cat([torch.zeros_like(values), time_to_cat], dim=2)
        aux = aux.view(-1, aux.shape[2])
        aux, _,_ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        aux = aux[:, args.effective_shape: ]


        if args.activation == "exp":
            # values 150 89 1 => transform_valeus 150 89 1 transform_lodget 150 89 
            transform_values, transform_logdet = log_jaco(values)
        elif args.activation == "softplus":
            transform_values, transform_logdet = inversoft_jaco(values)
        elif args.activation == "identity":
            transform_values = values
            transform_logdet = torch.sum(torch.zeros_like(values), dim=2)
        else:
            raise NotImplementedError
        # transform + aux = aug _values  13350 1 + 13350 11 => 13350 12
        aug_values = transform_values.view(-1, transform_values.shape[1])
        aug_values = torch.cat([aug_values, aux], dim=1)

        # input : 13350 12, 13350 1 => base_values: 13350 12 flow_lodget 13350 1 
        base_values, flow_logdet, reg_states = aug_model(
            aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
        )
        if args.kinetic_energy == None : 
            base_values, flow_logdet, _ = aug_model(
                aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
            )
        else:
            base_values, flow_logdet, reg_states = aug_model(
                aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
            )
            reg_states = tuple(torch.mean(rs) for rs in reg_states)
        # base_values -> 150 89 1
        base_values = base_values[:, : args.effective_shape]
        base_values = base_values.view(values.shape[0], -1, args.effective_shape)

        ## flow_logdet and transform_logdet are both of size length*batch_size x length
        # flow_logdet -> 150 89 transform_logdet -> 150 89
        flow_logdet = flow_logdet.sum(-1).view(1 * base_values.shape[0], -1)
        transform_logdet = transform_logdet.view(1 * base_values.shape[0], -1)

        if len(vars.shape) == 2:
            vars_unsqueed = vars.unsqueeze(-1)
        else:
            vars_unsqueed = vars

        ll = compute_ll(
            flow_logdet + transform_logdet,
            base_values,
            vars_unsqueed.repeat(1, 1, 1),
            masks.repeat(1, 1),
        )
        #ll = ll.view(1, base_values.shape[0])

        ll = ll.view(args.num_iwae_samples, int((base_values.shape[0]/args.num_iwae_samples)))
        
        weights = ll #+ prior_z - posterior_z
        loss = -torch.logsumexp(weights, 0) + np.log(args.num_iwae_samples)
        loss = torch.sum(loss) / (int((base_values.shape[0]/args.num_iwae_samples))*base_values.shape[1])
        loss_training = -torch.sum(F.softmax(weights, 0).detach() * weights) / (
                int((base_values.shape[0]/args.num_iwae_samples))*base_values.shape[1]
        )
        #weights = ll #+ prior_z - posterior_z
        #loss = -torch.logsumexp(weights, 0) + np.log(1)
        #loss = torch.sum(loss) / (base_values.shape[0] * base_values.shape[1])
        #loss_training = -torch.sum(F.softmax(weights, 0).detach() * weights) / (
        #base_values.shape[0] *  base_values.shape[1]
        #)
        if args.kinetic_energy == None : 
            return loss, loss_training
        else:
            return loss, loss_training, reg_states[0]
