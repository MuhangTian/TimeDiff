import math
import time

import torch

import models.p_or_t_forcing.cfg as cfg


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def pretrain_embedding(embedding, matrix, freeze=True):
    """
    Function that is made for setting the content
    of nn.Embedding layer to the specific matrix
    :param embedding: nn.Embedding instance
    :param matrix: torch.FloatTensor
    :param freeze: whether make parameters learnable
    :return: nothing, operations are in-place
    """
    assert isinstance(matrix, torch.FloatTensor)
    assert embedding._parameters['weight'].size() == matrix.size()

    embedding._parameters['weight'] = torch.nn.parameter.Parameter(
        matrix, requires_grad=not freeze
    )


def make_pretrained_embedding_matrix(dataset, pretrained_embedding):
    # TODO: not random
    """
    Creates pretrained embedding for out dataset from the generic one
    by adding first dataset.n_special random columns
    :param dataset:
    :param pretrained_embedding:
    :return:
    """
    assert isinstance(pretrained_embedding, torch.FloatTensor)
    _t = torch.zeros(dataset.n_special, pretrained_embedding.size(1))
    torch.normal(std=torch.ones_like(_t) * 4, out=_t)
    size = cfg.model.vocab_size - dataset.n_special

    new_vectors = torch.cat((_t, pretrained_embedding[:size]), 0)
    new_vectors = new_vectors.div(torch.norm(new_vectors, 2))
    return new_vectors

