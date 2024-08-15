import torch
from torch import optim

import models.p_or_t_forcing.cfg as cfg


class Trainer:
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model
        self.g_optim = optim.Adam(model.generator.parameters(), lr=opt.learning_rate)
        if opt.adversarial:
            self.d_optim = optim.Adam(model.discriminator.parameters(), lr=opt.learning_rate)

    def train(self, opt, input):
        self.model.zero_grad()
        nll_loss, tf_scores, ar_scores, is_nll, is_adv = self.model(input, adversarial=self.opt.adversarial)
        
        if opt.adversarial:
            if torch.any(torch.isnan(nll_loss)) or torch.any(torch.isnan(tf_scores)) or torch.any(torch.isnan(ar_scores)):
                raise ValueError("NaN detected! nll_loss: {}, tf_scores: {}, ar_scores: {}".format(nll_loss, tf_scores, ar_scores))
        else:
            if torch.any(torch.isnan(nll_loss)):
                raise ValueError("NaN detected!")
            
        nll_loss.backward(retain_graph=True)
        if self.opt.adversarial:
            update_g, update_d = self._need_update(tf_scores, ar_scores)
            g_loss = self._calculate_generator_loss(tf_scores, ar_scores).sum()
            d_loss = self._calcualte_discriminator_loss(tf_scores, ar_scores).sum()
            if not update_g:
                is_nll.detach_()
                is_adv.detach_()
            d_loss.backward(retain_graph=True)
            g_loss.backward()
            g_loss_value = g_loss.item()
            d_loss_value = d_loss.item()
        else:
            g_loss_value = None
            d_loss_value = None
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.clip)

        self.g_optim.step()
        if self.opt.adversarial and update_d:
            self.d_optim.step()

        return nll_loss.cpu().item(), g_loss_value, d_loss_value

    def sample(self, init_sample, batch_size):
        start_hidden = self.model.generator.init_hidden(batch_size, strategy=cfg.inits.zeros)
        one_sample = self.model.generator.sample(init_sample, start_hidden)
        return one_sample

    def _calculate_generator_loss(self, tf_scores, ar_scores):
        """
        Calculates Fool-The-Discriminator loss
        Optionally calculate the reverse loss
        :param tf_scores: Teacher Forcing scores
        :param ar_scores: AutoRegressive scores
        :return:
        """
        loss = torch.log(ar_scores) * (-1)

        if self.opt.optional_loss:
            loss += torch.log(1 - tf_scores) * (-1)
        return loss

    def _calcualte_discriminator_loss(self, tf_scores, ar_scores):
        tf_loss = torch.log(tf_scores) * (-1)
        ar_loss = torch.log(1 - ar_scores) * (-1)
        return tf_loss + ar_loss

    def _need_update(self, tf_scores, ar_scores):
        """
        Discriminator accuracy < 0.75 --> don't backpropagate to generator
        Discriminator accuracy > 0.99 --> don't train discriminator
        Discriminator guess is calculated as x > 0.5
        :param tf_scores: Teacher Forcing scores [batch_size * 1]
        :param ar_scores: AutoRegressive scores  [batch_size * 1]
        :return:
        """
        correct = float((tf_scores.view(-1) > 0.5).sum() + (ar_scores.view(-1) < 0.5).sum())
        d_accuracy = correct / (tf_scores.size(0) * 2)
        if d_accuracy < 0.75:
            return False, True
        elif d_accuracy > 0.99:
            return True, False
        else:
            return True, True