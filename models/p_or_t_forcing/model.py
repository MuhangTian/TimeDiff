from torch import nn

import models.p_or_t_forcing.cfg as cfg
from models.p_or_t_forcing.modules.discriminator import Discriminator
from models.p_or_t_forcing.modules.generator import Generator


class LMGan(nn.Module):

    def __init__(self, opt):
        super(LMGan, self).__init__()
        self.opt = opt
        self.generator = Generator(
            opt.vocab_size,
            opt.channels,
            opt.hidden_size,
            opt.device
        )

        self.discriminator = Discriminator(
            opt.hidden_size,
            opt.d_hidden_size,
            opt.d_linear_size,
            opt.d_dropout,
            opt.device
        ) if opt.adversarial else None

    def forward(self, input, adversarial=True):
        batch_size = input.size(0)
        if not adversarial: # train with teacher forcing
            start_hidden = self.generator.init_hidden(batch_size, strategy=cfg.inits.xavier)

            loss, gen_hidden_states, _ = self.generator.consume(input, start_hidden, sampling=False)

            return loss, None, None, None, None
        else:           # train with professor forcing
            start_hidden_nll = self.generator.init_hidden(batch_size, strategy=cfg.inits.xavier)
            loss_nll, gen_hidden_states_nll, _ = self.generator.consume(
                input, start_hidden_nll, sampling=False)

            # run one pass with sampling
            start_hidden_adv = self.generator.init_hidden(batch_size, strategy=cfg.inits.xavier)
            loss_adv, gen_hidden_states_adv, _ = self.generator.consume(
                input,
                start_hidden_adv,
                sampling=True,
                temperature=self.opt.temperature
            )
            # these two passes have computational graphs that are completely different, so
            # in the future we can call backwards for each loss consequently

            # Now, call the discriminator
            teacher_forcing_scores = self.discriminator(gen_hidden_states_nll)
            autoregressive_scores = self.discriminator(gen_hidden_states_adv)

            return loss_nll + loss_adv, teacher_forcing_scores, autoregressive_scores,\
        gen_hidden_states_nll, gen_hidden_states_adv

    def view_rnn_grad_norms(self):
        norms_dict = {
            k: v.grad.norm().item()
            for k, v in self.named_parameters()
            if 'rnn' in k
        }
        return norms_dict

