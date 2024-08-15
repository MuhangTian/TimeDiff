import models.p_or_t_forcing.cfg as cfg

import torch
import torch.nn as nn

import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, linear_size, lin_dropout, device):
        super(Discriminator, self).__init__()

        self.hidden_size = hidden_size
        self.linear_size = linear_size
        self.device = device

        self.rnn = nn.GRU(
            input_size, hidden_size,
            batch_first=True, bidirectional=True
        )

        self.linears = nn.Sequential(
            nn.Linear(hidden_size * 2, linear_size),
            nn.ReLU(),
            nn.Dropout(lin_dropout),
            nn.LayerNorm(linear_size),
            nn.Linear(linear_size, linear_size),
            nn.ReLU(),
            nn.Dropout(lin_dropout),
            nn.LayerNorm(linear_size),
            nn.Linear(linear_size, 1)
        )

    def forward(self, hidden_states):
        # hidden_states                                           # [batch_size * seq_len * hid_size]
        batch_size = hidden_states.size(0)
        initial_hidden = self.init_hidden(hidden_states.size(0))
        _, rnn_final_hidden = self.rnn(
            hidden_states, initial_hidden)                        # [2 * batch_size * hid_size]
        rnn_final_hidden = rnn_final_hidden.view(batch_size, -1)  # [batch_size * (2 * hidden_size)]
        scores = self.linears(rnn_final_hidden)                   # flatten
        scores = F.sigmoid(scores)          # predict probability for classification
        return scores

    def init_hidden(self, batch_size):
        hidden = torch.zeros(2, batch_size, self.hidden_size).to(self.device)
        return hidden