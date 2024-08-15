import torch
import torch.nn as nn
import torch.nn.functional as F

import models.p_or_t_forcing.cfg as cfg

multinomial = cfg.sample_methods.multinomial


class Generator(nn.Module):
    def __init__(
            self, input_size, channels, hidden_size, device=None
    ):
        # input size means embedding size as usual
        super(Generator, self).__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRUCell(
            channels,          # since just one value
            hidden_size
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, channels)

    def forward(self, inputs, hidden):
        next_hidden = self.rnn(inputs, hidden)          # [batch_size x hidden_size]
        next_hidden = self.norm(next_hidden)
        unnormalized_scores = self.linear(next_hidden)    # [batch_size x vocab_size]
        return unnormalized_scores, next_hidden

    def consume(self, word_input, hidden, sampling, temperature=3):
        hidden_states = [hidden]
        seq_len, batch_size = word_input.size(1), word_input.size(0)
        criterion = nn.MSELoss()            # for regression task
        loss = 0
        if sampling:
            # free-running mode
            current_word_inputs = word_input[:, 0]
            for idx in range(seq_len - 1):
                scores, hidden = self(current_word_inputs, hidden)
                loss += criterion(scores, word_input[:, idx + 1])
                hidden_states.append(hidden)
                current_word_inputs = scores                # use prediction as input for next step

        else:
            # teacher forcing mode
            for idx in range(seq_len - 1):
                scores, hidden = self(word_input[:, idx], hidden)
                loss += criterion(scores, word_input[:, idx + 1])
                hidden_states.append(hidden)

        hidden_states = torch.stack(hidden_states, dim=1)
        if torch.any(torch.isnan(hidden_states)):
            raise ValueError("NaN detected in hidden states!")
        return loss, hidden_states, None

    def sample(self, word_input, hidden):
        with torch.no_grad():
            samples = [word_input[:, 0]]
            seq_len, batch_size = word_input.size(1), word_input.size(0)
            current_word_inputs = word_input[:, 0]
            for idx in range(seq_len - 1):
                scores, hidden = self(current_word_inputs, hidden)
                current_word_inputs = scores                # use prediction as input for next step
                samples.append(scores)
        samples = torch.stack(samples, dim=1)
        return samples

    def init_hidden(self, batch_size, strategy=cfg.inits.xavier):
        if strategy == cfg.inits.zeros:
            hidden = torch.zeros(batch_size, self.hidden_size)
        elif strategy == cfg.inits.xavier:
            hidden = torch.zeros(batch_size, self.hidden_size)
            hidden = torch.nn.init.xavier_normal_(hidden)
        hidden = hidden.to(self.device)
        return hidden

