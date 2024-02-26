import torch
import torch.nn as nn
import torch.nn.functional as F


class Postnet(nn.Module):
    def __init__(self, in_dim, n_channels, kernel_size, n_layers, dropout):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        assert kernel_size % 2 == 1
        for i in range(n_layers):
            cur_layers = (
                [
                    nn.Conv1d(
                        in_dim if i == 0 else n_channels,
                        n_channels if i < n_layers - 1 else in_dim,
                        kernel_size=kernel_size,
                        padding=((kernel_size - 1) // 2),
                    ),
                    nn.BatchNorm1d(n_channels if i < n_layers - 1 else in_dim),
                ]
                + ([nn.Tanh()] if i < n_layers - 1 else [])
                + [nn.Dropout(dropout)]
            )
            nn.init.xavier_uniform_(
                cur_layers[0].weight,
                torch.nn.init.calculate_gain("tanh" if i < n_layers - 1 else "linear"),
            )
            self.convolutions.append(nn.Sequential(*cur_layers))

    def forward(self, x):
        x = x.transpose(1, 2)  # B x T x C -> B x C x T
        for conv in self.convolutions:
            x = conv(x)
        return x.transpose(1, 2)