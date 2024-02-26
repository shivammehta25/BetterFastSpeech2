""" from https://github.com/jaywalnut310/glow-tts """

import math

import torch
import torch.nn as nn
from einops import rearrange

import fs2.utils as utils
from fs2.utils.model import sequence_mask

log = utils.get_pylogger(__name__)


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    return m

class VariancePredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                args.d_model,
                args.hidden_dim,
                kernel_size=args.kernel_size,
                padding=(args.kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.ln1 = nn.LayerNorm(args.hidden_dim)
        self.dropout_module = nn.Dropout(
            p=args.dropout
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                args.hidden_dim,
                args.hidden_dim,
                kernel_size=args.kernel_size,
                padding=(args.kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.ln2 = nn.LayerNorm(args.hidden_dim)
        self.proj = nn.Linear(args.hidden_dim, 1)

    def forward(self, x):
        # Input: B x T x C; Output: B x T
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln1(x))
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln2(x))
        return self.proj(x).squeeze(dim=2)

class LengthRegulator(nn.Module):
    def forward(self, x, durations):
        # x: B x T x C
        out_lens = durations.sum(dim=1)
        max_len = out_lens.max()
        bsz, seq_len, dim = x.size()
        out = x.new_zeros((bsz, max_len, dim))

        for b in range(bsz):
            indices = []
            for t in range(seq_len):
                indices.extend([t] * utils.item(durations[b, t]))
            indices = torch.tensor(indices, dtype=torch.long).to(x.device)
            out_len = utils.item(out_lens[b])
            out[b, :out_len] = x[b].index_select(0, indices)

        return out, out_lens
    
class VarianceAdaptor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.length_regulator = LengthRegulator()
        self.duration_predictor = VariancePredictor(args)
        self.pitch_predictor = VariancePredictor(args)
        self.energy_predictor = VariancePredictor(args)

        n_bins, steps = self.args.n_bins, self.args.n_bins - 1
        self.pitch_bins = torch.linspace(args.pitch_min, args.pitch_max, steps)
        self.embed_pitch = Embedding(n_bins, args.d_model)
        self.energy_bins = torch.linspace(args.energy_min, args.energy_max, steps)
        self.embed_energy = Embedding(n_bins, args.d_model)

    def get_pitch_emb(self, x, tgt=None, factor=1.0):
        out = self.pitch_predictor(x)
        bins = self.pitch_bins.to(x.device)
        if tgt is None:
            out = out * factor
            emb = self.embed_pitch(torch.bucketize(out, bins))
        else:
            emb = self.embed_pitch(torch.bucketize(tgt, bins))
        return out, emb

    def get_energy_emb(self, x, tgt=None, factor=1.0):
        out = self.energy_predictor(x)
        bins = self.energy_bins.to(x.device)
        if tgt is None:
            out = out * factor
            emb = self.embed_energy(torch.bucketize(out, bins))
        else:
            emb = self.embed_energy(torch.bucketize(tgt, bins))
        return out, emb

    def forward(
        self,
        x,
        padding_mask,
        durations=None,
        pitches=None,
        energies=None,
        d_factor=1.0,
        p_factor=1.0,
        e_factor=1.0,
    ):
        # x: B x T x C
        log_dur_out = self.duration_predictor(x)
        dur_out = torch.clamp(
            torch.round((torch.exp(log_dur_out) - 1) * d_factor).long(), min=0
        )
        dur_out.masked_fill_(padding_mask, 0)

        pitch_out, pitch_emb = self.get_pitch_emb(x, pitches, p_factor)
        x = x + pitch_emb
        energy_out, energy_emb = self.get_energy_emb(x, energies, e_factor)
        x = x + energy_emb

        x, out_lens = self.length_regulator(
            x, dur_out if durations is None else durations
        )

        return x, out_lens, log_dur_out, pitch_out, energy_out