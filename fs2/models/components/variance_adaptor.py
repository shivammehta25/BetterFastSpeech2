import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange

import fs2.utils as utils
from fs2.utils.model import expand_lengths

log = utils.get_pylogger(__name__)


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

    def forward(self, x, mask):
        # Input: B x T x C; Output: B x T
        x = self.conv1((x * mask).transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln1(x))
        x = self.conv2((x * mask).transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln2(x))
        return (self.proj(x) * mask).squeeze(dim=2)

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
    
        
class VarianceAdaptor(nn.Module):
    def __init__(self, args, pitch_min, pitch_max, energy_min, energy_max):
        super().__init__()
        self.args = args
        self.duration_predictor = VariancePredictor(args)
        self.pitch_predictor = VariancePredictor(args)
        self.energy_predictor = VariancePredictor(args)

        n_bins, steps = self.args.n_bins, self.args.n_bins - 1
        self.pitch_bins = nn.Parameter(torch.linspace(pitch_min.item(), pitch_max.item(), steps), requires_grad=False)
        self.embed_pitch = nn.Embedding(n_bins, args.d_model)
        nn.init.normal_(self.embed_pitch.weight, mean=0, std=args.d_model**-0.5)
        self.energy_bins =  nn.Parameter(torch.linspace(energy_min.item(), energy_max.item(), steps), requires_grad=False)
        self.embed_energy = nn.Embedding(n_bins, args.d_model) 
        nn.init.normal_(self.embed_energy.weight, mean=0, std=args.d_model**-0.5)

    def get_pitch_emb(self, x, x_mask, tgt=None, factor=1.0):
        out = self.pitch_predictor(x, x_mask)
        if tgt is None:
            out = out * factor
            emb = self.embed_pitch(torch.bucketize(out, self.pitch_bins))
        else:
            emb = self.embed_pitch(torch.bucketize(tgt, self.pitch_bins))
        return out, emb * x_mask

    def get_energy_emb(self, x, x_mask, tgt=None, factor=1.0):
        out = self.energy_predictor(x, x_mask)
        if tgt is None:
            out = out * factor
            emb = self.embed_energy(torch.bucketize(out, self.energy_bins))
        else:
            emb = self.embed_energy(torch.bucketize(tgt, self.energy_bins))
        return out, emb * x_mask

    
    def forward(
        self,
        x,
        x_mask,
        durations,
        pitches,
        energies,
    ):
        # x: B x T x C
        # Get log durations
        logw = torch.log(durations + 1e-8) * x_mask.squeeze(2)

        logw_hat = self.duration_predictor(x, x_mask)
        dur_loss = F.mse_loss(logw_hat, logw, reduction="sum") / torch.sum(x_mask)
        

        log_pitch_out, pitch_emb = self.get_pitch_emb(x, x_mask, pitches)
        x = x + pitch_emb
        log_energy_out, energy_emb = self.get_energy_emb(x, x_mask, energies)
        x = x + energy_emb

        x, out_lens = expand_lengths(x, durations)
        
        pitch_loss = F.mse_loss(log_pitch_out, pitches)
        energy_loss = F.mse_loss(log_energy_out, energies)
        
        outputs = {
            'x_upscaled': x,
            'out_lens': out_lens,
        }
        losses = {
            'dur_loss': dur_loss,
            'pitch_loss': pitch_loss,
            'energy_loss': energy_loss,
        }
        
        return outputs, losses 
    
    @torch.inference_mode()
    def synthesise(
        self,
        x,
        x_mask,
        d_factor=1.0,
        p_factor=1.0,
        e_factor=1.0,
    ):
        # x: B x T x C
        # Get log durations

        logw_hat = self.duration_predictor(x, x_mask)
            
        w = torch.exp(logw_hat) * x_mask.squeeze(2)
        w_ceil = torch.ceil(w) * d_factor
        dur_out = torch.clamp(w_ceil.long(), min=0) 
        
        log_pitch_out, pitch_emb = self.get_pitch_emb(x, x_mask, factor=p_factor)
        log_pitch_out, _ = expand_lengths(log_pitch_out.unsqueeze(2), dur_out)        
        x = x + pitch_emb

        log_energy_out, energy_emb = self.get_energy_emb(x, x_mask, factor=e_factor)
        log_energy_out, _ = expand_lengths(log_energy_out.unsqueeze(2), dur_out)
        x = x + energy_emb
        
        x, out_lens = expand_lengths(x, dur_out)

        return {
            'x_upscaled': x,
            'out_lens': out_lens,
            'dur_pred': dur_out,
            'log_pitch_pred': log_pitch_out,
            'log_energy_pred': log_energy_out,
        }