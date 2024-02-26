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
    
class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: Callable = nn.SiLU(),
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = act_fn 

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = act_fn

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample

class DurationPredictorNetworkWithTimeStep(nn.Module):
    """Similar architecture but with a time embedding support"""

    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.time_embeddings = SinusoidalPosEmb(filter_channels)
        self.time_mlp = TimestepEmbedding(
            in_channels=filter_channels,
            time_embed_dim=filter_channels,
            act_fn=nn.SiLU(),
        )

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask, enc_outputs, t):
        t = self.time_embeddings(t)
        t = self.time_mlp(t).unsqueeze(-1)

        x = pack([x, enc_outputs], "b * t")[0]

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = x + t
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = x + t
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask

    
class VarianceAdaptor(nn.Module):
    def __init__(self, args, pitch_min, pitch_max, energy_min, energy_max):
        super().__init__()
        self.args = args
        self.duration_predictor_type = args.duration_prediction_type
        if self.duration_predictor_type == "det":
            self.duration_predictor = VariancePredictor(args)
        elif self.duration_predictor_type == "fm":
            self.duration_predictor = DurationPredictorNetworkWithTimeStep(
            1 + args.d_model ,  # 1 for the durations and n_channels for encoder outputs
            args.hidden_dim,
            args.kernel_size,
            args.dropout,
            )
            self.sigma_min = 1e-4
            self.n_steps = 10
        else:
            raise ValueError(f"Unknown duration predictor type: {self.duration_predictor_type}")

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
    
    def flow_matching_loss(self, x1, enc_outputs, mask):
            x1 = rearrange(x1, "b t -> b 1 t")
            enc_outputs = rearrange(enc_outputs, "b t c -> b c t")
            mask = rearrange(mask, "b t 1-> b 1 t")
            b, _, t = enc_outputs.shape

            # random timestep
            t = torch.rand([b, 1, 1], device=enc_outputs.device, dtype=enc_outputs.dtype)
            # sample noise p(x_0)
            z = torch.randn_like(x1)

            y = (1 - (1 - self.sigma_min) * t) * z + t * x1
            u = x1 - (1 - self.sigma_min) * z

            loss = F.mse_loss(self.duration_predictor(y, mask, enc_outputs, t.squeeze()), u, reduction="sum") / (
                torch.sum(mask) * u.shape[1]
            )
            return loss
        
    def fm_synthesise(self, enc_outputs, mask, n_timesteps=None, temperature=1):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        enc_outputs = rearrange(enc_outputs, "b t c -> b c t")
        mask = rearrange(mask, "b t 1-> b 1 t")
        
        if n_timesteps is None:
            n_timesteps = self.n_steps

        b, _, t = enc_outputs.shape
        z = torch.randn((b, 1, t), device=enc_outputs.device, dtype=enc_outputs.dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=enc_outputs.device)
        return self.solve_euler(z, t_span=t_span, enc_outputs=enc_outputs, mask=mask)

    def solve_euler(self, x, t_span, enc_outputs, mask):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.duration_predictor(x, mask, enc_outputs, t)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].squeeze(1)

    
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

        if self.duration_predictor_type == "det":
            logw_hat = self.duration_predictor(x, x_mask)
            dur_loss = F.mse_loss(logw_hat, logw, reduction="sum") / torch.sum(x_mask)
        elif self.duration_predictor_type == "fm":
            dur_loss = self.flow_matching_loss(logw, x, x_mask)
        

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

        if self.duration_predictor_type == "det":
            logw_hat = self.duration_predictor(x, x_mask)
        elif self.duration_predictor_type == "fm":
            logw_hat = self.fm_synthesise(x, x_mask)
            
        w = torch.exp(logw_hat) * x_mask.squeeze(2)
        w_ceil = torch.round(w) * d_factor
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