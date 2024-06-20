import datetime as dt
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fs2 import utils
from fs2.models.baselightningmodule import BaseLightningClass
from fs2.models.components.postnet import Postnet
from fs2.models.components.transformer import FFTransformer
from fs2.models.components.variance_adaptor import VarianceAdaptor
from fs2.utils.model import denormalize, invert_log_norm

log = utils.get_pylogger(__name__)


class FastSpeech2(BaseLightningClass): 
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        n_feats,
        encoder,
        decoder,
        variance_adaptor,
        postnet,
        data_statistics,
        add_postnet=True,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.update_data_statistics(data_statistics)
        
        self.encoder = FFTransformer(
            n_layer=encoder.n_layer,
            n_head=encoder.n_head,
            d_model=encoder.d_model,
            d_head=encoder.d_head,
            d_inner=encoder.d_inner,
            kernel_size=encoder.kernel_size,
            dropout=encoder.dropout,
            dropatt=encoder.dropatt,
            dropemb=encoder.dropemb,
            embed_input=True,
            d_embed=encoder.d_model,
            n_embed=n_vocab
        )
        
       
        if n_spks > 1:    
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim) 

        self.variance_adapter = VarianceAdaptor(variance_adaptor, self.pitch_min, self.pitch_max, self.energy_min, self.energy_max)
        
        self.decoder = FFTransformer(
            n_layer=decoder.n_layer, 
            n_head=decoder.n_head,
            d_model=decoder.d_model,
            d_head=decoder.d_head,
            d_inner=decoder.d_inner,
            kernel_size=decoder.kernel_size,
            dropout=decoder.dropout,
            dropatt=decoder.dropatt,
            dropemb=decoder.dropemb,
            embed_input=False,
            d_embed=decoder.d_model,
        )
        
        self.out_proj = nn.Linear(decoder.d_model, n_feats)
        
        if add_postnet:
            self.postnet = Postnet(
                self.n_feats,
                postnet.n_channels,
                postnet.kernel_size,
                postnet.n_layers,
                postnet.dropout,
            ) 
        else:
            self.postnet = None
        
       
    def forward(self, x, x_lengths, y, y_lengths, durations, pitches, energies, spks=None):
        
        x, x_mask = self.encoder(x, x_lengths)
        
        if self.n_spks > 1:
            spk_emb = self.spk_emb(spks)
            x = x + spk_emb.unsqueeze(1)
        
        # teacher forced durations during training 
        outputs, losses = self.variance_adapter(x, x_mask, durations, pitches, energies)
        
        decoder_out, y_mask = self.decoder(outputs['x_upscaled'], y_lengths)
        
        y_hat = self.out_proj(decoder_out) * y_mask
        
        if self.postnet is not None:
            y_hat_post = y_hat + (self.postnet(y_hat) * y_mask)
            
        
        mel_loss = F.l1_loss(y_hat, rearrange(y, "b c t-> b t c"), reduction="mean")
        postnet_mel_loss = F.l1_loss(y_hat_post, rearrange(y, "b c t-> b t c"), reduction="mean") if self.postnet is not None else 0.0

        losses.update({"mel_loss": mel_loss, "postnet_mel_loss": postnet_mel_loss}) 
        return losses
        
        

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, spks=None, length_scale=1.0, p_factor=1.0, e_factor=1.0, d_factor=1.0):
        # For RTF computation
        t = dt.datetime.now()
        
        x, x_mask = self.encoder(x, x_lengths)
        
        if self.n_spks > 1:
            spk_emb = self.spk_emb(spks)
            x = x + spk_emb.unsqueeze(1)
        
        # teacher forced durations during training 
        var_ada_outputs = self.variance_adapter.synthesise(x, x_mask, d_factor=length_scale, p_factor=p_factor, e_factor=e_factor)
        
        decoder_out, y_mask = self.decoder(var_ada_outputs['x_upscaled'], var_ada_outputs['out_lens'])
        
        y_hat = self.out_proj(decoder_out) * y_mask
        
        if self.postnet is not None:
            y_hat_post = y_hat + (self.postnet(y_hat) * y_mask) 
                        
        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (y_hat_post.shape[1] * 256) 

        return {
            "mel" : denormalize(y_hat_post, self.mel_mean, self.mel_std).transpose(1, 2),
            "decoder_output": denormalize(y_hat, self.mel_mean, self.mel_std).transpose(1, 2),
            "dur_pred": var_ada_outputs["dur_pred"],
            "pitch_pred": denormalize(var_ada_outputs["log_pitch_pred"], self.pitch_mean, self.pitch_std),
            "energy_pred": denormalize(var_ada_outputs["log_energy_pred"], self.energy_mean, self.energy_std),
            "rtf": rtf,
        }