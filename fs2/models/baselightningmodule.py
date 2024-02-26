"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""
import inspect
from abc import ABC
from typing import Any, Dict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from fs2 import utils
from fs2.utils.model import (denormalize, expand_lengths, invert_log_norm,
                             normalize)
from fs2.utils.utils import plot_line, plot_tensor

log = utils.get_pylogger(__name__)


class BaseLightningClass(LightningModule, ABC):
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            raise ValueError(f"data_statistics are not computed. \
                             Please run python fs2/utils/preprocess.py -i <dataset.yaml> \
                             to get statistics and update them in data_statistics field.")
        
        self.register_buffer("pitch_mean", torch.tensor(data_statistics["pitch_mean"]))
        self.register_buffer("pitch_std", torch.tensor(data_statistics["pitch_std"]))
        
        self.register_buffer("energy_mean", torch.tensor(data_statistics["energy_mean"]))
        self.register_buffer("energy_std", torch.tensor(data_statistics["energy_std"]))
         
        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))
        
        pitch_min = normalize(torch.tensor(data_statistics["pitch_min"]), self.pitch_mean, self.pitch_std)
        self.register_buffer("pitch_min", pitch_min)
        pitch_max = normalize(torch.tensor(data_statistics["pitch_max"]), self.pitch_mean, self.pitch_std)
        self.register_buffer("pitch_max", pitch_max)
        energy_min = normalize(torch.tensor(data_statistics["energy_min"]), self.energy_mean, self.energy_std)
        self.register_buffer("energy_min", energy_min)
        energy_max = normalize(torch.tensor(data_statistics["energy_max"]), self.energy_mean, self.energy_std)
        self.register_buffer("energy_max", energy_max)

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # Manage last epoch for exponential schedulers
            if "last_epoch" in inspect.signature(self.hparams.scheduler.scheduler).parameters:
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            scheduler = self.hparams.scheduler.scheduler(**scheduler_args)
            scheduler.last_epoch = current_epoch
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler.lightning_args.interval,
                    "frequency": self.hparams.scheduler.lightning_args.frequency,
                    "name": "learning_rate",
                },
            }

        return {"optimizer": optimizer}

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]
        durations, pitches, energies = batch["durations"], batch["pitches"], batch["energies"]
        

        losses = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            durations=durations,
            pitches=pitches,
            energies=energies,
            spks=spks,
        )
        return losses
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        for loss in loss_dict:
            self.log(
                f"sub_loss/train_{loss}",
                loss_dict[loss],
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        total_loss = sum(loss_dict.values())
        self.log(
            "loss/train",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        for loss in loss_dict:
            self.log(
                f"sub_loss/val_{loss}",
                loss_dict[loss],
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        total_loss = sum(loss_dict.values())
        self.log(
            "loss/val",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": total_loss, "log": loss_dict}

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )
                    durations = one_batch["durations"][i].to(self.device)[:one_batch["x_lengths"][i]]

                    # Plot pitch
                    original_pitch = one_batch["pitches"][i].unsqueeze(0).to(self.device)[:, :one_batch["x_lengths"][i]]
                    original_pitch = invert_log_norm(original_pitch, self.pitch_mean, self.pitch_std)
                    original_pitch, _ = expand_lengths(original_pitch.unsqueeze(2), durations.unsqueeze(0))
                    
                    self.logger.experiment.add_image(
                        f"original/pitch_{i}",
                        plot_line(
                            original_pitch.squeeze().cpu(),
                            min_value=invert_log_norm(self.pitch_min, self.pitch_mean, self.pitch_std).cpu().item(),
                            max_value=invert_log_norm(self.pitch_max, self.pitch_mean, self.pitch_std).cpu().item()
                        ),
                        self.current_epoch,
                        dataformats="HWC",
                    ) 
                    
                    # Plot energy
                    original_energy = one_batch["energies"][i].unsqueeze(0).to(self.device)[:, :one_batch["x_lengths"][i]]
                    original_energy = invert_log_norm(original_energy, self.energy_mean, self.energy_std)
                    original_energy, _ = expand_lengths(original_energy.unsqueeze(2), durations.unsqueeze(0))
                    
                    self.logger.experiment.add_image(
                        f"original/energy_{i}",
                        plot_line(
                            original_energy.squeeze().cpu(),
                            min_value=invert_log_norm(self.energy_min, self.energy_mean, self.energy_std).cpu().item(),
                            max_value=invert_log_norm(self.energy_max, self.energy_mean, self.energy_std).cpu().item()
                        ),
                        self.current_epoch,
                        dataformats="HWC",
                    ) 

            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None
                output = self.synthesise(x[:, :x_lengths], x_lengths, spks=spks)
                decoder_output, y_pred = output["decoder_output"], output["y_pred"]
                
                self.logger.experiment.add_image(
                    f"generated/dec_output_{i}",
                    plot_tensor(decoder_output.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"generated/final_mel_{i}",
                    plot_tensor(y_pred.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                
                
                
                energy_pred = output["pitch_pred"], output["energy_pred"]
                self.logger.experiment.add_image(
                    f"pitch/gen_{i}",
                    plot_line(
                        pitch_pred.squeeze().cpu(),
                        min_value=denormalize(self.pitch_min, self.pitch_mean, self.pitch_std),
                        max_value=denormalize(self.pitch_max, self.pitch_mean, self.pitch_std)
                    ),
                    self.current_epoch,
                    dataformats="HWC",
                )

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})
