"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""
import inspect
from abc import ABC
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from fs2 import utils
from fs2.utils.model import (denormalize, expand_lengths, invert_log_norm,
                             normalize)
from fs2.utils.utils import plot_line, plot_tensor, save_figure_to_numpy

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
        pitch_max = normalize(torch.tensor(data_statistics["pitch_max"]), self.pitch_mean, self.pitch_std)
        energy_min = normalize(torch.tensor(data_statistics["energy_min"]), self.energy_mean, self.energy_std)
        energy_max = normalize(torch.tensor(data_statistics["energy_max"]), self.energy_mean, self.energy_std)
        
        self.register_buffer("pitch_min", pitch_min)
        self.register_buffer("pitch_max", pitch_max)
        self.register_buffer("energy_min", energy_min)
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
                    y = denormalize(one_batch["y"][i].unsqueeze(0).to(self.device), self.mel_mean, self.mel_std)[:, :, :one_batch["y_lengths"][i]]
                    durations = one_batch["durations"][i].to(self.device)[:one_batch["x_lengths"][i]]

                    original_pitch = one_batch["pitches"][i].unsqueeze(0).to(self.device)[:, :one_batch["x_lengths"][i]]
                    original_pitch, _ = expand_lengths(original_pitch.unsqueeze(2), durations.unsqueeze(0))
                    original_pitch = denormalize(original_pitch, self.pitch_mean, self.pitch_std)
                    
                    original_energy = one_batch["energies"][i].unsqueeze(0).to(self.device)[:, :one_batch["x_lengths"][i]]
                    original_energy, _ = expand_lengths(original_energy.unsqueeze(2), durations.unsqueeze(0))
                    original_energy = denormalize(original_energy, self.energy_mean, self.energy_std)
                     
                    self.logger.experiment.add_image(
                        f"original/mel_{i}",
                        self.plot_mel([(y.squeeze().cpu().numpy(), original_pitch.cpu().squeeze(), original_energy.cpu().squeeze())], [f"Data_{i}"]),
                        self.current_epoch,
                        dataformats="HWC",
                    )

            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None
                output = self.synthesise(x[:, :x_lengths], x_lengths, spks=spks)
                decoder_output, y_pred = output["decoder_output"], output["mel"]
                pitch_pred, energy_pred = output["pitch_pred"], output["energy_pred"]
 
                self.logger.experiment.add_image(
                    f"dec_output/{i}",
                    plot_tensor(decoder_output.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )

                self.logger.experiment.add_image(
                    f"generated/mel_{i}",
                    self.plot_mel([(y_pred.squeeze().cpu().numpy(), pitch_pred.cpu().squeeze(), energy_pred.cpu().squeeze())], [f"Generated_{i}"]),
                    self.current_epoch,
                    dataformats="HWC",
                )
                

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})



    def plot_mel(self, data, titles, show=False):
        fig, axes = plt.subplots(len(data), 1, squeeze=False)
        if titles is None:
            titles = [None for i in range(len(data))]
        
        pitch_max = denormalize(self.pitch_max, self.pitch_mean, self.pitch_std).cpu().item()
        energy_min = denormalize(self.energy_min, self.energy_mean, self.energy_std).cpu().item()
        energy_max = denormalize(self.energy_max, self.energy_mean, self.energy_std).cpu().item()

        def add_axis(fig, old_ax):
            ax = fig.add_axes(old_ax.get_position(), anchor="W")
            ax.set_facecolor("None")
            return ax

        for i in range(len(data)):
            mel, pitch, energy = data[i]
            axes[i][0].imshow(mel, origin="lower")
            axes[i][0].set_aspect(2.5, adjustable="box")
            axes[i][0].set_ylim(0, mel.shape[0])
            axes[i][0].set_title(titles[i], fontsize="medium")
            axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
            axes[i][0].set_anchor("W")

            ax1 = add_axis(fig, axes[i][0])
            ax1.plot(pitch, color="tomato")
            ax1.set_xlim(0, mel.shape[1])
            ax1.set_ylim(0, pitch_max)
            ax1.set_ylabel("F0", color="tomato")
            ax1.tick_params(
                labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
            )

            ax2 = add_axis(fig, axes[i][0])
            ax2.plot(energy, color="darkviolet")
            ax2.set_xlim(0, mel.shape[1])
            ax2.set_ylim(energy_min, energy_max)
            ax2.set_ylabel("Energy", color="darkviolet")
            ax2.yaxis.set_label_position("right")
            ax2.tick_params(
                labelsize="x-small",
                colors="darkviolet",
                bottom=False,
                labelbottom=False,
                left=False,
                labelleft=False,
                right=True,
                labelright=True,
            )
        fig.canvas.draw()
        if show:
            plt.show()
            return

        data = save_figure_to_numpy(fig)
        plt.close()
        return data 