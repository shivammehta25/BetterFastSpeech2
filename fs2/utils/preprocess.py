r"""
The file creates a pickle file where the values needed for loading of dataset is stored and the model can load it
when needed.

Parameters from hparam.py will be used
"""
import argparse
import json
import os
import sys
from pathlib import Path

import lightning
import numpy as np
import rootutils
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, open_dict
from torch import nn
from tqdm.auto import tqdm

from fs2.data.text_mel_datamodule import TextMelDataModule
from fs2.utils.logging_utils import pylogger
from fs2.utils.utils import to_numpy

log = pylogger.get_pylogger(__name__)


@torch.inference_mode()
def generate_preprocessing_files(dataset: torch.utils.data.Dataset, output_folder: Path, cfg: DictConfig, save_stats=True):
    """Generate durations from the model for each datapoint and save it in a folder

    Args:
        data_loader (torch.utils.data.DataLoader): Dataloader
        model (nn.Module): MatchaTTS model
        device (torch.device): GPU or CPU
    """
    x_lengths = 0

    # Pitch stats
    pitch_min = float("inf")
    pitch_max = -float("inf")
    pitch_sum = 0
    pitch_sq_sum = 0
    
    # Energy stats
    energy_min = float("inf")
    energy_max = -float("inf")
    energy_sum = 0
    energy_sq_sum = 0
    
    # Mel stats
    mel_sum = 0
    mel_sq_sum = 0
    total_mel_len = 0
    
    processed_folder_name = output_folder 
    assert (processed_folder_name/ "durations").exists(), "Durations folder not found, it must be generated beforehand for this script to work"
    pitch_folder, energy_folder, mel_folder = init_folders(processed_folder_name)
    
    # Benefit of doing it over batch is the added speed due to multiprocessing
    for batch in tqdm(dataset, desc="🍵 Preprocessing durations 🍵"):
        # Get pre generated durations with Matcha-TTS
        for i in range(batch['x'].shape[0]):
            filname = Path(batch['filepaths'][i]).stem
            inp_len = batch['x_lengths'][i]
            mel_len = batch['y_lengths'][i]
            pitch = batch['pitches'][i][:inp_len]
            pitch_min = min(pitch_min, torch.min(pitch).item())
            pitch_max = max(pitch_max, torch.max(pitch).item())
            
            np.save(pitch_folder / f"{filname}.npy", to_numpy(pitch))
            energy = batch['energies'][i][:inp_len]
            energy_min = min(energy_min, torch.min(energy).item())
            energy_max = max(energy_max, torch.max(energy).item())
            np.save(energy_folder / f"{filname}.npy", to_numpy(energy))
            mel_spec = batch['y'][i][:, :mel_len]
            np.save(mel_folder / f"{filname}.npy", to_numpy(mel_spec))
            
            # normalisation statistics
            pitch_sum += torch.sum(pitch)
            pitch_sq_sum += torch.sum(torch.pow(pitch, 2))
            energy_sum += torch.sum(energy)
            energy_sq_sum += torch.sum(torch.pow(energy, 2))
            x_lengths += inp_len
            
            mel_sum += torch.sum(mel_spec)
            mel_sq_sum += torch.sum(mel_spec ** 2)
            total_mel_len += mel_len
    
    # Save normalisation statistics
    pitch_mean = pitch_sum / x_lengths
    pitch_std = torch.sqrt((pitch_sq_sum / x_lengths) - torch.pow(pitch_mean, 2))
    
    energy_mean = energy_sum / x_lengths
    energy_std = torch.sqrt((energy_sq_sum / x_lengths) - torch.pow(energy_mean,2))
    
    mel_mean = mel_sum / (total_mel_len * cfg['n_feats'])
    mel_std = torch.sqrt((mel_sq_sum / (total_mel_len * cfg['n_feats'])) - torch.pow(mel_mean, 2))


    stats = {
                "pitch_min": round(pitch_min, 6),
                "pitch_max": round(pitch_max, 6),
                "pitch_mean": round(pitch_mean.item(), 6),
                "pitch_std": round(pitch_std.item(), 6),
                "energy_min": round(energy_min, 6),
                "energy_max": round(energy_max, 6),
                "energy_mean": round(energy_mean.item(), 6),
                "energy_std": round(energy_std.item(), 6),
                "mel_mean": round(mel_mean.item(), 6),
                "mel_std": round(mel_std.item(), 6),
    }

    print(stats)
    if save_stats:
        with open(processed_folder_name / "stats.json", "w") as f:
            json.dump(stats,f, indent=4) 
    else:
        print("Stats not saved!")
    
    print("[+] Done! features saved to: ", processed_folder_name)

def init_folders(processed_folder_name):
    pitch_folder = processed_folder_name / "pitch"
    energy_folder = processed_folder_name / "energy"
    mel_folder = processed_folder_name / "mel"
    pitch_folder.mkdir(parents=True, exist_ok=True)
    energy_folder.mkdir(parents=True, exist_ok=True)
    mel_folder.mkdir(parents=True, exist_ok=True)
    return pitch_folder,energy_folder, mel_folder



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-config",
        type=str,
        default="vctk.yaml",
        help="The name of the yaml config file under configs/data",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default="32",
        help="Can have increased batch size for faster computation",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        required=False,
        help="force overwrite the file",
    )

    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        default=None,
        help="Output folder to save the data statistics",
    )
    args = parser.parse_args()

    with initialize(version_base="1.3", config_path="../../configs/data"):
        cfg = compose(config_name=args.input_config, return_hydra_config=True, overrides=[])

    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    with open_dict(cfg):
        del cfg["hydra"]
        del cfg["_target_"]
        cfg["seed"] = 1234
        cfg["batch_size"] = args.batch_size
        cfg["train_filelist_path"] = str(os.path.join(root_path, cfg["train_filelist_path"]))
        cfg["valid_filelist_path"] = str(os.path.join(root_path, cfg["valid_filelist_path"]))
        cfg['generate_properties'] = True
        # Remove this after testing let the multiprocessing do its job 
        # cfg['num_workers'] = 0

    if args.output_folder is not None:
        output_folder = Path(args.output_folder)
    else:
        output_folder = Path(cfg["train_filelist_path"]).parent

    # if os.path.exists(output_folder) and not args.force:
    #     print("Folder already exists. Use -f to force overwrite")
    #     sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing: {cfg['name']} from training filelist: {cfg['train_filelist_path']}")
       
 
    text_mel_datamodule = TextMelDataModule(**cfg)
    text_mel_datamodule.setup()
    try:
        print("Computing stats for training set if exists...")
        train_dataloader = text_mel_datamodule.train_dataloader()
        generate_preprocessing_files(train_dataloader, output_folder, cfg, save_stats=True)
    except lightning.fabric.utilities.exceptions.MisconfigurationException:
        print("No training set found")

    try:
        print("Computing stats for validation set if exists...")
        val_dataloader = text_mel_datamodule.val_dataloader()
        generate_preprocessing_files(val_dataloader, output_folder, cfg)
    except lightning.fabric.utilities.exceptions.MisconfigurationException:
        print("No validation set found")

    try:
        print("Computing stats for test set if exists...")
        test_dataloader = text_mel_datamodule.test_dataloader()
        generate_preprocessing_files(test_dataloader, output_folder, cfg)
    except lightning.fabric.utilities.exceptions.MisconfigurationException:
        print("No test set found")

    print(f"[+] Done! features saved to: {output_folder}")


if __name__ == "__main__":
    # Helps with generating durations for the dataset to train other architectures
    # that cannot learn to align due to limited size of dataset
    # Example usage:
    # python python matcha/utils/get_durations_from_trained_model.py -i ljspeech.yaml -c pretrained_model
    # This will create a folder in data/processed_data/durations/ljspeech with the durations
    main()
