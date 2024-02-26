import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pyworld as pw
import torch
import torchaudio as ta
from lightning import LightningDataModule
from scipy.interpolate import interp1d
from torch.utils.data.dataloader import DataLoader

from fs2.text import text_to_sequence
from fs2.utils.audio import mel_spectrogram
from fs2.utils.model import normalize
from fs2.utils.utils import intersperse, to_torch, trim_or_pad_to_target_length


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        cleaners,
        add_blank,
        n_spks,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        generate_properties,
        processed_folder_path
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        self.trainset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.generate_properties,
            self.hparams.processed_folder_path
        )
        self.validset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.generate_properties,
            self.hparams.processed_folder_path
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):  # pylint: disable=no-self-use
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        n_spks,
        cleaners,
        add_blank=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_statistics=None,
        seed=None,
        generate_properties=True,
        processed_folder_path=None
    ):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.n_spks = n_spks
        self.cleaners = cleaners
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.data_statistics = data_statistics
        self.generate_properties = generate_properties
        self.processed_folder_path = processed_folder_path

        random.seed(seed)
        random.shuffle(self.filepaths_and_text)
       
    def load_durations(self, filepath, text):
        durs = np.load(Path(self.processed_folder_path) / 'durations' / Path(Path(filepath).stem).with_suffix(".npy")).astype(int)
        assert len(durs) == len(text), f"Length of durations {len(durs)} and text {len(text)} do not match"
        return durs
        
     
    def get_pitch(self, filepath, phoneme_durations, log_scale=True):
        _waveform, _sr = ta.load(filepath)
        _waveform = _waveform.squeeze(0).double().numpy() 
        assert _sr == self.sample_rate, f"Sample rate mismatch => Found: {_sr} != {self.sample_rate} = Expected"
        
        pitch, t = pw.dio(
            _waveform, self.sample_rate, frame_period=self.hop_length / self.sample_rate * 1000
        )
        pitch = pw.stonemask(_waveform, pitch, t, self.sample_rate)
        # A cool function taken from fairseq 
        # https://github.com/facebookresearch/fairseq/blob/3f0f20f2d12403629224347664b3e75c13b2c8e0/examples/speech_synthesis/data_utils.py#L99
        pitch = trim_or_pad_to_target_length(pitch, sum(phoneme_durations))
        
        # Interpolate to cover the unvoiced segments as well 
        nonzero_ids = np.where(pitch != 0)[0]

        interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
        pitch = interp_fn(np.arange(0, len(pitch)))
        
        # Compute phoneme-wise average 
        d_cumsum = np.cumsum(np.concatenate([np.array([0]), phoneme_durations]))
        pitch = np.array(
            [
                np.mean(pitch[d_cumsum[i-1]: d_cumsum[i]])
                for i in range(1, len(d_cumsum))
            ]
        )
        assert len(pitch) == len(phoneme_durations)
        
        if log_scale:
            # In fairseq they do it
            pitch = np.log(pitch + 1)
        
        return pitch
    
    def mean_phoneme_energy(self, energy, phoneme_durations, log_scale=True):
        energy = trim_or_pad_to_target_length(energy, sum(phoneme_durations))
        d_cumsum = np.cumsum(np.concatenate([np.array([0]), phoneme_durations]))
        energy = np.array(
            [
                np.mean(energy[d_cumsum[i - 1]: d_cumsum[i]])
                for i in range(1, len(d_cumsum))
            ]
        )
        assert len(energy) == len(phoneme_durations)
        
        if log_scale:
            # In fairseq they do it
            energy = np.log(energy + 1)
        
        return energy

    def get_datapoint(self, filepath_and_text):
        if self.n_spks > 1:
            filepath, spk, text = (
                filepath_and_text[0],
                int(filepath_and_text[1]),
                filepath_and_text[2],
            )
        else:
            filepath, text = filepath_and_text[0], filepath_and_text[1]
            spk = None
            
        if self.generate_properties:
            text = self.get_text(text, add_blank=self.add_blank)
            phoneme_durations = self.load_durations(filepath, text)
            mel, energy = self.get_mel(filepath)
            energy = self.mean_phoneme_energy(energy.squeeze().cpu().numpy(), phoneme_durations)
            pitch = self.get_pitch(filepath, phoneme_durations)
            # Do not normalise them in this case as this is supposed to be called by
            # python fs2/utils/preprocess.py -i ljspeech
        else:
            text = self.get_text(text, add_blank=self.add_blank)
            phoneme_durations = self.load_durations(filepath, text)
            assert len(phoneme_durations) == len(text)
            #! TODO: implement duration and pitch loading
            pitch = np.load(Path(self.processed_folder_path) / 'pitch' / Path(Path(filepath).stem).with_suffix(".npy"))
            pitch = normalize(pitch, self.data_statistics['pitch_mean'], self.data_statistics['pitch_std'])
            assert len(pitch) == len(text)
            mel = np.load(Path(self.processed_folder_path) / 'mel' / Path(Path(filepath).stem).with_suffix(".npy"))
            mel = normalize(mel, self.data_statistics['mel_mean'], self.data_statistics['mel_std'])
            assert mel.shape[-1] == sum(phoneme_durations)
            energy = np.load(Path(self.processed_folder_path) / 'energy' / Path(Path(filepath).stem).with_suffix(".npy")) 
            energy = normalize(energy, self.data_statistics['energy_mean'], self.data_statistics['energy_std'])
            assert len(energy) == len(text)

        return {"x": text, "y": mel, "spk": spk, 'filepath': filepath, 
                'energy': energy, 'pitch': pitch, 'duration': phoneme_durations}

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel, energy = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        )
        return mel, energy 

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, self.cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.filepaths_and_text[index])
        return datapoint

    def __len__(self):
        return len(self.filepaths_and_text)


class TextMelBatchCollate:
    def __init__(self, n_spks):
        self.n_spks = n_spks

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        pitches = torch.zeros((B, x_max_length), dtype=torch.float)
        energies = torch.zeros((B, x_max_length), dtype=torch.float)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)
        
        
        y_lengths, x_lengths = [], []
        spks = []
        
        filepaths = [] 
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = to_torch(y_, torch.float32)
            x[i, : x_.shape[-1]] = to_torch(x_, torch.long)
            spks.append(item["spk"])
            
            pitches[i, : item["pitch"].shape[-1]] = to_torch(item["pitch"], torch.float)
            energies[i, : item["energy"].shape[-1]] = to_torch(item["energy"], torch.float)
            durations[i, : item["duration"].shape[-1]] = to_torch(item["duration"], torch.float)
            filepaths.append(item['filepath'])

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None

        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            'pitches': pitches,
            'energies': energies,
            'durations': durations,
            'filepaths': filepaths
        } 