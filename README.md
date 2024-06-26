<div align="center">

# BetterFastSpeech 2


[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)


</div>

It is the ordinary FastSpeech 2 architecture with some modifications. I just wanted to make the code base better and more readable. And finally have an open source implementation of [FastSpeech 2](https://arxiv.org/abs/2006.04558) that doesn't sounds bad and is easier to hack and work with.

If you like this you will love [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)

Changes from the original architecture: 
  - Instead of using MFA, I obtained alignment from a pretrained Matcha-TTS model. 
    - To save myself from the pain of setting up and training MFA
  - Used IPA phonemes with blanks in between phones.
  - No LR decay
  - Duration prediction in log domain
  - Everyone seems to be using the postnet from Tacotron 2; I've used it as well.


## Installation

1. Create an environment (suggested but optional)

```
conda create -n betterfs2 python=3.10 -y
conda activate betterfs2
```

2. Install from source

```bash
pip install git+https://github.com/shivammehta25/BetterFastSpeech2.git
cd BetterFastSpeech2
pip install -e .
```

3. Run CLI / gradio app / jupyter notebook

```bash
# This will download the required models
betterfs2 --text "<INPUT TEXT>"
```

or

```bash
betterfs2-tts-app
```
or open `synthesis.ipynb` on jupyter notebook

## Train with your own dataset
Let's assume we are training with LJ Speech

1. Download the dataset from [here](https://keithito.com/LJ-Speech-Dataset/), extract it to `data/LJSpeech-1.1`, and prepare the file lists to point to the extracted data like for [item 5 in the setup of the NVIDIA Tacotron 2 repo](https://github.com/NVIDIA/tacotron2#setup).


2. [Train a Matcha-TTS model to extract durations or if you have a pretrained model, you can use that as well.](https://github.com/shivammehta25/Matcha-TTS/wiki/Improve-GPU-utilisation-by-extracting-phoneme-alignments)

Your data directory should look like:
```bash
data/
└── LJSpeech-1.1
    ├── durations/ # Here
    ├── metadata.csv
    ├── README
    ├── test.txt
    ├── train.txt
    ├── val.txt
    └── wavs/
```

3. Clone and enter the BetterFastSpeech2 repository

```bash
git clone https://github.com/shivammehta25/BetterFastSpeech2.git
cd BetterFastSpeech2 
```

4. Install the package from source

```bash
pip install -e .
```

5. Go to `configs/data/ljspeech.yaml` and change

```yaml
train_filelist_path: data/LJSpeech-1.1/train.txt
valid_filelist_path: data/LJSpeech-1.1/val.txt
```

5. Generate normalisation statistics with the yaml file of dataset configuration

```bash
python fs2/utils/preprocess.py -i ljspeech
# Output:
#{'pitch_min': 67.836174, 'pitch_max': 578.637146, 'pitch_mean': 207.001846, 'pitch_std': 52.747742, 'energy_min': 0.084354, 'energy_max': 190.849121, 'energy_mean': 21.330254, 'energy_std': 17.663319, 'mel_mean': -5.554245, 'mel_std': 2.059021}
```

Update these values in `configs/data/ljspeech.yaml` under `data_statistics` key.

```bash
data_statistics:  # Computed for ljspeech dataset
    pitch_min: 67.836174 
    pitch_max: 792.962036
    pitch_mean: 211.046158
    pitch_std: 53.012085
    energy_min: 0.023226
    energy_max: 241.037918
    energy_mean: 21.821531
    energy_std: 18.17124
    mel_mean: -5.517035
    mel_std: 2.064413
```

to the paths of your train and validation filelists.

6. Run the training script

```bash
python fs2/train.py experiment=ljspeech
```

- for multi-gpu training, run

```bash
python fs2/train.py experiment=ljspeech trainer.devices=[0,1]
```

7. Synthesise from the custom trained model

```bash
betterfs2 --text "<INPUT TEXT>" --checkpoint_path <PATH TO CHECKPOINT>
```


## Citation information

If you use our code or otherwise find this work useful, please cite our paper:

```text
@inproceedings{mehta2024matcha,
  title={Matcha-{TTS}: A fast {TTS} architecture with conditional flow matching},
  author={Mehta, Shivam and Tu, Ruibo and Beskow, Jonas and Sz{\'e}kely, {\'E}va and Henter, Gustav Eje},
  booktitle={Proc. ICASSP},
  year={2024}
}
```

## Acknowledgements

Since this code uses [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template), you have all the powers that come with it.

Other source code we would like to acknowledge:

- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS): Base TTS from which we get alignments.
- [FastPitch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch): For transformer implementation
- [FastSpeech 2](https://github.com/ming024/FastSpeech2): For variance predictor implementations
