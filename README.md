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



## ONNX support

> Have to update but it is most likely the same as Matcha-TTS

> Special thanks to [@mush42](https://github.com/mush42) for implementing ONNX export and inference support.

It is possible to export Matcha checkpoints to [ONNX](https://onnx.ai/), and run inference on the exported ONNX graph.

### ONNX export

To export a checkpoint to ONNX, first install ONNX with

```bash
pip install onnx
```

then run the following:

```bash
python3 -m matcha.onnx.export matcha.ckpt model.onnx --n-timesteps 5
```

Optionally, the ONNX exporter accepts **vocoder-name** and **vocoder-checkpoint** arguments. This enables you to embed the vocoder in the exported graph and generate waveforms in a single run (similar to end-to-end TTS systems).

**Note** that `n_timesteps` is treated as a hyper-parameter rather than a model input. This means you should specify it during export (not during inference). If not specified, `n_timesteps` is set to **5**.

**Important**: for now, torch>=2.1.0 is needed for export since the `scaled_product_attention` operator is not exportable in older versions. Until the final version is released, those who want to export their models must install torch>=2.1.0 manually as a pre-release.

### ONNX Inference

To run inference on the exported model, first install `onnxruntime` using

```bash
pip install onnxruntime
pip install onnxruntime-gpu  # for GPU inference
```

then use the following:

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs
```

You can also control synthesis parameters:

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --temperature 0.4 --speaking_rate 0.9 --spk 0
```

To run inference on **GPU**, make sure to install **onnxruntime-gpu** package, and then pass `--gpu` to the inference command:

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --gpu
```

If you exported only Matcha to ONNX, this will write mel-spectrogram as graphs and `numpy` arrays to the output directory.
If you embedded the vocoder in the exported graph, this will write `.wav` audio files to the output directory.

If you exported only Matcha to ONNX, and you want to run a full TTS pipeline, you can pass a path to a vocoder model in `ONNX` format:

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --vocoder hifigan.small.onnx
```

This will write `.wav` audio files to the output directory.

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

- [Coqui-TTS](https://github.com/coqui-ai/TTS/tree/dev): For helping me figure out how to make cython binaries pip installable and encouragement
- [Hugging Face Diffusers](https://huggingface.co/): For their awesome diffusers library and its components
- [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS): For the monotonic alignment search source code
- [torchdyn](https://github.com/DiffEqML/torchdyn): Useful for trying other ODE solvers during research and development
- [labml.ai](https://nn.labml.ai/transformers/rope/index.html): For the RoPE implementation
