# Problems using deep generative models for probabilistic audio source separation

![Python](https://img.shields.io/badge/Python-3.7.5-blue?logo=python)
![GitHub](https://img.shields.io/github/license/morris-frank/unsupervised-source-separation)
[![Torch 1.5.0](https://img.shields.io/badge/PyTorch-1.5.0-orange)](https://pytorch.org/)

[Maurice Frank](https://scholar.google.com/citations?user=jCHjpIsAAAAJ), [Maximilian Ilse](https://scholar.google.com/citations?user=KNJIRGkAAAAJ)  
[[Pre-Print](https://arxiv.org/abs/2011.01761)] [[Poster](https://raw.githubusercontent.com/morris-frank/thesis-tex/master/poster.pdf)]

## Abstract

> Recent advancements in deep generative modeling make it possible to learn prior distributions from complex data that subsequently can be used for Bayesian inference. However, we find that distributions learned by deep generative models for audio signals do not exhibit the right properties that are necessary for tasks like audio source separation using a probabilistic approach. We observe that the learned prior distributions are either discriminative and extremely peaked or smooth and non-discriminative. We quantify this behavior for two types of deep generative models on two audio datasets.

## Installation

```bash
pip install -r requirements.txt
```

Most importantly they are

```
python>=3.7.5
torch~=1.5.0
torchaudio>=0.5.0
```

## Training

| Command                                                 | For                                                                    |
| ------------------------------------------------------- | ---------------------------------------------------------------------- |
| `./train.py prior_time --batch_size N --gpu GPU`        | train the flow priors for the toy data                                 |
| `./train.py prior_time -musdb --batch_size N --gpu GPU` | train the flow priors for musdb18                                      |
| `./train.py wavenet --batch_size N --gpu GPU`           | train the autoregressive priors for the toy data                       |
| `./train.py wavenet -musdb --batch_size N --gpu GPU`    | train the autoregressive priors for musdb18                            |
| `./make.py eval "Dec18-*"`                              | evaluate the trained model checkpoint matching the given globbing name |
