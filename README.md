# Problems using deep generative models for probabilistic audio source separation

[Maurice Frank](https://scholar.google.com/citations?user=jCHjpIsAAAAJ), [Maximilian Ilse](https://scholar.google.com/citations?user=KNJIRGkAAAAJ)  
[[Pre-Print](https://arxiv.org/abs/2011.01761)] [[Poster](https://raw.githubusercontent.com/morris-frank/thesis-tex/master/poster.pdf)]

## Installation

```bash
pip install -r requirements.txt
```

## Training

| Command                                                 | For                                                                    |
| ------------------------------------------------------- | ---------------------------------------------------------------------- |
| `./train.py prior_time --batch_size N --gpu GPU`        | train the flow priors for the toy data                                 |
| `./train.py prior_time -musdb --batch_size N --gpu GPU` | train the flow priors for musdb18                                      |
| `./train.py wavenet --batch_size N --gpu GPU`           | train the autoregressive priors for the toy data                       |
| `./train.py wavenet -musdb --batch_size N --gpu GPU`    | train the autoregressive priors for musdb18                            |
| `./make.py eval "Dec18-*"`                              | evaluate the trained model checkpoint matching the given globbing name |
