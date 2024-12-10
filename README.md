# Batched Spectral Attention (BSA)

This repository contains the official implementation of the paper: [Introducing Spectral Attention for Long-Range Dependency in Time Series Forecasting](https://openreview.net/forum?id=dxyNVEBQMp).

## Usage

To run the code, execute the following command in the terminal:

```bash
python3 run_exp/itransformer.py --cuda 0 --data 0 --len 0 --basic --model 0
```

- `--cuda`: GPU index
- `--data`: Dataset index (refer to `run_exp/itransformer.py`)
- `--len`: Output length index (96 / 192 / 336 / 720)
- `--basic`: Include this flag to run the baseline. Omit for finetuning only.
- `--model`: Finetuning model index (refer to `run_exp/itransformer.py`). Omit for baseline training only.

## Key Code Files

- `layers/Momentum.py`: Spectral Attention (SA) code
- `layers/Momentum_batch.py`: Batched Spectral Attention (BSA) code
- `layers/Mmomentum_learnable.py`: BSA with learnable EMA smoothing factor (alpha)
- `run_exp/`: Contains scripts for running the project, including baseline training and finetuning (with hyperparameter search)
- `config.py`: Configuration file. Some attributes are automatically adjusted in other files to fit the dataset, prediction length, etc.

## Acknowledgement

This project is built on the [Time-Series-Library GitHub Repository](https://github.com/thuml/Time-Series-Library), with modifications. Therefore, if you want to try other models, you can use the updated model Python files from this GitHub repository.

## Contact
- Dongjun Lee (elite1717@snu.ac.kr)
- Bong Gyun Kang (luckypanda@snu.ac.kr)