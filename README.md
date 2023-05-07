<div align="center">

# Unique Class Count

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<br>
[![Paper](https://img.shields.io/badge/paper--arxiv-1906.07647-B31B1B)](https://arxiv.org/abs/1906.07647)
[![Conference](https://img.shields.io/badge/ICLR-2020-4b44ce)](https://iclr.cc/Conferences/2020)

</div>

## Description

Pytorch implementation for ICLR 2020 paper "Weakly Supervised Clustering by Exploiting Unique Class Count"

![paper image](/ucc_framework.png "UCC Framework")


## Installation

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## How to run

### Download dataset first
mnist dataset is small, so it has been uploaded to github with the project.

for camelyon dataset, download the pre-processed [dataset](http://bit.ly/uniqueclasscount) first, and put the dataset in data dir.

### To see reproducing result
run notebook in notebooks dir

### Train
Train mnist model with default configuration

```bash
# train
python ucc/train.py
```

Train camelyon model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python ucc/train.py experiment=camelyon.yaml
```
