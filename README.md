<h1 align="center">
  <br>
  DLC
  <br>
</h1>

<h4 align="center">
  An official PyTorch implementation of the ICLR 2025 paper
  <br>
  "Exploring Learning Complexity for Efficient Downstream Dataset Pruning"
</h4>

<div align="center">
  <a href="https://arxiv.org/abs/2402.05356" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat-square">
  </a> &nbsp;&nbsp;&nbsp;
  <a href=''>
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
  </a>
</div>

<p align="center">
  <a href="#get-started">Get Started</a> •
  <a href="#citation">Citation</a>
</p>

## Get Started
### Overview
This repository is an official PyTorch implementation of the ICLR 2025 paper 'Exploring Learning Complexity for Efficient Downstream Dataset Pruning'. The illustration of our algorithm core is shown as below:
![diagram](https://github.com/lygjwy/DLC/blob/main/figs/diagram.png)

### Requirements
```bash
pip install -r requirements.txt
```

### Pruning
```bash
$ ./scripts/pruning.sh
```

### Tuning
```bash
$ ./scripts/tuning.sh
```

### Results
![diagram](https://github.com/lygjwy/DLC/blob/main/figs/result-vision.png)

## Citation
If you find our repository useful for your research, please consider citing our paper:
```
@inproceedings{
  jiang2025exploring,
  title={Exploring Learning Complexity for Efficient Downstream Dataset Pruning},
  author={Wenyu Jiang and Zhenlong Liu and Zejian Xie and Songxin Zhang and Bingyi Jing and Hongxin Wei},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=FN7n7JRjsk}
}
```
