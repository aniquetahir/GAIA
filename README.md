# Introduction
This repository contains the code for the paper [Fairness through Aleatoric Uncertainty](https://arxiv.org/abs/2304.03646). This paper will be presented at the 32nd ACM International Conference on Information and Knowledge Management (CIKM). 

# Installation
Desired pre-requisites are mentioned in `requirements.txt`. Please add additional requirements if needed.

# Running Experiments
Please use the script `run_experiments.py`. The results will be stored in the `compiled_results` folder.

# Image Experiments
Please use the `fair-mixup` papers [code](https://github.com/chingyaoc/fair-mixup) and run `data_processing.py` to generate the train/test splits. Change the `CELEBA_PATH` variable in `utils/jax/models/image.py` to point to the celeba directory generated with the splits. 

Image experiments can be executed using `run_image_experiments.py`.

# Citation
If you find this code useful, please cite our paper:
```bibtex
@article{tahir2023fairness,
  title={Fairness through Aleatoric Uncertainty},
  author={Tahir, Anique and Cheng, Lu and Liu, Huan},
  journal={arXiv preprint arXiv:2304.03646},
  year={2023}
}
```
