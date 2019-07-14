# Variational Autoencoders with Jointly Optimized Latent Dependency Structure

This is a PyTorch implementation of ["Variational Autoencoders with Jointly Optimized Latent Dependency Structure"](https://openreview.net/forum?id=SJgsCjCqt7) which appeared in ICLR '19.

## Initial setup
Clone the repository
```bash
git clone https://github.com/ys1998/vae-latent-structure.git
cd vae-latent-structure/
```
Ensure that the requirements mentioned [here](https://github.com/victoresque/pytorch-template#requirements) are met.

## Usage
```bash
# train
python train.py --config config.json
# resume from checkpoint
python train.py --resume <path_to_checkpoint>
# using multiple GPUs (equivalent to "CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py")
python train.py --device 2,3 -c config.json 
# test
python test.py --resume <path_to_checkpoint>
# visualization
tensorboard --logdir <path_to_log_dir>
```

## Code description
You can find a detailed description of the folder structure [here](https://github.com/victoresque/pytorch-template#folder-structure). The code is distributed across several branches, as described below
1. `master`: original implementation (GraphVAE)
2. `LSTM`: replacing top-down inference using a recurrent network (GraphLSTMVAE)
3. `vrnn`: extension of GraphVAE to sequential data (RecurrentGraphVAE)

All models have been described in the `model` folder in the respective branch. **Please look at our [report](https://ys1998.github.io/research/vae_latent.pdf) for more details on our variants and results.**

## Configuration
All config files are specified in JSON format as described [here](https://github.com/victoresque/pytorch-template#usage). Model-specific parameters/options are provided via the `arch` field. The `config.json` files in each branch already contain default parameters for the respective model.

## Acknowledgements
We would like to thank the authors of the PyTorch template https://github.com/victoresque/pytorch-template which served as a starting point for our project.

## References
Jiawei He and Yu Gong and Joseph Marino and Greg Mori and Andreas Lehrmann. *Variational Autoencoders with Jointly Optimized Latent Dependency Structure*, ICLR 2019. URL https://openreview.net/forum?id=SJgsCjCqt7
