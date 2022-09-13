## installation

#### System Requirement

- Linux

#### Package Requirements

- PytTorch
- tensorboard
- opencv-python
- [compressai](https://github.com/InterDigitalInc/CompressAI/) 
  - using entropy-bottleneck block via [Variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436) to compress latent representation and count bpp.
- tqdm
- scikit-image

#### Data Preparation

##### Dataset

Download *The original training + test set (82GB)* from [vimeo-90k](http://toflow.csail.mit.edu/) , use '*vimeo_septuplet/sequences/* ' as  dataset root.

## Usage

Parameters in config.cfg

```shell
[train]         // training configuration

batch_size      // batch size
gpu_id          // gpu id
dataset_root    // path to vimeo_septuplet/sequences
gop             // group of pictures
train_fraction  // (0,1], sampling to avoid too large training dataset 

[test]          // testing configuration

dataset_root    // path to vimeo_septuplet/sequences
gop             // group of pictures
index           // dataset index in vimeo90k

```

Parameters in parameters.cfg

```
[ME]            // motion estimaton net
[MC]            // motion compensation net
[MVE]           // motion vector encoder
[MVD]           // motion vector decoder
[REE]           // residual encoder
[RED]           // residual decoder

name            // class name in .models
lr              // train models using specific learning rate
pretrain        // path to pretrain models
optimizer       // optimizer
```

#### Networks

1. Add a new network to .models

2. Open the .models.____init____.py 

   ```python
   from models.model_name import ModelName
   ```

3. run auxiliary trainer
4. Exchange *name* in parameters.cfg by name = ModelName, pretrain = path/to/pretrained_model.pth

#### Training

Training models cascade, run

```bash
python train_codec.py
```

Training a specific model 

```
python aux_trainer_model.py
```

