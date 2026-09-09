# Mug Segmentation

Binary semantic segmentation of mugs in RGB images, built from scratch in PyTorch.
A U-Net baseline and a residual-U-Net variant are trained with a combined
BCE + Dice loss and compared across two model sizes.

**Best validation mean IoU: 0.9465** 

## Task
Given an RGB image (252 × 378 × 3), predict a binary mask of the same size where
mug pixels are 1 and everything else 0.

## Approach
- **Baseline:** classical U-Net with configurable depth and base channel count.
- **Improved:** U-Net with residual blocks in encoder and decoder (`model_name: resunet`)
  for better gradient flow and faster convergence.
- **Configurations:** `base_channels: 32` vs `64` at equal depth — a size/accuracy comparison.
- **Loss:** BCE + Dice; Dice handles the heavy class imbalance (mug pixels are a
  small fraction of each image).
- **Augmentations:** horizontal flip, random rotation (±15°), colour jitter,
  random crop + resize. Geometric transforms apply to both image and mask;
  colour jitter to the image only.

## Repository structure
```
config.yaml          # all hyperparameters in one flat file
mugs_dataset.py      # dataset class + augmentations
train.py             # models, loss, training, prediction
checkpoints/         # *.pt (gitignored)
predictions/         # submission_*.csv (gitignored)
environment.yml
```

## Running
```bash
conda env create -f environment.yml
python train.py --help   # train / predict options
```
