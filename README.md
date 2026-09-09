# Mug Segmentation

## Task
Given an RGB image (252 × 378 × 3), predict a binary mask of the same spatial dimensions where pixels belonging to mugs are 1 and all other pixels are 0. 

## Repository Structure
```
├── config.yaml             # All hyperparameters in one flat file
├── mugs_dataset.py     # Dataset class with augmentations
├── train.py                # Everything: models, loss, training, prediction
├── datasets/
│   ├── train_data/
│   │   ├── rgb/            # *_rgb.jpg
│   │   └── masks/          # *_mask.png
│   └── test_data/
│       └── rgb/            # *_rgb.jpg
├── checkpoints/            # *.pt files (gitignored)
├── predictions/            # submission_*.csv (gitignored)
└── environment.yml
```

## Approach

**Baseline:** Classical U-Net with configurable depth and base channel count.

**Improved:** U-Net with residual blocks in encoder and decoder (`model_name: resunet`) — better gradient flow, typically converges faster.

**Two configurations** for the report comparison: `base_channels: 32` (small) vs `base_channels: 64` (large), same depth.

**Loss:** BCE + Dice (handles class imbalance — mug pixels are a small fraction of the image).

**Augmentations:** horizontal flip, random rotation (±15°), color jitter, random crop+resize. Geometric transforms applied to both image and mask; color jitter only to image.
