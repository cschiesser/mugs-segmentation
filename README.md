# SML Project 2 — ETH Mug Segmentation

Binary image segmentation of ETH-branded mugs in RGB images. Built for the SML 2026 Kaggle competition.

## Task
Given an RGB image (252 × 378 × 3), predict a binary mask of the same spatial dimensions where pixels belonging to ETH mugs are 1 and all other pixels are 0. Evaluation metric: mean Intersection-over-Union (IoU).

## Repository Structure
```
sml_project2/
├── config.yaml             # All hyperparameters in one flat file
├── eth_mugs_dataset.py     # Dataset class with augmentations
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

## Setup

### 1. Create conda environment
```bash
conda env create -f environment.yml
conda activate sml_p2
```

If you have a CUDA GPU and `environment.yml` installs the CPU build, install the CUDA wheel manually:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Download data
Download the dataset from the Kaggle competition page and place it as:
```
datasets/train_data/rgb/0001_rgb.jpg, 0002_rgb.jpg, ...
datasets/train_data/masks/0001_mask.png, 0002_mask.png, ...
datasets/test_data/rgb/0001_rgb.jpg, ...
```

### 3. Sanity check
```bash
python -c "
from eth_mugs_dataset import make_train_val_split
train_ds, val_ds = make_train_val_split('datasets/train_data', val_fraction=0.15, seed=42)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}')
img, mask = train_ds[0]
print(f'img: {img.shape}, mask mean: {mask.mean().item():.4f}')
"
```

## Training

Edit `config.yaml` to configure your run, then:

```bash
python train.py                        # uses config.yaml by default
python train.py --config config.yaml   # same
python train.py --resume checkpoints/unet_small_best.pt  # resume
```

To switch models or configurations, just edit `config.yaml`:
- `model_name: unet` or `model_name: resunet`
- `base_channels: 32` (small) or `base_channels: 64` (large)

Checkpoints are saved as `checkpoints/<run_name>_best.pt` (best validation IoU) and `<run_name>_last.pt` (last epoch).

## Prediction & Kaggle Submission

```bash
python train.py --predict \
  --checkpoint checkpoints/unet_small_best.pt \
  --output predictions/submission.csv
```

Optional flags:
- `--threshold 0.5` — sigmoid threshold (default: 0.5)
- `--tta` — horizontal-flip test-time augmentation

Upload the resulting CSV to Kaggle. Format:
```
ImageId,EncodedPixels
0001,1 5 10 3 ...
0002,...
```

## Approach

**Baseline:** Classical U-Net with configurable depth and base channel count.

**Improved:** U-Net with residual blocks in encoder and decoder (`model_name: resunet`) — better gradient flow, typically converges faster.

**Two configurations** for the report comparison: `base_channels: 32` (small) vs `base_channels: 64` (large), same depth.

**Loss:** BCE + Dice (handles class imbalance — mug pixels are a small fraction of the image).

**Augmentations:** horizontal flip, random rotation (±15°), color jitter, random crop+resize. Geometric transforms applied to both image and mask; color jitter only to image.

## Notes

- No pretrained models allowed (per project specs). All weights initialized from scratch.
- Image native resolution is 378 × 252; we train at this resolution by default.
- Random seed pinned in `config.yaml` for reproducibility.
