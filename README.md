# SML Project 2 — ETH Mug Segmentation

Binary image segmentation of ETH-branded mugs in RGB images. Built for the SML 2026 Kaggle competition.

## Task
Given an RGB image (252 × 378 × 3), predict a binary mask of the same spatial dimensions where pixels belonging to ETH mugs are 1 and all other pixels are 0. Evaluation metric: mean Intersection-over-Union (IoU).

## Repository Structure
```
sml_project2/
├── eth_mugs_dataset.py     # Dataset class with augmentations
├── train.py                # Training loop with checkpointing
├── predict.py              # Inference + RLE encoding for Kaggle
├── utils.py                # Metrics, RLE, save_predictions
├── models/
│   ├── __init__.py
│   ├── unet.py             # Classical U-Net (baseline)
│   └── unet_residual.py    # U-Net with residual blocks (improved)
├── configs/
│   ├── unet_small.yaml     # base_channels=32, depth=4
│   ├── unet_large.yaml     # base_channels=64, depth=4
│   └── unet_residual.yaml  # residual variant
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
python -c "from eth_mugs_dataset import ETHMugsDataset; \
  d = ETHMugsDataset('datasets/train_data', mode='train'); \
  print(f'Train samples: {len(d)}'); \
  img, mask = d[0]; \
  print(f'img: {img.shape}, mask: {mask.shape}, mask values: {mask.unique()}')"
```

## Training

Train with a config file:
```bash
python train.py --config configs/unet_small.yaml
python train.py --config configs/unet_large.yaml
python train.py --config configs/unet_residual.yaml
```

Useful flags:
- `--resume checkpoints/unet_small_best.pt` — resume from checkpoint
- `--no-amp` — disable mixed-precision (useful for debugging)

Checkpoints are saved as `checkpoints/<run_name>_best.pt` (best validation IoU) and `<run_name>_last.pt` (last epoch).

## Prediction & Kaggle Submission

```bash
python predict.py \
  --config configs/unet_residual.yaml \
  --checkpoint checkpoints/unet_residual_best.pt \
  --output predictions/submission_unet_residual.csv
```

Upload the resulting CSV to Kaggle. Format:
```
ImageId,EncodedPixels
0001,1 5 10 3 ...
0002,...
```

## Approach

**Baseline:** Classical U-Net with configurable depth and base channel count (`models/unet.py`).

**Improved:** U-Net with residual blocks in encoder and decoder (`models/unet_residual.py`) — better gradient flow, can train deeper.

**Two configurations** for the report comparison: `unet_small` (base=32) vs `unet_large` (base=64), same depth.

**Loss:** BCE + Dice (handles class imbalance — mug pixels are a small fraction of the image).

**Augmentations:** horizontal flip, random rotation (±15°), color jitter, random crop+resize. Geometric transforms applied to both image and mask; color jitter only to image.

## Notes

- No pretrained models allowed (per project specs). All weights initialized from scratch.
- Image native resolution is 378 × 252; we train at this resolution by default.
- Random seed pinned in configs for reproducibility.
