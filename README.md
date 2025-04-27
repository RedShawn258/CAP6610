# Controllable Image Generation using Diffusion Models

This project implements and evaluates controllable image generation using diffusion models, focusing on how structural controls like edge maps, pose estimates, and segmentation masks can guide the generation process.

## Project Overview

This research project explores two key architectures for controllable image generation:
1. **Latent Diffusion Models (LDM)** - Operating in a compressed latent space to reduce computational costs
2. **ControlNet** - Adding conditional control to pre-trained diffusion models

The project demonstrates how these models can effectively guide the image generation process while maintaining high visual quality.

## Features

- Edge-to-photo synthesis: Generate realistic images from edge maps
- Depth-to-image synthesis: Creating scenes with proper spatial relationships from depth maps
- Segmentation-to-image rendering: Create realistic scenes from semantic layouts
- Multiple control types supported in a unified framework
- Evaluation tools for comparing output quality

## Installation

Create and activate a conda environment:

```bash
conda create -n diffusion python=3.10
conda activate diffusion
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/                  # Dataset storage (will be created)
├── checkpoints/           # Model checkpoints (will be created)
├── logs/                  # Training logs (will be created)
├── outputs/               # Generated images
├── src/                   # Source code
│   ├── data/              # Dataset classes
│   ├── models/            # Model architectures
│   ├── training/          # Training loop and utilities
│   ├── evaluation/        # Evaluation metrics
│   └── utils/             # Utility functions
├── main.py                # Training script
├── inference.py           # Inference script
└── requirements.txt       # Required packages
```

## Usage

### Training

To train a model:

```bash
python main.py --model_type ldm --control_type edge --dataset coco --num_epochs 100
```

Key parameters:
- `--model_type`: Model architecture (`ldm` or `controlnet`)
- `--control_type`: Type of conditioning (`edge`, `depth`, or `segmentation`)
- `--dataset`: Dataset to use (`coco` or `celeba`)
- `--image_size`: Output resolution (default: 256)
- `--batch_size`: Batch size for training (default: 8)

### Inference

To generate images using a pre-trained model:

```bash
python inference.py --checkpoint ./checkpoints/model_best.pt --condition_image path/to/edge_map.png --output_dir ./outputs --num_samples 4
```

Key parameters:
- `--checkpoint`: Path to model checkpoint
- `--condition_image`: Path to the conditioning image
- `--model_type`: Type of model (`ldm` or `controlnet`)
- `--image_size`: Output resolution (default: 128)
- `--num_samples`: Number of samples to generate (default: 2)

## Model Architecture

The implementation is based on:

1. **Latent Diffusion Model**:
   - VAE encoder/decoder for latent space compression
   - U-Net backbone for the diffusion model
   - Conditioning mechanisms for structural controls

2. **ControlNet**:
   - Uses a frozen copy of a pre-trained diffusion model
   - "Zero convolution" layers for learning specific conditioning signals
   - Preserves the base model's capabilities while adding conditioning

## Experimental Results

The models were evaluated on multiple datasets including COCO-Stuff and CelebA-HQ with various conditioning signals. Experimental results demonstrate that:

- The models successfully preserve structural information from conditioning signals
- They generate diverse and realistic outputs across different control types
- The conditioning mechanism effectively guides the generation process

Sample outputs are available in the `outputs/` directory.

## Citation

If you use this code in your research, please cite the following papers:

```
@inproceedings{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bjorn},
  booktitle={CVPR},
  year={2022}
}

@article{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Zhang, Lvmin and Agrawala, Maneesh and Zhu, Jun-Yan},
  journal={arXiv preprint arXiv:2302.05543},
  year={2023}
}
```