# Controllable Image Generation using Diffusion Models

This academic project explores the application of diffusion models for controllable image generation, focusing on how structural controls like edge maps and pose estimates can guide the generation process.

## Project Overview

This project implements the concepts presented in two key papers:
1. Rombach et al. (2022) - "High-Resolution Image Synthesis with Latent Diffusion Models"
2. Zhang et al. (2023) - "Adding Conditional Control to Text-to-Image Diffusion Models"

The core components include:
- Using a pre-trained latent diffusion model as the base
- Integrating structural controls (edge maps, pose estimates, segmentation masks)
- Evaluating the quality and controllability of the generated images

## Project Structure

The project follows a standard academic format:

1. **Introduction**: Background on diffusion models and conditional generation
2. **Problem Statement**: Formulation of controllable generation as a probabilistic inference problem
3. **Algorithms**: Implementation of latent diffusion with conditioning signals
4. **Experiments**: Demonstration of the model on various control types
5. **Conclusion**: Analysis of results and potential applications

## Environment Setup

Create and activate a conda environment:

```bash
conda create -n diffusion python=3.10
conda activate diffusion
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Inference with Pre-trained Models

To generate images using a pre-trained model with edge conditioning:

```bash
python inference.py --checkpoint ./checkpoints/model_final.pt --condition_image path/to/edge_map.png --output_dir ./outputs
```

### Options

- `--checkpoint`: Path to the pre-trained model checkpoint
- `--condition_image`: Path to the condition image (edge map, pose, or segmentation)
- `--control_type`: Type of conditioning ('edge', 'pose', 'segmentation', default: 'edge')
- `--image_size`: Output image resolution (default: 256)
- `--num_samples`: Number of samples to generate (default: 4)

### Targeted Tasks

1. **Edge-to-photo synthesis**: Generate realistic images from edge maps
2. **Pose-to-human synthesis**: Generate human figures from pose skeletons
3. **Segmentation-to-image rendering**: Create realistic scenes from semantic layouts

## Model Architecture

The implementation utilizes:

1. **Latent Diffusion Model (LDM)**: Operating in a compressed latent space to reduce computational costs
2. **Conditioning Mechanisms**: Techniques to inject structural information into the diffusion process
3. **Evaluation Metrics**: LPIPS, MSE, and FID to assess generation quality

## Experimental Setup

The project evaluates the model on:
- COCO dataset for diverse scene generation
- CelebA-HQ for facial image generation
- Synthetic edge maps, pose estimates, and segmentation masks

## Results

Sample outputs showing the effect of different conditioning signals are saved in the `outputs` directory.

## References

- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. In CVPR. arXiv:2112.10752.
- Zhang, Y., Agrawala, M., & Zhu, J. Y. (2023). Adding Conditional Control to Text-to-Image Diffusion Models. arXiv:2302.05543.
- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models" 