import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm

from src.models.latent_diffusion import SimpleUNet, LatentDiffusion
from src.models.controlnet import ControlNet

def parse_args():
    parser = argparse.ArgumentParser(description="Simple Inference with Weight Matching")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="ldm", choices=["ldm", "controlnet"],
                        help="Type of model to use (ldm or controlnet)")
    parser.add_argument("--condition_image", type=str, required=True, 
                        help="Path to condition image (edge map, pose, or segmentation)")
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                        help="Directory to save output images")
    parser.add_argument("--image_size", type=int, default=128, 
                        help="Size of output image")
    parser.add_argument("--num_samples", type=int, default=2, 
                        help="Number of samples to generate")
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    
    return parser.parse_args()

def load_condition_image(path, image_size, output_channels=3):
    # Load image and convert to tensor
    if not os.path.exists(path):
        raise FileNotFoundError(f"Condition image not found: {path}")
    
    image = Image.open(path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    rgb_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    print(f"Loaded RGB image: shape={rgb_tensor.shape}")
    
    # Create the final condition tensor with the right number of channels
    if output_channels == 1:
        # Convert to grayscale using ITU-R BT.601 conversion
        condition = 0.299 * rgb_tensor[:, 0:1] + 0.587 * rgb_tensor[:, 1:2] + 0.114 * rgb_tensor[:, 2:3]
    elif output_channels == 3:
        condition = rgb_tensor
    elif output_channels == 2:
        # Use first two channels (RG)
        condition = rgb_tensor[:, :2]
    elif output_channels > 3:
        # Expand by duplicating channels
        condition = torch.cat([rgb_tensor] + [rgb_tensor[:, :output_channels-3]] * (output_channels // 3), dim=1)
        condition = condition[:, :output_channels]
    
    print(f"Converted condition: shape={condition.shape}, min={condition.min()}, max={condition.max()}")
    return condition

def extract_model_params(checkpoint):
    # Try to determine model parameters from checkpoint
    latent_dim = 4
    condition_channels = 1
    
    # Check for UNet weights to determine channel dimensions
    if isinstance(checkpoint, dict):
        for key in checkpoint:
            if isinstance(checkpoint[key], dict):
                # Look in nested dictionaries
                for subkey, value in checkpoint[key].items():
                    if 'unet.conv_in.weight' in subkey or subkey == 'conv_in.weight':
                        if isinstance(value, torch.Tensor):
                            in_channels = value.shape[1]
                            # Assume input is [latent_dim + condition_channels]
                            if in_channels > latent_dim:
                                condition_channels = in_channels - latent_dim
                            print(f"Found weight tensor with in_channels={in_channels}")
                            print(f"Inferred: latent_dim={latent_dim}, condition_channels={condition_channels}")
                            return latent_dim, condition_channels
            elif 'unet.conv_in.weight' in key or key == 'conv_in.weight':
                if isinstance(checkpoint[key], torch.Tensor):
                    in_channels = checkpoint[key].shape[1]
                    # Assume input is [latent_dim + condition_channels]
                    if in_channels > latent_dim:
                        condition_channels = in_channels - latent_dim
                    print(f"Found weight tensor with in_channels={in_channels}")
                    print(f"Inferred: latent_dim={latent_dim}, condition_channels={condition_channels}")
                    return latent_dim, condition_channels
    
    # Fallback to defaults
    print(f"Could not determine from checkpoint, using defaults: latent_dim={latent_dim}, condition_channels={condition_channels}")
    return latent_dim, condition_channels

def create_simple_unet_model(in_channels, out_channels):
    print(f"Creating UNet with in_channels={in_channels}, out_channels={out_channels}")
    
    model = SimpleUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=64,
        time_embedding_dim=128,
    )
    
    return model

def diffusion_sample(model, condition, latent_shape, timesteps=1000, device="cpu"):
    """Sample from diffusion model - basic implementation"""
    # Create linear beta schedule
    beta_start = 1e-4
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Start with random noise
    x = torch.randn(latent_shape, device=device)
    
    # Sample using DDIM for speed
    sampling_timesteps = 100
    times = torch.linspace(
        timesteps - 1, 0, sampling_timesteps, 
        dtype=torch.long, device=device
    )
    
    # Resize condition to match latent shape
    h, w = latent_shape[2], latent_shape[3]
    condition_resized = F.interpolate(condition, size=(h, w), mode='bilinear', align_corners=False)
    
    # Sampling loop
    for i, t in enumerate(tqdm(times, desc="Sampling")):
        # Current timestep alpha values
        a_t = alphas_cumprod[t].view(-1, 1, 1, 1)
        a_prev = alphas_cumprod[t-1].view(-1, 1, 1, 1) if t > 0 else torch.ones_like(a_t)
        
        # Time embedding
        t_batch = torch.full((latent_shape[0],), t, device=device)
        
        # Concat latent and condition
        model_input = torch.cat([x, condition_resized], dim=1)
        
        # Predict noise
        with torch.no_grad():
            pred_noise = model(model_input, t_batch)
        
        # DDIM update formula
        pred_x0 = (x - torch.sqrt(1 - a_t) * pred_noise) / torch.sqrt(a_t)
        pred_x0 = torch.clamp(pred_x0, -1., 1.)
        
        # No noise for the last step
        if i < sampling_timesteps - 1:
            noise = torch.randn_like(x)
            eta = 0.0  # 0 = deterministic, 1 = stochastic
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(1 - a_t / a_prev)
            
            # Update x
            c1 = torch.sqrt(a_prev)
            c2 = torch.sqrt(1 - a_prev - sigma**2)
            x = c1 * pred_x0 + c2 * pred_noise + sigma * noise
        else:
            # Last step, use predicted x0
            x = pred_x0
    
    return x

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load checkpoint first to determine condition channels
        print(f"Loading checkpoint: {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            
            # Determine latent_dim and condition_channels from checkpoint
            latent_dim, condition_channels = extract_model_params(checkpoint)
            
            # Get the actual state dict to use
            if 'model_state_dict' in checkpoint:
                print("Using model state dict from checkpoint")
                state_dict = checkpoint['model_state_dict']
            elif 'ema_model_state_dict' in checkpoint:
                print("Using EMA model state dict from checkpoint")
                state_dict = checkpoint['ema_model_state_dict']
            else:
                print("Using checkpoint directly as state dict")
                state_dict = checkpoint
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Using default parameters")
            latent_dim, condition_channels = 4, 1
            state_dict = None
        
        # Load condition image with the right number of channels
        condition = load_condition_image(
            args.condition_image, 
            args.image_size, 
            output_channels=condition_channels
        ).to(device)
        
        # Create model with matching dimensions
        in_channels = latent_dim + condition_channels
        model = create_simple_unet_model(in_channels, latent_dim).to(device)
        
        # Try to load weights
        if state_dict is not None:
            try:
                # Try to find UNet weights in the state dict
                filtered_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('unet.') or key.startswith('diffusion.model.'):
                        # Remove the prefix
                        new_key = key.replace('unet.', '').replace('diffusion.model.', '')
                        filtered_state_dict[new_key] = value
                
                if filtered_state_dict:
                    # Try to load filtered weights
                    try:
                        model.load_state_dict(filtered_state_dict, strict=False)
                        print("Loaded weights from checkpoint")
                    except Exception as e:
                        print(f"Could not load filtered weights: {e}")
                else:
                    print("Could not find compatible weights in checkpoint")
            except Exception as e:
                print(f"Error preparing weights: {e}")
                
        # Generate samples
        print(f"Generating {args.num_samples} samples...")
        samples = []
        
        for i in range(args.num_samples):
            print(f"Generating sample {i+1}/{args.num_samples}")
            
            # Sample in latent space
            latent_shape = (1, latent_dim, args.image_size // 4, args.image_size // 4)
            latent = diffusion_sample(
                model, condition, latent_shape, 
                timesteps=args.timesteps, device=device
            )
            
            # Upscale and normalize
            sample_upscaled = F.interpolate(latent, size=(args.image_size, args.image_size), 
                                           mode='bilinear', align_corners=False)
            
            # Normalize to [-1, 1]
            sample_normalized = (sample_upscaled - sample_upscaled.min()) / (sample_upscaled.max() - sample_upscaled.min()) * 2 - 1
            
            # Add to samples
            samples.append(sample_normalized)
        
        # Concatenate samples
        samples = torch.cat(samples, dim=0)
        print(f"Generated samples shape: {samples.shape}")
        
        # Save samples
        for i in range(samples.shape[0]):
            # Convert to 3 channels if needed
            if samples.shape[1] == 1:
                sample_rgb = samples[i].repeat(3, 1, 1)
            else:
                sample_rgb = samples[i, :3]  # Take first 3 channels if more than 3
                
            vutils.save_image(
                sample_rgb,
                os.path.join(args.output_dir, f"sample_{i:03d}.png"),
                normalize=True,
                value_range=(-1, 1)
            )
        
        # Save condition
        if condition.shape[1] == 1:
            condition_rgb = condition[0].repeat(3, 1, 1)
        else:
            condition_rgb = condition[0, :3]  # Take first 3 channels if more than 3
            
        vutils.save_image(
            condition_rgb,
            os.path.join(args.output_dir, "condition.png"),
            normalize=True
        )
        
        print(f"Samples saved to {args.output_dir}")
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 