'''
This file is used to define the ControlNet model for conditional image generation 
using diffusion models.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Union
from tqdm import tqdm

from src.models.latent_diffusion import timestep_embedding as get_timestep_embedding, SimpleUNet, ResidualBlock as ResBlock

class ControlNet(nn.Module):
    """
    ControlNet model for conditional image generation using diffusion models.
    This model adapts to the actual condition channels in the data.
    """
    def __init__(
        self,
        latent_dim=4,
        image_size=256,
        input_channels=3,
        condition_channels=1,  # Number of condition channels (e.g., 1 for edge maps)
        model_channels=64,
        channel_mult=(1, 2, 4, 4),
        time_dim=128,
        num_res_blocks=2,
        attention_resolutions=(8, 16, 32),
        dropout=0.0,
        timesteps=1000,
        loss_type="l2",
        beta_schedule="linear",
        pretrained_path=None,
        use_fp16=False,  # Whether to use fp16 precision
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.condition_channels = condition_channels
        self.use_fp16 = use_fp16
        self.timesteps = timesteps
        self.model_channels = model_channels
        self.time_dim = time_dim
        
        print(f"Initializing ControlNet with latent_dim={latent_dim}, condition_channels={condition_channels}")
        
        # Autoencoder (Encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1),
        )
        
        # Autoencoder (Decoder)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        
        # UNet model for diffusion with latent + condition input
        unet_in_channels = latent_dim + condition_channels
        
        # Create the base UNet for diffusion
        self.unet = SimpleUNet(
            in_channels=unet_in_channels,
            out_channels=latent_dim,
            hidden_size=model_channels,
            time_embedding_dim=time_dim,
        )
        
        # Create the control UNet with the same parameters
        self.control_model = BasicControlNet(
            in_channels=condition_channels,  # Just the condition image
            model_channels=model_channels,
            out_channels=latent_dim,  # Output matches latent dimension
            time_dim=time_dim,
        )
        
        # Calculate beta schedule (for diffusion)
        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, timesteps, dtype=torch.float32)
        else:
            # Default to linear
            betas = torch.linspace(0.0001, 0.02, timesteps, dtype=torch.float32)
        
        # Register buffers for diffusion process
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            print(f"Loaded pretrained model from {pretrained_path}")
    
    def encode(self, x):
        """Encode an image to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space to image"""
        return self.decoder(z)
    
    def add_noise(self, x, t, noise=None):
        """Add noise to input x at timestep t"""
        if noise is None:
            noise = torch.randn_like(x)
            
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Extract the appropriate alpha values
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        
        # Add noise
        noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        
        return noisy_x, noise
    
    def _ensure_correct_unet(self, condition):
        """Ensure UNet has the correct input dimensions based on condition channels"""
        actual_condition_channels = condition.shape[1]
        if actual_condition_channels != self.condition_channels:
            print(f"Condition channels mismatch! Expected {self.condition_channels}, got {actual_condition_channels}")
            # Dynamically recreate the UNet if needed for correct channel dimensions
            if not hasattr(self, 'fixed_unet'):
                self.fixed_unet = SimpleUNet(
                    in_channels=self.latent_dim + actual_condition_channels,
                    out_channels=self.latent_dim,
                    hidden_size=self.model_channels,
                    time_embedding_dim=self.time_dim,
                )
                self.fixed_unet.to(next(self.parameters()).device)
                print(f"Created fixed UNet with in_channels={self.latent_dim + actual_condition_channels}")
                
        # Return the correct UNet to use
        return self.fixed_unet if hasattr(self, 'fixed_unet') else self.unet

    def forward(self, batch):
        """
        Forward pass for training
        
        Args:
            batch: Dictionary containing 'image' and 'condition'
        
        Returns:
            Loss value
        """
        x = batch["image"]
        condition = batch["condition"]
        
        # Ensure UNet has correct input dimensions
        unet = self._ensure_correct_unet(condition)
        
        # Encode image to latent space
        z = self.encode(x)
        
        # Generate random timesteps for diffusion
        b = z.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=z.device).long()
        
        # Add noise to latent
        noisy_z, noise = self.add_noise(z, t)
        
        # Get control features from condition - resize to match latent
        condition_down = F.interpolate(condition, size=z.shape[2:], mode="bilinear", align_corners=False)
        
        # Combine noisy latent with condition
        model_input = torch.cat([noisy_z, condition_down], dim=1)
        
        # Predict noise
        predicted_noise = unet(model_input, t)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss

    def sample(self, condition, batch_size=1, device=None):
        """
        Sample from the model using the condition
        
        Args:
            condition: Conditioning signal [B, C, H, W]
            batch_size: Number of samples to generate
            device: Device to use
        
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Prepare condition
        if condition.shape[0] != batch_size:
            condition = condition.repeat(batch_size, 1, 1, 1)
        
        # Ensure UNet has correct input dimensions
        unet = self._ensure_correct_unet(condition)
        
        # Get latent size
        latent_size = self.image_size // 4
        
        # Sample initial noise
        z = torch.randn((batch_size, self.latent_dim, latent_size, latent_size), device=device)
        
        # Downsample condition to match latent size
        condition_down = F.interpolate(condition, size=(latent_size, latent_size), mode="bilinear", align_corners=False)
        
        # Reverse diffusion process
        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            with torch.no_grad():
                # Combine latent and condition
                model_input = torch.cat([z, condition_down], dim=1)
                
                # Predict noise
                predicted_noise = unet(model_input, t_batch)
                
                # Sample from posterior at timestep t
                z = self._sample_posterior(z, t_batch, predicted_noise)
                
        # Decode final latent
        samples = self.decode(z)
        
        return samples
    
    def _sample_posterior(self, x, t, predicted_noise):
        """Sample from the posterior distribution at timestep t"""
        # Get alpha and beta values
        alpha = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta = self.betas[t].view(-1, 1, 1, 1)
        
        # Calculate predicted original image
        predicted_x0 = (x - torch.sqrt(1.0 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
        predicted_x0 = torch.clamp(predicted_x0, -1.0, 1.0)
        
        # Calculate mean of the posterior
        posterior_mean = predicted_x0 * torch.sqrt(alpha) * (1.0 - alpha_cumprod) / (1.0 - alpha_cumprod)
        posterior_mean += x * (1.0 - alpha) * torch.sqrt(alpha_cumprod) / (1.0 - alpha_cumprod)
        
        # Calculate posterior variance
        posterior_variance = beta * (1.0 - alpha_cumprod) / (1.0 - alpha_cumprod)
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))
        
        # Sample from posterior
        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        return posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise

    def load_state_dict(self, state_dict, strict=True):
        """
        Override to handle the fixed_unet when loading checkpoints
        """
        # If we created a fixed_unet but it's not in the state_dict, ignore the missing keys
        if hasattr(self, 'fixed_unet'):
            # First load what we can
            missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)
            
            print(f"INFO: Ignoring {len(missing_keys)} missing keys for fixed_unet when loading checkpoint")
            return missing_keys, unexpected_keys
        else:
            # Regular load
            return super().load_state_dict(state_dict, strict)


class BasicControlNet(nn.Module):
    """
    Basic version of ControlNet for conditioning - simpler to avoid recursion
    """
    def __init__(
        self,
        in_channels=1,
        model_channels=64,
        out_channels=4,
        time_dim=128,
    ):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Encoder pathway
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsample blocks
        self.down1 = nn.Sequential(
            ResBlock(model_channels, model_channels, time_dim),
            nn.Conv2d(model_channels, model_channels*2, kernel_size=3, stride=2, padding=1),
        )
        
        self.down2 = nn.Sequential(
            ResBlock(model_channels*2, model_channels*2, time_dim),
            nn.Conv2d(model_channels*2, model_channels*4, kernel_size=3, stride=2, padding=1),
        )
        
        self.down3 = nn.Sequential(
            ResBlock(model_channels*4, model_channels*4, time_dim),
            nn.Conv2d(model_channels*4, model_channels*8, kernel_size=3, stride=2, padding=1),
        )
        
        # Zero convs for each level
        self.zero_conv_0 = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
        self.zero_conv_1 = nn.Conv2d(model_channels*2, out_channels, kernel_size=3, padding=1)
        self.zero_conv_2 = nn.Conv2d(model_channels*4, out_channels, kernel_size=3, padding=1)
        self.zero_conv_3 = nn.Conv2d(model_channels*8, out_channels, kernel_size=3, padding=1)
        
        # Initialize zero convs with zeros
        for m in [self.zero_conv_0, self.zero_conv_1, self.zero_conv_2, self.zero_conv_3]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = get_timestep_embedding(t, self.time_embed.parameters().__next__().shape[1])
        t_emb = self.time_embed(t_emb)
        
        # Initial conv
        h0 = self.conv_in(x)
        
        # Control outputs for each level
        control0 = self.zero_conv_0(h0)
        
        # Downsample
        h1 = self.down1(h0)
        control1 = self.zero_conv_1(h1)
        
        h2 = self.down2(h1)
        control2 = self.zero_conv_2(h2)
        
        h3 = self.down3(h2)
        control3 = self.zero_conv_3(h3)
        
        # Return control signals
        return {
            "level_0": control0,
            "level_1": control1,
            "level_2": control2,
            "level_3": control3
        } 