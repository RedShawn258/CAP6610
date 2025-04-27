'''
This file is used to define the Latent Diffusion model for controllable image generation.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from typing import Optional, List, Tuple, Union

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    
    Returns:
        an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, device=timesteps.device)) * 
        torch.arange(start=0, end=half, device=timesteps.device) / half
    )
    args = timesteps[:, None] * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# Define the UNet architecture
class ResidualBlock(nn.Module):
    """
    Residual block for UNet that adapts to different input channel dimensions
    """
    def __init__(self, in_channels, out_channels, time_dim=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_time = time_dim is not None
        
        # First conv block
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Second conv block
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
        # Time embedding
        if self.use_time:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, out_channels)
            )
    
    def forward(self, x, time_emb=None):
        """
        Forward pass through the residual block
        
        Args:
            x: Input tensor [B, C, H, W]
            time_emb: Time embedding [B, time_dim]
        """
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # Add time embedding if provided
        if self.use_time and time_emb is not None:
            time_signal = self.time_mlp(time_emb)
            time_signal = time_signal.view(-1, self.out_channels, 1, 1)
            h = h + time_signal
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """
    Self-attention block for adding attention to the UNet
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Normalization
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        
        # Query, key, value projections
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Output projection
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Scaling factor for dot product attention
        self.scale = (channels ** -0.5)
    
    def forward(self, x):
        """Forward pass with self-attention mechanism"""
        residual = x
        batch, channels, height, width = x.shape
        
        # Normalize
        norm_x = self.norm(x)
        
        # Get query, key, value projections
        q = self.q(norm_x).view(batch, channels, -1)
        k = self.k(norm_x).view(batch, channels, -1)
        v = self.v(norm_x).view(batch, channels, -1)
        
        # Transpose for attention dot product: b, c, hw -> b, hw, c
        q = q.permute(0, 2, 1)
        
        # Attention
        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Value projection
        out = torch.bmm(attn, v.permute(0, 2, 1))
        out = out.permute(0, 2, 1).view(batch, channels, height, width)
        
        # Output projection
        out = self.proj_out(out)
        
        # Residual connection
        return out + residual

class Downsample(nn.Module):
    """
    Downsampling module for UNet
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """
    Upsampling module for UNet
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timestep
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        
        # Zero pad if dim is odd
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))
            
        return embeddings

class SimpleUNet(nn.Module):
    """
    A simple UNet model with fixed dimensions, avoiding complex skip connections
    """
    def __init__(
        self,
        in_channels=7,
        out_channels=4,
        hidden_size=64,
        time_embedding_dim=128,
        use_attention=True,  # Whether to use attention modules
    ):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(hidden_size, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        
        # Flag for attention usage
        self.use_attention = use_attention
        
        # Encoder
        self.conv_in = nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1)
        
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_size),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_size * 2),
            nn.SiLU(),
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_size * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_size * 2, hidden_size * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_size * 4),
            nn.SiLU(),
        )
        
        # Attention layers
        if self.use_attention:
            self.attn1 = AttentionBlock(hidden_size * 2)
            self.attn2 = AttentionBlock(hidden_size * 4)
            self.attn_middle = AttentionBlock(hidden_size * 4)
        
        # Time embedding for middle
        self.time_middle = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, hidden_size * 4)
        )
        
        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(hidden_size * 4, hidden_size * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_size * 4),
            nn.SiLU(),
            nn.Conv2d(hidden_size * 4, hidden_size * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_size * 4),
            nn.SiLU(),
        )
        
        # Decoder
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_size * 4, hidden_size * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_size * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_size * 2, hidden_size * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_size * 2),
            nn.SiLU(),
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_size),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_size),
            nn.SiLU(),
        )
        
        # Output
        self.conv_out = nn.Conv2d(hidden_size, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            t: Timestep tensor [B]
        """
        # Time embedding
        t_emb = timestep_embedding(t, self.time_embedding.parameters().__next__().shape[1])
        t_emb = self.time_embedding(t_emb)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Downsampling
        h = self.down1(h)
        if self.use_attention:
            h = self.attn1(h)
            
        h = self.down2(h)
        if self.use_attention:
            h = self.attn2(h)
        
        # Middle with time conditioning
        middle_t_emb = self.time_middle(t_emb).view(-1, h.shape[1], 1, 1)
        h = self.middle(h) + middle_t_emb
        
        if self.use_attention:
            h = self.attn_middle(h)
        
        # Upsampling
        h = self.up1(h)
        h = self.up2(h)
        
        # Output
        return self.conv_out(h)

class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion model process
    """
    def __init__(
        self,
        model,
        image_size,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        loss_type="l2",
        v_posterior=0.0,
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.v_posterior = v_posterior
        
        # Define beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute diffusion parameters
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1. - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - self.alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1. - self.alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1. / self.alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / self.alphas_cumprod - 1))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod))
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_mean_variance(self, x_t, t, clip_denoised=True):
        """
        Predict mean and variance for reverse diffusion step
        """
        pred_noise = self.model(x_t, t)
        
        # Extract parameters
        alpha_t = extract(self.alphas, t, x_t.shape)
        alpha_cumprod_t = extract(self.alphas_cumprod, t, x_t.shape)
        beta_t = extract(self.betas, t, x_t.shape)
        
        # Calculate predicted x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
        # Calculate mean using posterior formula
        model_mean = beta_t * extract(self.posterior_mean_coef1, t, x_t.shape) * pred_x0 + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return {
            "mean": model_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance_clipped,
            "pred_x0": pred_x0,
        }
    
    def p_sample(self, x_t, t, clip_denoised=True):
        """
        Sample from p(x_{t-1} | x_t)
        """
        out = self.p_mean_variance(x_t, t, clip_denoised)
        noise = torch.randn_like(x_t)
        
        # No noise for t=0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, device, noise=None):
        """
        Generate samples using the reverse diffusion process
        """
        batch_size = shape[0]
        
        # Start from pure noise
        if noise is None:
            img = torch.randn(shape, device=device)
        else:
            img = noise
            
        # Iterate over diffusion steps
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            img = self.p_sample(img, t_batch)
            
        return img
    
    @torch.no_grad()
    def sample(self, batch_size=1, device=None, noise=None):
        """
        Sample from the model
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate samples on
            noise: Initial noise (if None, random noise will be used)
        
        Returns:
            Samples
        """
        if device is None:
            device = next(self.model.parameters()).device
            
        shape = (batch_size, self.model.out_channels, self.image_size, self.image_size)
        
        # Sample Gaussian noise
        x = noise if noise is not None else torch.randn(shape, device=device)
        
        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get model prediction (noise)
            noise_pred = self.model(x, t_batch)
            
            # Calculate denoised image
            alpha_t = extract(self.alphas, t_batch, x.shape)
            alpha_cumprod_t = extract(self.alphas_cumprod, t_batch, x.shape)
            beta_t = extract(self.betas, t_batch, x.shape)
            
            # Calculate mean using predicted noise
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Get mean of the posterior
            mean = beta_t * extract(self.posterior_mean_coef1, t_batch, x.shape) * pred_x0
            mean += extract(self.posterior_mean_coef2, t_batch, x.shape) * x
            
            # Get variance of the posterior
            var = extract(self.posterior_variance, t_batch, x.shape)
            
            # Sample from the posterior
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = mean + torch.sqrt(var) * noise
        
        return x
    
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute mean and variance of posterior q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_losses(self, x_0, t, noise=None, z_cond=None):
        """
        Training losses for denoising diffusion probabilistic model
        
        Args:
            x_0: Clean data
            t: Timesteps
            noise: Noise to add (optional, will be generated if None)
            z_cond: Conditioning signal (optional)
        """
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Forward diffusion to noisy sample
        x_t = self.q_sample(x_0, t, noise)
        
        # Create model input (either just x_t or combined with z_cond)
        if z_cond is not None:
            # Combine noised input with conditioning
            model_input = torch.cat([x_t, z_cond], dim=1)
        else:
            model_input = x_t
        
        # Predict noise
        noise_pred = self.model(model_input, t)
        
        # Compute loss
        if self.loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise_pred, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return loss
    
    def forward(self, x, condition=None):
        """
        Forward pass for training
        
        Args:
            x: Input tensor
            condition: Conditioning signal (optional)
        """
        b, c, h, w = x.shape
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        
        return self.p_losses(x, t, z_cond=condition)

class LatentDiffusion(nn.Module):
    """
    Latent Diffusion Model implementation
    """
    def __init__(
        self,
        latent_dim=4,
        image_size=256,
        input_channels=3,
        condition_channels=3,
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
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
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
        
        # UNet model for diffusion in latent space
        latent_size = image_size // 4  # 2x downsampling in encoder
        
        # Combine latent and condition channels
        unet_in_channels = latent_dim + condition_channels
        
        # Use the simplified UNet for more stability
        self.unet = SimpleUNet(
            in_channels=unet_in_channels,
            out_channels=latent_dim,
            hidden_size=model_channels,
            time_embedding_dim=time_dim,
            use_attention=True,
        )
        
        # Diffusion process
        self.diffusion = GaussianDiffusion(
            model=self.unet,
            image_size=latent_size,
            timesteps=timesteps,
            loss_type=loss_type,
            beta_schedule=beta_schedule,
        )
        
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
    
    def forward(self, batch):
        """
        Forward pass for training
        """
        x = batch["image"]
        condition = batch["condition"]
        
        # Encode image to latent space
        z = self.encode(x)
        
        # Downsample condition to match latent space size
        condition_down = F.interpolate(condition, size=z.shape[2:], mode="bilinear", align_corners=False)
        
        # Generate random timesteps for diffusion
        b = z.shape[0]
        t = torch.randint(0, self.diffusion.timesteps, (b,), device=z.device).long()
        
        # Pass to diffusion model for loss calculation
        return self.diffusion.p_losses(z, z_cond=condition_down, t=t, noise=None)
    
    @torch.no_grad()
    def sample(self, condition, batch_size=None, device=None):
        """
        Sample images conditioned on input
        
        Args:
            condition: Conditioning signal [B, C, H, W]
            batch_size: Number of samples to generate (if None, uses condition batch size)
            device: Device to generate samples on
        
        Returns:
            Generated images [B, C, H, W]
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Ensure condition is on the correct device
        condition = condition.to(device)
        
        # Use condition's batch size if batch_size is not specified
        if batch_size is None:
            batch_size = condition.shape[0]
        
        # Downsample condition to match latent space size
        latent_size = self.image_size // 4
        condition_down = F.interpolate(condition, size=(latent_size, latent_size), mode="bilinear", align_corners=False)
        
        # Repeat condition for each sample if needed or trim if larger
        if condition_down.shape[0] == 1 and batch_size > 1:
            condition_down = condition_down.repeat(batch_size, 1, 1, 1)
        elif condition_down.shape[0] > batch_size:
            condition_down = condition_down[:batch_size]
        elif condition_down.shape[0] < batch_size and condition_down.shape[0] > 1:
            # If condition has multiple samples but not enough, repeat the last ones
            repeats_needed = batch_size - condition_down.shape[0]
            extra_conditions = condition_down[-1:].repeat(repeats_needed, 1, 1, 1)
            condition_down = torch.cat([condition_down, extra_conditions], dim=0)
            
        # Use the actual batch size from condition
        actual_batch_size = condition_down.shape[0]
        
        # Start from noise
        latent_shape = (actual_batch_size, self.latent_dim, latent_size, latent_size)
        noise = torch.randn(latent_shape, device=device)
        
        # Sample using the diffusion model with conditioning
        for t in reversed(range(self.diffusion.timesteps)):
            t_batch = torch.full((actual_batch_size,), t, device=device, dtype=torch.long)
            
            # Combine latent and condition for model input
            model_input = torch.cat([noise, condition_down], dim=1)
            
            # Get model prediction (noise)
            noise_pred = self.unet(model_input, t_batch)
            
            # Calculate denoised image using predicted noise
            alpha_t = extract(self.diffusion.alphas, t_batch, noise.shape)
            alpha_cumprod_t = extract(self.diffusion.alphas_cumprod, t_batch, noise.shape)
            beta_t = extract(self.diffusion.betas, t_batch, noise.shape)
            
            # Get predicted original
            pred_x0 = (noise - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Get mean for next step
            mean = beta_t * extract(self.diffusion.posterior_mean_coef1, t_batch, noise.shape) * pred_x0
            mean += extract(self.diffusion.posterior_mean_coef2, t_batch, noise.shape) * noise
            
            # Get variance
            var = extract(self.diffusion.posterior_variance, t_batch, noise.shape)
            
            # Sample using mean and variance
            eps = torch.randn_like(noise) if t > 0 else torch.zeros_like(noise)
            noise = mean + torch.sqrt(var) * eps
        
        # Decode latent to image
        samples = self.decode(noise)
        
        return samples

# Helper function for extracting timestep embeddings
def extract(a, t, x_shape):
    """
    Extract timestep embeddings
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))) 