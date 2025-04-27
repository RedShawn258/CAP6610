import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from contextlib import nullcontext

class Trainer:
    """
    Trainer class for diffusion models
    """
    def __init__(
        self,
        model,
        dataloader,
        device,
        args,
        lr=1e-4,
        weight_decay=0.0,
        ema_decay=0.9999,
        amp_training=True,
        patience=10,  # Early stopping patience
        min_delta=0.001,  # Minimum change to qualify as improvement
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.args = args
        self.lr = lr if args.lr is None else args.lr
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.amp_training = amp_training
        
        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = float('inf')
        self.early_stop = False
        
        # Create paths for model checkpoints
        self.checkpoint_path = os.path.join(args.save_dir, "checkpoint.pt")
        self.best_model_path = os.path.join(args.save_dir, "model_best.pt")
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Create learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.dataloader) * args.num_epochs,
            eta_min=1e-6,
        )
        
        # Create EMA model
        self.ema_model = None
        if ema_decay > 0:
            self.ema_model = self.create_ema_model()
            self.update_ema_model(0.0)  # Initialize with current weights
        
        # Create gradient scaler for mixed precision training
        self.scaler = GradScaler() if amp_training else None
        
        # Create checkpoint directory
        os.makedirs(args.save_dir, exist_ok=True)
    
    def create_ema_model(self):
        """Create a copy of the model for EMA updates"""
        # Extract only the parameters needed for model creation
        model_params = {
            'latent_dim': self.args.latent_dim,
            'image_size': self.args.image_size,
            'pretrained_path': None  # Don't load pretrained weights for EMA model
        }
        
        # Create a new instance of the model type with the proper parameters
        ema_model = type(self.model)(**model_params)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        ema_model.eval()
        
        # Disable gradient computation for EMA model
        for param in ema_model.parameters():
            param.requires_grad = False
            
        return ema_model
    
    def update_ema_model(self, decay):
        """Update EMA model parameters"""
        with torch.no_grad():
            model_params = dict(self.model.named_parameters())
            ema_params = dict(self.ema_model.named_parameters())
            
            for name, param in model_params.items():
                if name in ema_params:
                    ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    def save_checkpoint(self, path=None, epoch=None, val_loss=None, is_best=False):
        """Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint (if None, use self.checkpoint_path)
            epoch: Current epoch number
            val_loss: Current validation loss
            is_best: Whether this is the best model so far
        """
        if path is None and self.checkpoint_path is None:
            return
            
        # Use provided path or default
        target_path = path if path is not None else self.checkpoint_path
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Create checkpoint dictionary
        checkpoint = {
            "epoch": epoch if epoch is not None else 0,
            "best_valid_loss": self.best_val_loss,
            "early_stop_counter": self.counter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        
        # Add EMA model if available
        if self.ema_model is not None:
            checkpoint["ema_model_state_dict"] = self.ema_model.state_dict()
            
        # Save checkpoint
        torch.save(checkpoint, target_path)
        
        # Save best model separately if this is the best model so far
        if is_best:
            best_path = os.path.join(os.path.dirname(target_path), "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
            
        print(f"Saved checkpoint to {target_path}" + (" [BEST]" if is_best else ""))
    
    def load_checkpoint(self, path=None):
        """Load model from checkpoint."""
        if path is None:
            path = self.checkpoint_path

        if not os.path.exists(path):
            print(f"Checkpoint not found at {path}. Starting from scratch.")
            return 0, float('inf')

        print(f"Loading checkpoint from {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Try to load the state dict and handle missing keys gracefully
            if hasattr(self.model, 'load_state_dict') and callable(getattr(self.model, 'load_state_dict')):
                try:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                except RuntimeError as e:
                    # Handle missing or unexpected keys
                    if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                        print(f"Warning when loading state dict: {e}")
                        # Try with strict=False
                        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                        print("Loaded checkpoint with non-strict matching.")
                    else:
                        raise e
            else:
                # Fallback to direct assignment
                self.model = checkpoint["model_state_dict"]
                
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            best_valid_loss = checkpoint["best_valid_loss"]
            
            # Load EMA model if present in checkpoint and model has EMA
            if self.ema_model is not None and "ema_model_state_dict" in checkpoint:
                try:
                    self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
                except RuntimeError as e:
                    print(f"Warning when loading EMA state dict: {e}")
                    self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)
                    print("Loaded EMA checkpoint with non-strict matching.")
            
            return epoch, best_valid_loss
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 0, float('inf')
    
    def check_early_stopping(self, val_loss):
        """Check if early stopping criteria is met.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            bool: True if this is the best model so far
        """
        is_best = False
        if val_loss < self.best_val_loss - self.min_delta:
            # Improvement in validation loss
            self.best_val_loss = val_loss
            self.counter = 0
            is_best = True
            # Save best model
            self.save_checkpoint(path=self.best_model_path, val_loss=val_loss, is_best=True)
            return is_best
        else:
            # No improvement in validation loss
            self.counter += 1
            if self.counter >= self.patience:
                # Early stopping triggered
                self.early_stop = True
                print(f"Early stopping triggered! No improvement for {self.patience} epochs.")
            return is_best
    
    def train_epoch(self, epoch):
        """Train model for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = len(self.dataloader)
        
        start_time = time.time()
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
        
        # Store first batch of conditions for visualization
        monitor_batch = None
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            # Save the first batch for monitoring progress
            if batch_idx == 0 and monitor_batch is None:
                monitor_batch = batch["condition"].clone().detach()
                
            # Optimization step
            self.optimizer.zero_grad()
            
            if self.amp_training:
                with autocast():
                    loss = self.model(batch)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.model(batch)
                loss.backward()
                self.optimizer.step()
            
            # Update EMA model
            if self.ema_model is not None:
                self.update_ema_model(self.ema_decay)
            
            # Update learning rate
            self.scheduler.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        epoch_loss /= num_batches
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{self.args.num_epochs} completed in {elapsed_time:.2f}s, Loss: {epoch_loss:.4f}")
        
        # Use evaluator to visualize training progress if available
        if hasattr(self, 'evaluator') and self.evaluator is not None and monitor_batch is not None:
            # Visualize training progress every few epochs 
            # (not every epoch to avoid slowing down training)
            if (epoch + 1) % max(1, self.args.num_epochs // 10) == 0:
                if hasattr(self.args, 'log_dir'):
                    vis_metrics = self.evaluator.monitor_training(epoch, monitor_batch)
                    print(f"Visualization metrics: LPIPS diversity: {vis_metrics.get('lpips_diversity', 'N/A'):.4f}")
        
        return epoch_loss

    def validate(self, val_dataloader):
        """Evaluate the model on validation data"""
        self.model.eval()
        
        val_loss = 0.0
        num_batches = len(val_dataloader)
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                # Calculate loss
                with autocast() if self.amp_training else nullcontext():
                    loss = self.model(batch)
                
                val_loss += loss.item()
        
        return val_loss / num_batches

    def train(self, num_epochs, val_dataloader=None):
        """Train the model for a specified number of epochs."""
        print(f"Starting training for {num_epochs} epochs")
        
        start_epoch = 0
        
        # Try to load checkpoint if it exists
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                start_epoch, best_val_loss = self.load_checkpoint()
                self.best_val_loss = best_val_loss
                print(f"Loaded checkpoint. Starting from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
            except Exception as e:
                print(f"Could not load checkpoint: {e}")
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate model if validation dataloader is provided
            if val_dataloader is not None:
                print("Validating model...")
                val_loss = self.validate(val_dataloader)
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Check for early stopping and best model
                is_best = self.check_early_stopping(val_loss)
                
                # Evaluate model periodically
                if is_best or (epoch + 1) % self.args.eval_every == 0:
                    # Generate and save samples if evaluator is provided
                    if hasattr(self, 'evaluator') and self.evaluator is not None:
                        metrics = self.evaluator.evaluate()
                        print(f"Evaluation metrics: FID: {metrics.get('fid', 'N/A'):.4f}")
                        
                        # Generate and save samples
                        sample_path = f"{self.args.log_dir}/samples_epoch_{epoch+1:03d}.png"
                        self.evaluator.generate_samples(sample_path)
                        
                # Early stopping check
                if self.early_stop:
                    print(f"Early stopping triggered after {epoch+1} epochs!")
                    break
            else:
                # If no validation set, just save based on training loss
                val_loss = train_loss
                is_best = epoch == 0 or val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
            
            # Save checkpoint periodically
            if (epoch + 1) % self.args.save_every == 0 or epoch == num_epochs - 1:
                checkpoint_path = f"{self.args.save_dir}/model_epoch_{epoch+1:03d}.pt"
                self.save_checkpoint(path=checkpoint_path, epoch=epoch, val_loss=val_loss)
            
            # Save best model
            if is_best:
                best_model_path = os.path.join(self.args.save_dir, "model_best.pt")
                self.save_checkpoint(path=best_model_path, epoch=epoch, val_loss=val_loss, is_best=True)
        
        print("Training completed!")
        
        # Load best model for final return if we were validating
        if val_dataloader is not None and os.path.exists(os.path.join(self.args.save_dir, "model_best.pt")):
            try:
                best_path = os.path.join(self.args.save_dir, "model_best.pt")
                self.load_checkpoint(best_path)
                print("Loaded the best model for final return")
            except Exception as e:
                print(f"Could not load best model: {e}")
                
        return self.model
    
    def set_evaluator(self, evaluator):
        """Set the evaluator for the trainer."""
        self.evaluator = evaluator 