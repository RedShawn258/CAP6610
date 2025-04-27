import os
import argparse
import torch
from torch.utils.data import DataLoader

# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Controllable Image Generation using Diffusion Models")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "celeba"], 
                        help="Dataset to use")
    parser.add_argument("--data_dir", type=str, default="./data", 
                        help="Directory containing the datasets")
    parser.add_argument("--image_size", type=int, default=256, 
                        help="Image resolution for training")
    parser.add_argument("--control_type", type=str, default="edge", 
                        choices=["edge", "pose", "segmentation"], 
                        help="Type of conditioning signal")
    parser.add_argument("--use_small_subset", action="store_true", 
                        help="Use a small subset of images for testing")
    parser.add_argument("--max_images", type=int, default=100, 
                        help="Maximum number of images to use if use_small_subset is True")
    parser.add_argument("--augment", action="store_true",
                        help="Use augmentations during training")
    parser.add_argument("--augmentation_strength", type=str, default="medium",
                        choices=["light", "medium", "heavy"],
                        help="Strength of augmentations to apply if augment is True")
    
    # Model parameters
    parser.add_argument("--latent_dim", type=int, default=4, 
                        help="Latent space dimensionality factor")
    parser.add_argument("--model_type", type=str, default="ldm", 
                        choices=["ldm", "controlnet"], 
                        help="Type of diffusion model to use")
    parser.add_argument("--pretrained_model", type=str, default=None, 
                        help="Path to pretrained model if using")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, 
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", 
                        help="Directory to save logs")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=0.001, 
                        help="Minimum change in validation loss to be considered as improvement")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    # Evaluation parameters
    parser.add_argument("--eval_every", type=int, default=5, 
                        help="Evaluate every N epochs")
    parser.add_argument("--num_samples", type=int, default=16, 
                        help="Number of samples to generate for evaluation")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create necessary directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Import components based on user selection
    if args.dataset == "coco":
        from src.data.coco_dataset import COCODataset as Dataset
    else:
        from src.data.celeba_dataset import CelebADataset as Dataset
    
    if args.model_type == "ldm":
        from src.models.latent_diffusion import LatentDiffusion as Model
    else:
        from src.models.controlnet import ControlNet as Model
    
    from src.training.trainer import Trainer
    from src.evaluation.evaluator import Evaluator
    
    print(f"Loading {args.dataset} dataset with {args.control_type} conditioning...")
    
    # Create dataset and dataloader
    dataset = Dataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        control_type=args.control_type,
        use_small_subset=args.use_small_subset,
        max_images=args.max_images,
        augment=args.augment,
        augmentation_strength=args.augmentation_strength
    )
    
    # Split dataset into training and validation sets (90% train, 10% val)
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.9)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    print(f"Creating {args.model_type} model...")
    
    # Create model
    if args.model_type == "ldm":
        model = Model(
            latent_dim=args.latent_dim,
            image_size=args.image_size,
            pretrained_path=args.pretrained_model
        ).to(device)
    else:
        # For controlnet, create a more limited model to avoid recursion issues
        from src.models.controlnet import ControlNet
        model = ControlNet(
            latent_dim=args.latent_dim,
            image_size=args.image_size,
            input_channels=3,
            condition_channels=1,
            model_channels=64,
            time_dim=128,
            num_res_blocks=2,
            channel_mult=(1, 2, 4, 4),
            pretrained_path=args.pretrained_model
        ).to(device)
    
    # Create evaluator for metrics and visualization
    evaluator = Evaluator(model, dataset, device, args)
    
    # Create trainer
    trainer = Trainer(
        model=model, 
        dataloader=train_dataloader, 
        device=device, 
        args=args,
        patience=args.patience if hasattr(args, 'patience') else 10  # Default patience of 10
    )
    
    # Set evaluator in trainer for visualization during training
    trainer.set_evaluator(evaluator)
    
    # Train the model
    print("Starting training...")
    trainer.train(args.num_epochs, val_dataloader)
    
    print("Training completed!")
    
    # Generate final samples
    print("Generating final samples...")
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Get a batch from the dataset for conditioning
    sample_batch = next(iter(val_dataloader))
    condition = sample_batch["condition"].to(device)
    
    # Generate samples
    with torch.no_grad():
        samples = model.sample(condition, batch_size=args.batch_size)
    
    # Save images
    import torchvision.utils as vutils
    grid = vutils.make_grid(samples, normalize=True, value_range=(-1, 1), nrow=int(args.batch_size**0.5))
    vutils.save_image(grid, f"{args.log_dir}/final_samples.png")
    print(f"Samples saved to {args.log_dir}/final_samples.png")

if __name__ == "__main__":
    main() 