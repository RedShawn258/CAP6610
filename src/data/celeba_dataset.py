'''
This file is used to load the CelebA dataset and generate the condition 
for the controllable image generation.
'''

import os
import zipfile
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import gdown

from src.utils.image_utils import (
    tensor_to_image,
    image_to_tensor,
    get_advanced_augmentations
)

class CelebADataset(Dataset):
    """
    CelebA dataset for controllable image generation
    """
    def __init__(
        self,
        data_dir="data/celeba",
        image_size=256,
        control_type="edge",
        use_small_subset=False,
        max_images=None,
        augment=True,
        augmentation_strength='medium'
    ):
        """
        Dataset for controllable image generation using CelebA
        
        Args:
            data_dir (str): Directory containing the dataset
            image_size (int): Size of the images
            control_type (str): Type of control signal to use
            use_small_subset (bool): Whether to use a small subset of the data
            max_images (int): Maximum number of images to use
            augment (bool): Whether to use data augmentation
            augmentation_strength (str): Strength of augmentations ('light', 'medium', or 'heavy')
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.control_type = control_type.lower()
        self.use_small_subset = use_small_subset
        self.max_images = max_images
        self.augment = augment
        self.augmentation_strength = augmentation_strength

        # Set up directories
        self.img_dir = os.path.join(data_dir, "img_align_celeba")
        self.condition_dir = os.path.join(data_dir, f"{control_type}_conditions")

        # Download dataset if needed
        if not os.path.exists(self.img_dir):
            print("CelebA dataset not found. Downloading...")
            self.download_celeba()

        # Create condition directory if needed
        os.makedirs(self.condition_dir, exist_ok=True)

        # Get image filenames
        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])

        # Use small subset or limit number of images if specified
        if use_small_subset:
            self.image_files = self.image_files[:1000]  # Use first 1000 images for faster testing
        elif max_images is not None:
            self.image_files = self.image_files[:max_images]

        # Set up basic transforms
        self.basic_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        if self.augment:
            self.augmentation_transform = get_advanced_augmentations(augmentation_strength)
        else:
            self.augmentation_transform = None

        print(f"CelebA dataset loaded with {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get an item from the dataset
        
        Args:
            idx (int): Index of the item
            
        Returns:
            tuple: (image, condition)
        """
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        
        # Load image
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            
            # Apply basic resize transform
            img_tensor = self.basic_transform(img)
            
            # Apply augmentation if enabled
            if self.augment and self.augmentation_transform is not None:
                img_tensor = self.augmentation_transform(img_tensor)
                
            # Convert tensor back to image for condition generation
            img_np = tensor_to_image(img_tensor)
            
            # Generate condition
            condition = self.load_or_generate_condition(self.image_files[idx], img_np)
            condition_tensor = image_to_tensor(condition)
            
            return img_tensor, condition_tensor
    
    def download_celeba(self):
        """Download CelebA dataset if not available"""
        import subprocess
        
        print("Downloading CelebA dataset...")
        os.makedirs(self.img_dir, exist_ok=True)
        
        # URLs for CelebA
        gdrive_url = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
        
        # Download using gdown if available, or wget otherwise
        try:
            print("Using gdown to download CelebA...")
            zip_path = os.path.join(self.img_dir, "img_align_celeba.zip")
            gdown.download(gdrive_url, zip_path, quiet=False)
        except ImportError:
            print("gdown not installed, using alternative method...")
            # Alternative download method
            try:
                # Try using curl with cookies to bypass Google Drive limitations
                print("Using curl to download CelebA (this may take a while)...")
                subprocess.run([
                    "curl", 
                    "https://docs.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM",
                    "-L", 
                    "-o", os.path.join(self.img_dir, "img_align_celeba.zip")
                ])
            except:
                print("Download failed. You may need to manually download CelebA dataset.")
                print("Please download from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
                print(f"And extract to: {self.img_dir}")
                return
        
        # Extract the zip file
        print("Extracting CelebA dataset...")
        zip_path = os.path.join(self.img_dir, "img_align_celeba.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(self.img_dir)
            print(f"Extracted CelebA dataset to {self.img_dir}")
    
    def generate_edge_map(self, image):
        """Generate edge map from image using Canny edge detector"""
        # Convert to numpy array if it's a PIL Image
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Convert to grayscale if needed
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
            
        # Apply Canny edge detector
        edges = cv2.Canny(gray, 100, 200)
        
        # Convert back to PIL Image
        return Image.fromarray(edges)
    
    def generate_condition_from_image(self, image):
        """Generate condition directly from an image without caching"""
        # Generate condition based on control type
        if self.control_type == "edge":
            condition = self.generate_edge_map(image)
            # Keep edge map as single-channel grayscale
            condition = Image.fromarray(np.array(condition))
        elif self.control_type == "pose":
            # Use image_utils to generate pose map
            image_np = np.array(image)
            from src.utils.image_utils import generate_pose_map
            pose_map = generate_pose_map(image_np)
            condition = Image.fromarray(pose_map)
        elif self.control_type == "segmentation":
            # Use image_utils to generate segmentation map
            image_np = np.array(image)
            from src.utils.image_utils import generate_segmentation_map
            seg_map = generate_segmentation_map(image_np)
            condition = Image.fromarray(seg_map)
        else:
            raise ValueError(f"Unsupported control type: {self.control_type}")
        
        return condition

    def load_or_generate_condition(self, image_filename, image_np):
        """Load cached condition or generate a new one and cache it"""
        image_id = image_filename.split('.')[0]
        condition_path = os.path.join(self.condition_dir, f"{image_id}.png")
        
        # Check if condition already generated
        if os.path.exists(condition_path):
            try:
                return Image.open(condition_path).convert('RGB')
            except:
                # If we can't open it, regenerate
                pass
        
        # Generate condition
        condition = self.generate_condition_from_image(Image.fromarray(image_np))
        
        # Save generated condition
        os.makedirs(os.path.dirname(condition_path), exist_ok=True)
        condition.save(condition_path)
        
        return condition 