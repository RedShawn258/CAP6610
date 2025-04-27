'''
This file is used to load the COCO dataset and generate the condition 
for the controllable image generation.
'''

import os
import json
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.image_utils import (
    tensor_to_image, 
    image_to_tensor, 
    get_advanced_augmentations
)

class COCODataset(Dataset):
    def __init__(
        self,
        data_dir="data/coco",
        split="val2017",
        image_size=256,
        control_type="edge",
        use_small_subset=False,
        max_images=None,
        augment=True,
        augmentation_strength='medium'
    ):
        """
        Dataset for controllable image generation using COCO
        
        Args:
            data_dir (str): Directory containing the dataset
            split (str): Data split to use ('train2017' or 'val2017')
            image_size (int): Size of the images
            control_type (str): Type of control signal to use
            use_small_subset (bool): Whether to use a small subset of the data
            max_images (int): Maximum number of images to use
            augment (bool): Whether to use data augmentation
            augmentation_strength (str): Strength of augmentations ('light', 'medium', or 'heavy')
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.control_type = control_type.lower()
        self.use_small_subset = use_small_subset
        self.max_images = max_images
        self.augment = augment
        self.augmentation_strength = augmentation_strength
        
        # Set up directories
        self.img_dir = os.path.join(data_dir, split)
        self.condition_dir = os.path.join(data_dir, f"{control_type}_conditions_{split}")
        
        # Download dataset if needed
        if not os.path.exists(self.img_dir):
            print("COCO dataset not found. Downloading...")
            self.download_coco()
            
        # Create condition directory if needed
        os.makedirs(self.condition_dir, exist_ok=True)
        
        # Get image filenames from annotations
        self.anno_path = os.path.join(data_dir, 'annotations', f'instances_{split}.json')
        with open(self.anno_path, 'r') as f:
            annotations = json.load(f)
            
        self.image_info = annotations['images']
        
        # Use small subset or limit number of images if specified
        if use_small_subset:
            self.image_info = self.image_info[:1000]  # Use first 1000 images for faster testing
        elif max_images is not None:
            self.image_info = self.image_info[:max_images]
            
        # Set up basic transforms
        self.basic_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        if self.augment:
            self.augmentation_transform = get_advanced_augmentations(augmentation_strength)
        else:
            self.augmentation_transform = None
            
        print(f"COCO {split} dataset loaded with {len(self.image_info)} images")
        
    def __len__(self):
        return len(self.image_info)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset
        
        Args:
            idx (int): Index of the item
            
        Returns:
            dict: Dictionary with 'image' and 'condition' keys
        """
        img_info = self.image_info[idx]
        img_filename = img_info['file_name']
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_filename)
        
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
            condition = self.load_or_generate_condition(img_id, img_np)
            condition_tensor = image_to_tensor(condition)
            
            return {
                'image': img_tensor,
                'condition': condition_tensor
            }
    
    def download_coco(self):
        """Download COCO dataset if not available"""
        from pycocotools.coco import COCO
        import subprocess
        import os
        
        # Create directory
        os.makedirs(self.img_dir, exist_ok=True)
        
        # Set URLs
        if self.split == "train2017":
            image_url = "http://images.cocodataset.org/zips/train2017.zip"
            anno_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        else:
            image_url = "http://images.cocodataset.org/zips/val2017.zip"
            anno_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        
        # Download images
        image_zip = os.path.join(self.data_dir, f"{self.split}.zip")
        if not os.path.exists(image_zip):
            print(f"Downloading COCO {self.split} images...")
            subprocess.run(["wget", image_url, "-O", image_zip])
            
        # Download annotations
        anno_zip = os.path.join(self.data_dir, "annotations.zip")
        if not os.path.exists(anno_zip):
            print(f"Downloading COCO annotations...")
            subprocess.run(["wget", anno_url, "-O", anno_zip])
            
        # Extract images
        if len(os.listdir(self.img_dir)) == 0:
            print(f"Extracting COCO {self.split} images...")
            subprocess.run(["unzip", "-q", image_zip, "-d", self.data_dir])
            
        # Extract annotations
        anno_dir = os.path.join(self.data_dir, "annotations")
        if not os.path.exists(anno_dir):
            print(f"Extracting COCO annotations...")
            subprocess.run(["unzip", "-q", anno_zip, "-d", self.data_dir])
        
        print(f"COCO dataset extraction completed")
    
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
    
    def load_or_generate_condition(self, image_id, image_np):
        """Load cached condition or generate a new one and cache it"""
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
        condition.save(condition_path)
        
        return condition 