import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import os
import requests
from io import BytesIO

def tensor_to_image(tensor):
    """
    Convert a tensor to a numpy image
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [C, H, W] in range [-1, 1]
        
    Returns:
        numpy.ndarray: Output image of shape [H, W, C] in range [0, 255]
    """
    # Ensure tensor is on CPU and detached
    img = tensor.cpu().detach()
    
    # Convert to range [0, 1]
    img = (img + 1) / 2
    
    # Convert to numpy and transpose
    img = img.numpy().transpose(1, 2, 0)
    
    # Clip values to [0, 1]
    img = np.clip(img, 0, 1)
    
    # Convert to uint8
    img = (img * 255).astype(np.uint8)
    
    return img

def image_to_tensor(img, normalize=True):
    """
    Convert a numpy image or PIL image to a tensor
    
    Args:
        img (numpy.ndarray or PIL.Image): Input image
        normalize (bool): Whether to normalize to [-1, 1]
        
    Returns:
        torch.Tensor: Output tensor of shape [C, H, W]
    """
    if isinstance(img, np.ndarray):
        if img.ndim == 2:  # Grayscale
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)  # Convert to RGB
        
        # Convert to PIL
        img = Image.fromarray(img)
    
    # Define transform
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.ToTensor()
        
    return transform(img)

def generate_edge_map(img, low_threshold=100, high_threshold=200):
    """
    Generate edge map from an image using Canny edge detector
    
    Args:
        img (numpy.ndarray or PIL.Image): Input image
        low_threshold (int): Lower threshold for Canny edge detector
        high_threshold (int): Upper threshold for Canny edge detector
        
    Returns:
        numpy.ndarray: Edge map
    """
    # Convert to numpy if necessary
    if isinstance(img, torch.Tensor):
        img = tensor_to_image(img)
    elif isinstance(img, Image.Image):
        img = np.array(img)
    
    # Convert to grayscale if needed
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
        
    # Apply Canny edge detector
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    return edges

def generate_pose_map(img, use_mediapipe=True):
    """
    Generate pose map from an image using MediaPipe or a simpler approach
    
    Args:
        img (numpy.ndarray or PIL.Image): Input image
        use_mediapipe (bool): Whether to use MediaPipe (if available)
        
    Returns:
        numpy.ndarray: Pose map
    """
    # Convert to numpy if necessary
    if isinstance(img, torch.Tensor):
        img = tensor_to_image(img)
    elif isinstance(img, Image.Image):
        img = np.array(img)
    
    # Make a copy of the image to draw on
    h, w = img.shape[:2]
    pose_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    try:
        if use_mediapipe:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            
            # Initialize mediapipe
            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                # Convert to RGB
                if img.shape[2] == 3:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = img
                
                # Process the image
                results = pose.process(img_rgb)
                
                # Draw the pose landmarks on the pose map
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        pose_map,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    return pose_map
    except ImportError:
        print("MediaPipe not installed, using fallback pose estimation")
    except Exception as e:
        print(f"Error using MediaPipe: {e}")
    
    # Fallback: basic pose estimation using contours and shape detection
    # Convert to grayscale
    if img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    large_contours = [c for c in contours if cv2.contourArea(c) > 500]
    
    # Draw contours on pose map
    cv2.drawContours(pose_map, large_contours, -1, (0, 255, 0), 2)
    
    return pose_map

def generate_segmentation_map(img, method="grabcut"):
    """
    Generate segmentation map from an image
    
    Args:
        img (numpy.ndarray or PIL.Image): Input image
        method (str): Segmentation method ('grabcut', 'watershed', or 'kmeans')
        
    Returns:
        numpy.ndarray: Segmentation map
    """
    # Convert to numpy if necessary
    if isinstance(img, torch.Tensor):
        img = tensor_to_image(img)
    elif isinstance(img, Image.Image):
        img = np.array(img)
    
    # Ensure image is in the right format
    if img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    
    h, w = img.shape[:2]
    
    if method == "grabcut":
        # Initialize mask
        mask = np.zeros(img.shape[:2], np.uint8)
        
        # Set rectangle for foreground
        rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
        
        # Initialize background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create mask where sure and probable foreground are set to 1, rest to 0
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Create colored segmentation map
        segmap = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create colored segments
        segmap[mask2 == 1] = [255, 0, 0]  # Foreground in red
    
    elif method == "watershed":
        # Convert to grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labeling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all labels so that background is not 0, but 1
        markers = markers + 1
        
        # Mark unknown with 0
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(img_bgr, markers)
        
        # Create segmentation map
        segmap = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Color segments
        for i in range(2, np.max(markers) + 1):
            segmap[markers == i] = np.random.randint(0, 255, 3)
    
    elif method == "kmeans":
        # Reshape image for k-means
        Z = img_bgr.reshape((-1, 3))
        Z = np.float32(Z)
        
        # Define criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # Apply k-means
        K = 5
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and reshape
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        segmap = res.reshape((img.shape))
    
    else:
        raise ValueError(f"Unsupported segmentation method: {method}")
    
    return segmap

def visualize_batch(batch, denormalize=True, max_images=16):
    """
    Visualize a batch of images and conditions
    
    Args:
        batch (dict): Batch of data with 'image' and 'condition' keys
        denormalize (bool): Whether to denormalize images
        max_images (int): Maximum number of images to display
        
    Returns:
        numpy.ndarray: Grid of images
    """
    images = batch["image"][:max_images]
    conditions = batch["condition"][:max_images]
    
    # Denormalize if needed
    if denormalize:
        images = (images + 1) / 2
        conditions = (conditions + 1) / 2
    
    # Create a grid of images and conditions side by side
    grid_images = []
    for i in range(min(len(images), max_images)):
        grid_images.append(conditions[i])
        grid_images.append(images[i])
    
    # Create grid
    grid = make_grid(grid_images, nrow=4, padding=2)
    
    # Convert to numpy
    grid = grid.cpu().numpy().transpose(1, 2, 0)
    
    return grid

def save_image_grid(grid, path):
    """
    Save a grid of images
    
    Args:
        grid (numpy.ndarray): Grid of images
        path (str): Path to save the grid
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def convert_to_condition(img, condition_type="edge", **kwargs):
    """
    Convert an image to a conditioning signal
    
    Args:
        img (numpy.ndarray or PIL.Image): Input image
        condition_type (str): Type of conditioning signal
        **kwargs: Additional arguments for the specific condition type
        
    Returns:
        numpy.ndarray: Conditioning signal
    """
    # Convert to numpy if needed
    if isinstance(img, torch.Tensor):
        img = tensor_to_image(img)
    elif isinstance(img, Image.Image):
        img = np.array(img)
    
    # Generate condition based on type
    if condition_type == "edge":
        low_threshold = kwargs.get('low_threshold', 100)
        high_threshold = kwargs.get('high_threshold', 200)
        condition = generate_edge_map(img, low_threshold, high_threshold)
        
        # Convert to RGB for consistency
        condition = np.stack([condition] * 3, axis=2)
        
    elif condition_type == "pose":
        use_mediapipe = kwargs.get('use_mediapipe', True)
        condition = generate_pose_map(img, use_mediapipe)
        
    elif condition_type == "segmentation":
        method = kwargs.get('method', 'grabcut')
        condition = generate_segmentation_map(img, method)
        
    else:
        raise ValueError(f"Unsupported condition type: {condition_type}")
        
    return condition 

def get_advanced_augmentations(augmentation_strength='medium'):
    """
    Get advanced image augmentation transforms with different strength levels
    
    Args:
        augmentation_strength (str): Strength of augmentations ('light', 'medium', or 'heavy')
        
    Returns:
        transforms.Compose: Composed transforms for image augmentation
    """
    if augmentation_strength == 'light':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])
    
    elif augmentation_strength == 'medium':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ])
    
    elif augmentation_strength == 'heavy':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.RandomAffine(degrees=(-20, 20), translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-10, 10)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.4),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        ])
    
    else:
        raise ValueError(f"Unsupported augmentation strength: {augmentation_strength}") 