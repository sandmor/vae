import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json
from transformers import Dinov2Model
from pathlib import Path
from typing import Optional, Tuple, List
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CatDataset(Dataset):
    """
    Custom dataset for cat images with flexible preprocessing
    """
    
    def __init__(self, 
                 root_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 image_size: int = 224,
                 max_images: Optional[int] = None,
                 valid_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')):
        """
        Initialize Cat Dataset
        
        Args:
            root_dir: Path to dataset directory
            transform: Optional transform to apply to images
            image_size: Target image size (assumes square images)
            max_images: Maximum number of images to load (for testing)
            valid_extensions: Valid image file extensions
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.max_images = max_images
        self.valid_extensions = valid_extensions
        
        # Find all image files
        self.image_paths = self._find_images()
        
        # Set up transforms
        self.transform = transform if transform else self._default_transform()
        
        logger.info(f"Found {len(self.image_paths)} cat images in {root_dir}")
        
    def _find_images(self) -> List[Path]:
        """Find all valid image files in the dataset directory"""
        image_paths = []
        
        # Handle different possible dataset structures
        if (self.root_dir / "PetImages" / "Cat").exists():
            # Microsoft Cat vs Dog dataset structure
            cat_dir = self.root_dir / "PetImages" / "Cat"
            search_dirs = [cat_dir]
        elif (self.root_dir / "cats").exists():
            # Simple cats directory structure
            search_dirs = [self.root_dir / "cats"]
        else:
            # Search in root directory and subdirectories
            search_dirs = [self.root_dir]
        
        for search_dir in search_dirs:
            for ext in self.valid_extensions:
                pattern = f"*{ext}"
                found_files = list(search_dir.rglob(pattern))
                image_paths.extend(found_files)
        
        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))
        
        # Filter corrupted images
        image_paths = self._filter_valid_images(image_paths)
        
        # Limit number of images if specified
        if self.max_images:
            image_paths = image_paths[:self.max_images]
            
        return image_paths
    
    def _filter_valid_images(self, image_paths: List[Path]) -> List[Path]:
        """Filter out corrupted or invalid images"""
        valid_paths = []
        
        for path in image_paths:
            try:
                # Quick check: file size should be reasonable
                if path.stat().st_size < 1024:  # Less than 1KB
                    continue
                    
                # Try to open image
                with Image.open(path) as img:
                    # Check if image can be loaded and has reasonable dimensions
                    if img.size[0] < 32 or img.size[1] < 32:
                        continue
                    # Convert to RGB to ensure compatibility
                    img.convert('RGB')
                    
                valid_paths.append(path)
                
            except Exception as e:
                logger.warning(f"Skipping corrupted image {path}: {e}")
                continue
        
        logger.info(f"Filtered {len(image_paths)} -> {len(valid_paths)} valid images")
        return valid_paths
    
    def _default_transform(self) -> transforms.Compose:
        """Default image preprocessing pipeline"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single item from the dataset
        
        Returns:
            image: Preprocessed image tensor
            label: Dummy label (always 0 for unsupervised learning)
        """
        image_path = self.image_paths[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Return image with dummy label (VAE doesn't need labels)
            return image, 0
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            fallback_image = torch.zeros(3, self.image_size, self.image_size)
            return fallback_image, 0
    
    def get_sample_images(self, num_samples: int = 8) -> List[torch.Tensor]:
        """Get sample images for visualization"""
        indices = np.random.choice(len(self), min(num_samples, len(self)), replace=False)
        samples = []
        
        for idx in indices:
            image, _ = self[idx]
            samples.append(image)
            
        return samples

def get_data_transforms(image_size: int = 224, 
                       augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation transforms
    
    Args:
        image_size: Target image size
        augment: Whether to apply data augmentation for training
    
    Returns:
        train_transform, val_transform
    """
    
    # Base transforms
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    
    # Training transforms with augmentation
    if augment:
        train_transforms = [
            transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),  # Slightly larger
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    else:
        train_transforms = base_transforms
    
    # Validation transforms (no augmentation)
    val_transforms = base_transforms
    
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


def create_data_loaders(dataset_path: Optional[str] = None,
                       batch_size: int = 32,
                       image_size: int = 224,
                       train_split: float = 0.8,
                       augment: bool = True,
                       num_workers: int = 4,
                       max_images: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        dataset_path: Path to dataset (if None, loads from environment)
        batch_size: Batch size for data loaders
        image_size: Target image size
        train_split: Fraction of data to use for training
        augment: Whether to apply data augmentation
        num_workers: Number of worker processes for data loading
        max_images: Maximum number of images to load (for testing)
    
    Returns:
        train_loader, val_loader
    """
    
    # Get dataset path from environment if not provided
    if dataset_path is None:
        dataset_path = os.getenv('CAT_DATASET_PATH')
        if dataset_path is None:
            raise ValueError("CAT_DATASET_PATH environment variable not set and no path provided")
    
    # Verify path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(image_size, augment)
    
    # Create full dataset
    full_dataset = CatDataset(
        root_dir=dataset_path,
        transform=None,  # We'll apply transforms separately
        image_size=image_size,
        max_images=max_images
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Apply different transforms to splits
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Ensures consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Training batches: {len(train_loader)}")
    logger.info(f"  Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader

def visualize_batch(data_loader: DataLoader, 
                   num_images: int = 8,
                   figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Visualize a batch of images from the data loader
    
    Args:
        data_loader: DataLoader to sample from
        num_images: Number of images to display
        figsize: Figure size for matplotlib
    """
    
    # Get a batch
    batch, _ = next(iter(data_loader))
    
    # Select subset of images
    images = batch[:num_images]
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    
    for i, img in enumerate(images):
        # Convert from [-1, 1] to [0, 1] for display
        img_display = (img + 1) / 2
        img_display = torch.clamp(img_display, 0, 1)
        
        # Convert to numpy and transpose for matplotlib
        img_np = img_display.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].axis('off')
        axes[i].set_title(f'Image {i+1}')
    
    plt.tight_layout()
    plt.show()

class VAE(nn.Module):
    def __init__(self, latent_dim=512, image_size=224, projection_multiplier=2.0, min_hidden_dim=512):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size

        self.dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.dinov2_dim = self.dinov2.config.hidden_size

        # Freeze the DINOv2 model parameters
        for param in self.dinov2.parameters():
            param.requires_grad = False
        
        self.encoder_projection = self._build_encoder_projection(
            latent_dim, projection_multiplier, min_hidden_dim
        )

        self.mu_layer = nn.Linear(latent_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, self.latent_dim)

        self.decoder = self._build_decoder()
    
    def _build_encoder_projection(self, latent_dim, projection_multiplier, min_hidden_dim):
        hidden_dim = max(min_hidden_dim, int(latent_dim * projection_multiplier))
        
        if latent_dim >= self.dinov2_dim:
            hidden_dim = max(hidden_dim, int(self.dinov2_dim * 1.5))
        
        return nn.Sequential(
            nn.Linear(self.dinov2_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    
    def _build_decoder(self):
        if self.latent_dim <= 256:
            spatial_dim = 7
            initial_channels = min(512, self.latent_dim * 2)
        elif self.latent_dim <= 512:
            spatial_dim = 7
            initial_channels = 512
        else:
            spatial_dim = 8
            initial_channels = min(768, self.latent_dim)
        
        initial_spatial_dim = spatial_dim * spatial_dim * initial_channels

        decoder_layers = []

        expansion_dims = self._calculate_expansion_dims(
            self.latent_dim,
            initial_spatial_dim
        )

        for i in range(len(expansion_dims) - 1):
            decoder_layers.extend([
                nn.Linear(expansion_dims[i], expansion_dims[i + 1]),
                nn.ReLU(),
            ])
        
        decoder_layers.append(Reshape(-1, initial_channels, spatial_dim, spatial_dim))

        conv_layers = self._build_conv_layers(initial_channels, spatial_dim)
        decoder_layers.extend(conv_layers)

        return nn.Sequential(*decoder_layers)
    
    def _calculate_expansion_dims(self, start_dim, end_dim):
        if start_dim >= end_dim:
            return start_dim, end_dim
        
        expansion_factor = 2.0
        max_layers = 5

        dims = [start_dim]
        current_dim = start_dim
        layer_count = 0

        while current_dim < end_dim and layer_count < max_layers:
            current_dim = min(int(current_dim * expansion_factor), end_dim)
            dims.append(current_dim)
            layer_count += 1
        
        if dims[-1] < end_dim:
            dims.append(end_dim)
        
        return dims
    
    def _build_conv_layers(self, initial_channels, spatial_dim):
        """
        Build transposed convolution layers that scale to target image size
        """
        layers = []
        current_channels = initial_channels
        target_spatial = self.image_size
        
        # Calculate exact upsampling path
        upsampling_path = self._calculate_upsampling_path(spatial_dim, target_spatial)
        
        # Channel schedule: reduce channels as we increase spatial resolution
        channel_schedule = []
        for i in range(len(upsampling_path) - 1):
            # Exponential decay in channels
            channels = max(16, current_channels // (2 ** i))
            channel_schedule.append(channels)
        
        # Add transposed convolution layers
        for i, (current_size, next_size) in enumerate(zip(upsampling_path[:-1], upsampling_path[1:])):
            out_channels = channel_schedule[i]
            
            # Calculate kernel size, stride, and padding for exact size
            kernel_size, stride, padding = self._calculate_conv_params(current_size, next_size)
            
            layers.extend([
                nn.ConvTranspose2d(
                    current_channels, out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ])
            current_channels = out_channels
        
        # Final layer to RGB with adaptive interpolation if needed
        layers.extend([
            AdaptiveUpsampler(current_channels, 3, target_spatial),
            nn.Tanh()
        ])
        
        return layers
    
    def _calculate_upsampling_path(self, start_size, target_size):
        """
        Calculate the optimal upsampling path from start_size to target_size
        """
        path = [start_size]
        current = start_size
        
        while current < target_size:
            # Try to double the size, but don't exceed target
            next_size = min(current * 2, target_size)
            
            # If we can't double exactly to target, use intermediate steps
            if next_size == target_size and current * 2 != target_size:
                # Add intermediate step if the jump is too large
                if target_size / current > 2.5:
                    intermediate = current * 2
                    if intermediate < target_size:
                        path.append(intermediate)
                        current = intermediate
                        continue
            
            path.append(next_size)
            current = next_size
        
        return path
    
    def _calculate_conv_params(self, input_size, output_size):
        """
        Calculate kernel_size, stride, and padding for exact output size
        Formula: output_size = (input_size - 1) * stride - 2 * padding + kernel_size
        """
        if output_size == input_size * 2:
            # Standard 2x upsampling
            return 4, 2, 1
        elif output_size == input_size:
            # No upsampling
            return 3, 1, 1
        else:
            # Custom upsampling - use adaptive approach
            stride = max(1, output_size // input_size)
            kernel_size = max(3, stride + 2)
            padding = (stride * (input_size - 1) + kernel_size - output_size) // 2
            return kernel_size, stride, max(0, padding)

    def encode(self, x):
        with torch.set_grad_enabled(False):
            dinov2_outputs = self.dinov2(x)
            features = dinov2_outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, 768)
        
        projected_features = self.encoder_projection(features)
        mu = self.mu_layer(projected_features)
        logvar = self.logvar_layer(projected_features)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

    def sample(self, num_samples, device):
        """Generate new samples from the latent space"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def get_architecture_info(self):
        """
        Return information about the architecture for debugging/analysis
        """
        # Calculate projection dimensions
        projection_info = []
        for i, layer in enumerate(self.encoder_projection):
            if isinstance(layer, nn.Linear):
                projection_info.append(f"Linear {layer.in_features} -> {layer.out_features}")
        
        # Calculate decoder expansion
        decoder_info = []
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.Linear):
                decoder_info.append(f"Linear {layer.in_features} -> {layer.out_features}")
            elif isinstance(layer, nn.ConvTranspose2d):
                decoder_info.append(f"ConvTranspose2d {layer.in_channels} -> {layer.out_channels}")
        
        return {
            'latent_dim': self.latent_dim,
            'dinov2_dim': self.dinov2_dim,
            'projection_layers': projection_info,
            'decoder_layers': decoder_info,
            'total_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class AdaptiveUpsampler(nn.Module):
    """
    Adaptive upsampler that ensures exact output size
    """
    def __init__(self, in_channels, out_channels, target_size):
        super().__init__()
        self.target_size = target_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Apply convolution first
        x = self.conv(x)
        
        # Get current size
        current_size = x.size(-1)  # Assumes square images
        
        # If size doesn't match target, use interpolation
        if current_size != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), 
                            mode='bilinear', align_corners=False)
        
        return x

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function combining reconstruction loss and KL divergence
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (beta-VAE)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_epoch(model, dataloader, optimizer, device, beta=1.0):
    """
    Train one epoch with improved monitoring
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, beta)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon_loss = total_recon_loss / len(dataloader.dataset)
    avg_kl_loss = total_kl_loss / len(dataloader.dataset)
    
    return avg_loss, avg_recon_loss, avg_kl_loss

class VAETrainer:
    """
    Complete training and testing pipeline for Cat VAE
    """
    
    def __init__(self, 
                 model_config: dict,
                 training_config: dict,
                 data_config: dict,
                 save_dir: str = "experiments"):
        """
        Initialize the trainer
        
        Args:
            model_config: Model configuration parameters
            training_config: Training hyperparameters
            data_config: Data loading configuration
            save_dir: Directory to save results
        """
        
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.save_dir = Path(save_dir)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.save_dir / f"vae_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configs
        self._save_configs()
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []
        self.val_recon_losses = []
        self.val_kl_losses = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        logger.info(f"Experiment directory: {self.experiment_dir}")
        
        # Log parameter information
        params_info = self._get_trainable_params_info()
        logger.info(f"Model parameters:")
        logger.info(f"  Total: {params_info['total_params']:,}")
        logger.info(f"  Trainable: {params_info['trainable_params']:,}")
        logger.info(f"  DINOv2 (frozen): {params_info['dinov2_params']:,}")
        logger.info(f"  Savable in checkpoint: {params_info['savable_params']:,}")
        logger.info(f"Trainable parameters (excluding DINOv2): {self._get_trainable_params_info()['trainable_params']:,}")
    
    def _get_trainable_params_info(self):
        """Get information about trainable parameters (excluding DINOv2)"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        dinov2_params = sum(p.numel() for p in self.model.dinov2.parameters())
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'dinov2_params': dinov2_params,
            'savable_params': total_params - dinov2_params
        }
    
    def _save_configs(self):
        """Save configuration files"""
        configs = {
            'model_config': self.model_config,
            'training_config': self.training_config,
            'data_config': self.data_config
        }
        
        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(configs, f, indent=2)
    
    def _create_model(self) -> VAE:
        """Create and initialize the model"""
        model = VAE(**self.model_config)
        model.to(self.device)
        
        # Log model architecture
        arch_info = model.get_architecture_info()
        logger.info("Model Architecture:")
        for key, value in arch_info.items():
            logger.info(f"  {key}: {value}")
        
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with different learning rates for encoder/decoder"""
        
        # Separate parameters
        encoder_params = (
            list(self.model.encoder_projection.parameters()) +
            list(self.model.mu_layer.parameters()) +
            list(self.model.logvar_layer.parameters())
        )
        decoder_params = list(self.model.decoder.parameters())
        
        # Add DINOv2 parameters if not frozen
        dinov2_params = []
        if not self.model_config.get('freeze_encoder', True):
            dinov2_params = list(self.model.dinov2.parameters())
        
        # Create parameter groups
        param_groups = [
            {'params': encoder_params, 'lr': self.training_config['encoder_lr']},
            {'params': decoder_params, 'lr': self.training_config['decoder_lr']}
        ]
        
        if dinov2_params:
            param_groups.append({
                'params': dinov2_params, 
                'lr': self.training_config.get('dinov2_lr', 1e-6)
            })
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.training_config.get('weight_decay', 1e-4)
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_config = self.training_config.get('scheduler', {})
        
        if scheduler_config.get('type') == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_config.get('type') == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        else:
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
            )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        # Progressive beta schedule for beta-VAE
        beta = self._get_beta_schedule(epoch)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.training_config["epochs"]}')
        
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            
            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, beta)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.training_config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.training_config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}',
                'Beta': f'{beta:.4f}'
            })
        
        # Calculate averages
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def validate(self, val_loader: DataLoader, epoch: int) -> tuple:
        """Validate the model"""
        
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        beta = self._get_beta_schedule(epoch)
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                
                recon_batch, mu, logvar = self.model(data)
                loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, beta)
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        # Calculate averages
        avg_loss = total_loss / len(val_loader)
        avg_recon_loss = total_recon_loss / len(val_loader)
        avg_kl_loss = total_kl_loss / len(val_loader)
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def _get_beta_schedule(self, epoch: int) -> float:
        """Get beta value for current epoch (beta-VAE scheduling)"""
        schedule = self.training_config.get('beta_schedule', {})
        
        if schedule.get('type') == 'linear':
            start_beta = schedule.get('start_beta', 0.0)
            end_beta = schedule.get('end_beta', 1.0)
            warmup_epochs = schedule.get('warmup_epochs', 10)
            
            if epoch < warmup_epochs:
                return start_beta + (end_beta - start_beta) * (epoch / warmup_epochs)
            else:
                return end_beta
        
        elif schedule.get('type') == 'cyclical':
            # Cyclical beta schedule
            cycle_length = schedule.get('cycle_length', 10)
            min_beta = schedule.get('min_beta', 0.0)
            max_beta = schedule.get('max_beta', 1.0)
            
            cycle_position = (epoch % cycle_length) / cycle_length
            return min_beta + (max_beta - min_beta) * (1 - np.cos(np.pi * cycle_position)) / 2
        
        else:
            # Constant beta
            return schedule.get('beta', 1.0)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint (excluding DINOv2 to reduce size)"""
        
        # Get model state dict without DINOv2 parameters
        model_state_dict = {}
        for name, param in self.model.state_dict().items():
            # Skip DINOv2 parameters since they're frozen and can be reloaded
            if not name.startswith('dinov2.'):
                model_state_dict[name] = param
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_recon_losses': self.train_recon_losses,
            'train_kl_losses': self.train_kl_losses,
            'val_recon_losses': self.val_recon_losses,
            'val_kl_losses': self.val_kl_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': {
                'model_config': self.model_config,
                'training_config': self.training_config,
                'data_config': self.data_config
            }
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.experiment_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.experiment_dir / 'best_checkpoint.pth')
            logger.info(f"New best model saved at epoch {epoch+1}")
        
        # Log checkpoint size
        checkpoint_size = (self.experiment_dir / 'latest_checkpoint.pth').stat().st_size / (1024 * 1024)
        logger.info(f"Checkpoint saved (size: {checkpoint_size:.2f} MB)")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint (DINOv2 will be reloaded from pretrained weights)"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state dict, but skip missing DINOv2 parameters
        model_state_dict = checkpoint['model_state_dict']
        
        # Get current model state dict
        current_state_dict = self.model.state_dict()
        
        # Update only the parameters that exist in the checkpoint
        # DINOv2 parameters will remain as pretrained weights
        for name, param in model_state_dict.items():
            if name in current_state_dict:
                current_state_dict[name] = param
        
        # Load the updated state dict
        self.model.load_state_dict(current_state_dict)
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_recon_losses = checkpoint['train_recon_losses']
        self.train_kl_losses = checkpoint['train_kl_losses']
        self.val_recon_losses = checkpoint['val_recon_losses']
        self.val_kl_losses = checkpoint['val_kl_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info("DINOv2 parameters kept as pretrained weights")
        return checkpoint['epoch']
    
    def plot_training_curves(self):
        """Plot training curves"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.train_losses, label='Train')
        axes[0, 0].plot(self.val_losses, label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        axes[0, 1].plot(self.train_recon_losses, label='Train')
        axes[0, 1].plot(self.val_recon_losses, label='Validation')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # KL divergence
        axes[1, 0].plot(self.train_kl_losses, label='Train')
        axes[1, 0].plot(self.val_kl_losses, label='Validation')
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        if hasattr(self.scheduler, 'get_last_lr'):
            lrs = [group['lr'] for group in self.optimizer.param_groups]
            axes[1, 1].plot(lrs[0], label='Encoder LR')
            axes[1, 1].plot(lrs[1], label='Decoder LR')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Complete training loop"""
        
        logger.info("Starting training...")
        
        for epoch in range(self.training_config['epochs']):
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_recon, val_kl = self.validate(val_loader, epoch)
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Track losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_recon_losses.append(train_recon)
            self.train_kl_losses.append(train_kl)
            self.val_recon_losses.append(val_recon)
            self.val_kl_losses.append(val_kl)
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
            
            # Save checkpoint
            if (epoch + 1) % self.training_config.get('save_freq', 10) == 0:
                self.save_checkpoint(epoch, is_best)
            
            # Log epoch results
            logger.info(f'Epoch {epoch+1}/{self.training_config["epochs"]}:')
            logger.info(f'  Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})')
            logger.info(f'  Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})')
            logger.info(f'  Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})')
            
            # Generate samples periodically
            if (epoch + 1) % self.training_config.get('sample_freq', 10) == 0:
                self.generate_samples(epoch, num_samples=8)
                self.test_reconstruction(val_loader, epoch, num_samples=8)
        
        # Final save
        self.save_checkpoint(self.training_config['epochs'] - 1, is_best=False)
        
        # Plot training curves
        self.plot_training_curves()
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")
    
    def generate_samples(self, epoch: int, num_samples: int = 8):
        """Generate and save sample images"""
        
        self.model.eval()
        
        with torch.no_grad():
            samples = self.model.sample(num_samples, self.device)
            
            # Convert from [-1, 1] to [0, 1]
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            # Create grid
            cols = 4
            rows = math.ceil(num_samples / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
            
            # Handle case where there's only one row
            if rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            for i in range(num_samples):
                img = samples[i].cpu().permute(1, 2, 0).numpy()
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f'Sample {i+1}')
            
            # Hide unused subplots
            for i in range(num_samples, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'Generated Samples - Epoch {epoch+1}')
            plt.tight_layout()
            plt.savefig(self.experiment_dir / f'samples_epoch_{epoch+1}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def test_reconstruction(self, data_loader: DataLoader, epoch: int, num_samples: int = 8):
        """Test reconstruction quality"""
        
        self.model.eval()
        
        with torch.no_grad():
            # Get a batch
            data_batch, _ = next(iter(data_loader))
            data_batch = data_batch[:num_samples].to(self.device)
            
            # Reconstruct
            recon_batch, _, _ = self.model(data_batch)
            
            # Convert to display format
            originals = (data_batch + 1) / 2
            reconstructions = (recon_batch + 1) / 2
            
            originals = torch.clamp(originals, 0, 1)
            reconstructions = torch.clamp(reconstructions, 0, 1)
            
            # Create comparison plot
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            
            for i in range(num_samples):
                # Original
                orig_img = originals[i].cpu().permute(1, 2, 0).numpy()
                axes[i//2, (i%2)*2].imshow(orig_img)
                axes[i//2, (i%2)*2].axis('off')
                axes[i//2, (i%2)*2].set_title(f'Original {i+1}')
                
                # Reconstruction
                recon_img = reconstructions[i].cpu().permute(1, 2, 0).numpy()
                axes[i//2, (i%2)*2+1].imshow(recon_img)
                axes[i//2, (i%2)*2+1].axis('off')
                axes[i//2, (i%2)*2+1].set_title(f'Reconstruction {i+1}')
            
            plt.suptitle(f'Reconstruction Quality - Epoch {epoch+1}')
            plt.tight_layout()
            plt.savefig(self.experiment_dir / f'reconstruction_epoch_{epoch+1}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Main training function"""
    
    # Configuration
    model_config = {
        'latent_dim': 128,
        'image_size': 128,
        'projection_multiplier': 1.5,
        'min_hidden_dim': 512
    }
    
    training_config = {
        'epochs': 100,
        'encoder_lr': 1e-4,
        'decoder_lr': 2e-4,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'save_freq': 10,
        'sample_freq': 5,
        'beta_schedule': {
            'type': 'linear',
            'start_beta': 0.0,
            'end_beta': 1.0,
            'warmup_epochs': 20
        },
        'scheduler': {
            'type': 'cosine',
            'min_lr': 1e-6
        }
    }
    
    data_config = {
        'batch_size': 16,
        'image_size': 128,
        'train_split': 0.8,
        'augment': True,
        'num_workers': 4,
        'max_images': 1000
    }
    
    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(**data_config)
        
        # Visualize some training data
        logger.info("Visualizing training data...")
        visualize_batch(train_loader, num_images=8)
        
        # Initialize trainer
        trainer = VAETrainer(model_config, training_config, data_config)
        
        # Start training
        trainer.train(train_loader, val_loader)
        
        # Final testing
        logger.info("Generating final samples...")
        trainer.generate_samples(epoch=training_config['epochs']-1, num_samples=16)
        trainer.test_reconstruction(val_loader, epoch=training_config['epochs']-1, num_samples=8)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
