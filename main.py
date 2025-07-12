import os
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class CelebADataset(Dataset):
    """Simple CelebA dataset loader for Kaggle dataset structure"""
    
    def __init__(self, root_dir: str, transform=None, max_images=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Kaggle CelebA structure: img_align_celeba/ folder with .jpg files
        img_dir = self.root_dir
        new_img_dir = img_dir / "img_align_celeba"
        if new_img_dir.exists():
            img_dir = new_img_dir
            new_img_dir = img_dir / "img_align_celeba"
            if new_img_dir.exists():
                img_dir = new_img_dir

        self.image_paths = list(img_dir.glob("*.jpg"))
        
        if max_images:
            self.image_paths = self.image_paths[:max_images]
        
        logger.info(f"Found {len(self.image_paths)} CelebA images in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0

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
    
    # Base transforms with ImageNet normalization
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        train_transforms = base_transforms
    
    # Validation transforms (no augmentation)
    val_transforms = base_transforms
    
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


def create_data_loaders(dataset_path=None, batch_size=32, 
                       image_size=224, train_split=0.8, augment=True, 
                       num_workers=4, max_images=None):
    """Create training and validation data loaders"""
    
    # Get dataset path
    if dataset_path is None:
        dataset_path = os.getenv('CELEBA_DATASET_PATH', 'celeba/img_align_celeba/img_align_celeba')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(image_size, augment)
    
    # Create full dataset and split manually
    full_dataset = CelebADataset(dataset_path, transform=None, max_images=max_images)
    
    # Simple split
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    
    # Create datasets with transforms
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Wrap with transform
    train_dataset = TransformDataset(train_dataset, train_transform)
    val_dataset = TransformDataset(val_dataset, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=torch.cuda.is_available())

    logger.info(f"Created CelebA data loaders: {len(train_dataset)} train, {len(val_dataset)} val")

    return train_loader, val_loader

class TransformDataset(Dataset):
    """Simple wrapper to apply transforms to a dataset"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if hasattr(self.dataset, 'dataset'):
            # Handle Subset case
            image_path = self.dataset.dataset.image_paths[self.dataset.indices[idx]]
            image = Image.open(image_path).convert('RGB')
        else:
            image, _ = self.dataset[idx]
            if isinstance(image, torch.Tensor):
                # Convert tensor back to PIL for transforms
                image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0

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
    
    # ImageNet normalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i, img in enumerate(images):
        # Denormalize from ImageNet normalization
        img_display = img * std + mean
        img_display = torch.clamp(img_display, 0, 1)
        
        # Convert to numpy and transpose for matplotlib
        img_np = img_display.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].axis('off')
        axes[i].set_title(f'Image {i+1}')
    
    plt.tight_layout()
    plt.show()

class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder for VAE
    """
    
    def __init__(self, 
                 latent_dim: int = 512,
                 resnet_variant: str = 'resnet18',
                 pretrained: bool = True,
                 freeze_early_layers: bool = True):
        """
        Initialize ResNet encoder
        
        Args:
            latent_dim: Dimension of latent space
            resnet_variant: Which ResNet variant to use
            pretrained: Whether to use pretrained weights
            freeze_early_layers: Whether to freeze early convolutional layers
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Load ResNet backbone
        if resnet_variant == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif resnet_variant == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif resnet_variant == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {resnet_variant}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze early layers if requested
        if freeze_early_layers:
            self._freeze_early_layers()
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
        
        # Projection layers
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Latent space parameters
        self.mu_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)
        
        logger.info(f"ResNet encoder initialized with {resnet_variant}")
        logger.info(f"Feature dimension: {feature_dim}")
        logger.info(f"Latent dimension: {latent_dim}")
    
    def _freeze_early_layers(self):
        """Freeze early convolutional layers"""
        # Freeze first few layers (conv1, bn1, relu, maxpool, layer1)
        layers_to_freeze = ['0', '1', '2', '3', '4']  # indices in backbone
        
        for i, (name, module) in enumerate(self.backbone.named_children()):
            if str(i) in layers_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
                logger.info(f"Froze layer {name} in ResNet backbone")
        
        logger.info("Frozen early ResNet layers")
    
    def forward(self, x):
        """
        Forward pass through encoder
        
        Args:
            x: Input images [batch_size, 3, H, W]
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Extract features using ResNet backbone
        features = self.backbone(x)  # [batch_size, feature_dim, 1, 1]
        features = features.view(features.size(0), -1)  # [batch_size, feature_dim]
        
        # Project to latent space
        projected = self.feature_projection(features)
        
        # Get mu and logvar
        mu = self.mu_layer(projected)
        logvar = self.logvar_layer(projected)
        
        return mu, logvar

class VAE(nn.Module):
    def __init__(self, 
                 latent_dim: int = 512, 
                 image_size: int = 224,
                 resnet_variant: str = 'resnet18',
                 pretrained: bool = True,
                 freeze_early_layers: bool = True):
        """
        ResNet-based VAE
        
        Args:
            latent_dim: Dimension of latent space
            image_size: Size of input/output images
            resnet_variant: Which ResNet variant to use
            pretrained: Whether to use pretrained ResNet weights
            freeze_early_layers: Whether to freeze early ResNet layers
        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size

        # ResNet encoder
        self.encoder = ResNetEncoder(
            latent_dim=latent_dim,
            resnet_variant=resnet_variant,
            pretrained=pretrained,
            freeze_early_layers=freeze_early_layers
        )

        # Decoder
        self.decoder = self._build_decoder()
    
    def _build_decoder(self):
        """Build decoder network"""
        
        # Calculate initial spatial dimensions based on image size
        if self.image_size <= 64:
            initial_spatial = 4
            initial_channels = min(512, self.latent_dim * 2)
        elif self.image_size <= 128:
            initial_spatial = 8
            initial_channels = min(512, self.latent_dim)
        elif self.image_size <= 256:
            initial_spatial = 8
            initial_channels = 512
        else:
            initial_spatial = 16
            initial_channels = 512
        
        # Initial linear projection
        initial_size = initial_channels * initial_spatial * initial_spatial
        
        decoder_layers = [
            nn.Linear(self.latent_dim, initial_size),
            nn.ReLU(),
            Reshape(-1, initial_channels, initial_spatial, initial_spatial)
        ]
        
        # Convolutional layers for upsampling
        current_spatial = initial_spatial
        current_channels = initial_channels
        
        # Calculate upsampling path
        upsampling_steps = []
        temp_spatial = current_spatial
        while temp_spatial < self.image_size:
            temp_spatial *= 2
            upsampling_steps.append(min(temp_spatial, self.image_size))
        
        # Build upsampling layers
        for target_spatial in upsampling_steps:
            next_channels = max(32, current_channels // 2)
            
            if target_spatial == current_spatial * 2:
                # Standard 2x upsampling
                decoder_layers.extend([
                    nn.ConvTranspose2d(current_channels, next_channels, 
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU()
                ])
            else:
                # Custom upsampling with interpolation
                decoder_layers.extend([
                    nn.Conv2d(current_channels, next_channels, 
                             kernel_size=3, padding=1),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(),
                    nn.Upsample(size=(target_spatial, target_spatial), 
                               mode='bilinear', align_corners=False)
                ])
            
            current_channels = next_channels
            current_spatial = target_spatial
        
        # Final layer to RGB
        decoder_layers.extend([
            nn.Conv2d(current_channels, 3, kernel_size=3, padding=1),
            nn.Tanh()
        ])
        
        return nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode images to latent space"""
        return self.encoder(x)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vectors to images"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

    def sample(self, num_samples, device):
        """Generate new samples from the latent space"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def get_architecture_info(self):
        """Return information about the architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        return {
            'latent_dim': self.latent_dim,
            'image_size': self.image_size,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'encoder_params': encoder_params,
            'decoder_params': decoder_params
        }

class Reshape(nn.Module):
    """Reshape layer"""
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

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

class VAETrainer:
    """
    Complete training and testing pipeline for VAE
    """
    
    def __init__(self, 
                 model_config: dict,
                 training_config: dict,
                 data_config: dict,
                 save_dir: str = "experiments",
                 resume_from: Optional[str] = None,
                 save_path: Optional[str] = None):
        """
        Initialize the trainer
        
        Args:
            model_config: Model configuration parameters
            training_config: Training hyperparameters
            data_config: Data loading configuration
            save_dir: Directory to save results
            resume_from: Path to checkpoint to resume from, or 'latest' to auto-resume
            save_path: Specific path to save the experiment, overrides save_dir and timestamp
        """
        
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        # Early stopping parameters
        self.early_stopping_patience = training_config.get('early_stopping_patience', None)
        self.early_stopping_delta = training_config.get('early_stopping_delta', 0.001)
        self.patience_counter = 0
        self.should_stop_early = False
        
        # Handle resuming from checkpoint
        self.resume_from = resume_from
        self.start_epoch = 0
        
        if save_path:
            self.experiment_dir = Path(save_path)
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
        elif resume_from == 'latest':
            # Find the latest experiment directory
            self.experiment_dir = self._find_latest_experiment(Path(save_dir))
        elif resume_from and Path(resume_from).exists():
            # Resume from specific checkpoint
            self.experiment_dir = Path(resume_from).parent
        else:
            # Create new experiment directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = Path(save_dir) / f"vae_resnet_{timestamp}"
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configs
        if not resume_from:
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
        
        # Load checkpoint if resuming
        if resume_from:
            self._resume_training()
        
        logger.info(f"Experiment directory: {self.experiment_dir}")
        
        # Early stopping info
        if self.early_stopping_patience:
            logger.info(f"Early stopping enabled: patience={self.early_stopping_patience}, delta={self.early_stopping_delta}")
    
        # Log parameter information
        arch_info = self.model.get_architecture_info()
        logger.info(f"Model architecture:")
        for key, value in arch_info.items():
            logger.info(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    
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
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        
        # Separate parameters for different learning rates
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())
        
        # Create parameter groups
        param_groups = [
            {'params': encoder_params, 'lr': self.training_config['encoder_lr']},
            {'params': decoder_params, 'lr': self.training_config['decoder_lr']}
        ]
        
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
        """Save model checkpoint, including epoch in the filename."""
        
        # Determine if the full model (including ResNet) should be saved
        save_full_model = not self.model_config.get('freeze_early_layers', True)
        
        if save_full_model:
            model_state_dict = self.model.state_dict()
            filename_marker = "_with_resnet"
            logger.info("Saving full model checkpoint (including ResNet)")
        else:
            model_state_dict = {k: v for k, v in self.model.state_dict().items() if not k.startswith('encoder.backbone.')}
            filename_marker = ""
            logger.info("Saving partial model checkpoint (excluding ResNet backbone)")

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
            'patience_counter': self.patience_counter,
            'contains_resnet': save_full_model,
            'config': {
                'model_config': self.model_config,
                'training_config': self.training_config,
                'data_config': self.data_config
            }
        }
        
        # Save epoch-specific checkpoint
        epoch_checkpoint_path = self.experiment_dir / f'checkpoint_epoch_{epoch+1}{filename_marker}.pth'
        torch.save(checkpoint, epoch_checkpoint_path)
        
        # Save latest checkpoint (symlink or copy)
        latest_checkpoint_path = self.experiment_dir / 'latest_checkpoint.pth'
        if latest_checkpoint_path.exists():
            latest_checkpoint_path.unlink()
        os.symlink(epoch_checkpoint_path.name, latest_checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = self.experiment_dir / f'best_checkpoint{filename_marker}.pth'
            # Remove the old best checkpoint if the name changes
            old_best_checkpoint_path = self.experiment_dir / f'best_checkpoint{"" if save_full_model else "_with_resnet"}.pth'
            if old_best_checkpoint_path.exists() and old_best_checkpoint_path != best_checkpoint_path:
                old_best_checkpoint_path.unlink()
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"New best model saved at epoch {epoch+1}")
        
        checkpoint_size = epoch_checkpoint_path.stat().st_size / (1024 * 1024)
        logger.info(f"Checkpoint saved to {epoch_checkpoint_path} (size: {checkpoint_size:.2f} MB)")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Check if the checkpoint contains the full model
        contains_resnet = checkpoint.get('contains_resnet', False)
        model_state_dict = checkpoint['model_state_dict']
        
        if contains_resnet:
            logger.info("Loading full model state from checkpoint (including ResNet)")
            self.model.load_state_dict(model_state_dict)
        else:
            logger.info("Loading partial model state from checkpoint (excluding ResNet)")
            current_state_dict = self.model.state_dict()
            
            # Update only the parameters that exist in the checkpoint
            for name, param in model_state_dict.items():
                if name in current_state_dict:
                    current_state_dict[name] = param
            
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
        
        # Restore early stopping state if available
        if 'patience_counter' in checkpoint:
            self.patience_counter = checkpoint['patience_counter']
        else:
            # Reset patience counter when loading old checkpoints
            self.patience_counter = 0
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from epoch {checkpoint['epoch'] + 1}")
        logger.info(f"Best validation loss so far: {self.best_val_loss:.4f}")
        if self.early_stopping_patience:
            logger.info(f"Early stopping patience: {self.patience_counter}/{self.early_stopping_patience}")
        
        return checkpoint['epoch']
    
    def _find_latest_experiment(self, save_dir: Path) -> Path:
        """Find the latest experiment directory"""
        experiment_dirs = list(save_dir.glob("vae_resnet_*"))
        
        if not experiment_dirs:
            raise FileNotFoundError("No previous experiments found for resuming")
        
        # Sort by timestamp in directory name
        experiment_dirs.sort(key=lambda x: x.name.split('_')[-1])
        latest_dir = experiment_dirs[-1]
        
        logger.info(f"Found latest experiment: {latest_dir}")
        return latest_dir
    
    def _resume_training(self):
        """Resume training from checkpoint"""
        # Try to find checkpoint in experiment directory
        checkpoint_path = None
        
        if self.resume_from == 'latest':
            # Look for best checkpoint first, then latest
            best_checkpoint_regular = self.experiment_dir / 'best_checkpoint.pth'
            best_checkpoint_with_resnet = self.experiment_dir / 'best_checkpoint_with_resnet.pth'
            latest_checkpoint = self.experiment_dir / 'latest_checkpoint.pth'

            if best_checkpoint_with_resnet.exists():
                checkpoint_path = best_checkpoint_with_resnet
                logger.info("Resuming from best checkpoint (with ResNet)")
            elif best_checkpoint_regular.exists():
                checkpoint_path = best_checkpoint_regular
                logger.info("Resuming from best checkpoint")
            elif latest_checkpoint.exists():
                checkpoint_path = latest_checkpoint
                logger.info("Resuming from latest checkpoint")
        else:
            if self.resume_from:
                checkpoint_path = Path(self.resume_from)
        
        if checkpoint_path and checkpoint_path.exists():
            self.start_epoch = self.load_checkpoint(str(checkpoint_path)) + 1
            logger.info(f"Resuming training from epoch {self.start_epoch}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if training should stop early"""
        if self.early_stopping_patience is None:
            return False
        
        # Check if validation loss improved significantly
        if self.best_val_loss - val_loss >= self.early_stopping_delta:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.early_stopping_patience:
            logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
            return True
            
        return False
    
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
        
        # Handle keyboard interruption gracefully
        try:
            for epoch in range(self.start_epoch, self.training_config['epochs']):
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
                
                # Check early stopping
                if self._check_early_stopping(val_loss):
                    self.should_stop_early = True
                
                # Check for best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                
                # Save checkpoint
                if (epoch + 1) % self.training_config.get('save_freq', 10) == 0 or self.should_stop_early:
                    self.save_checkpoint(epoch, is_best)
                
                # Log epoch results
                logger.info(f'Epoch {epoch+1}/{self.training_config["epochs"]}:')
                logger.info(f'  Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})')
                logger.info(f'  Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})')
                logger.info(f'  Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})')
                
                if self.early_stopping_patience:
                    logger.info(f'  Early stopping patience: {self.patience_counter}/{self.early_stopping_patience}')
                
                # Generate samples periodically
                if (epoch + 1) % self.training_config.get('sample_freq', 10) == 0:
                    self.generate_samples(epoch, num_samples=8)
                    self.test_reconstruction(val_loader, epoch, num_samples=8)
                
                # Early stopping check
                if self.should_stop_early:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
            logger.info("Saving current checkpoint...")
            self.save_checkpoint(epoch, is_best=False)
            logger.info("Checkpoint saved successfully")
            raise
        
        # Final save
        if not self.should_stop_early:
            self.save_checkpoint(self.training_config['epochs'] - 1, is_best=False)
        
        # Plot training curves
        self.plot_training_curves()
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")
        
        if self.should_stop_early:
            logger.info(f"Training stopped early due to no improvement for {self.early_stopping_patience} epochs")
    
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
        'image_size': 224,
        'resnet_variant': 'resnet50',
        'freeze_early_layers': False,
    }
    
    training_config = {
        'epochs': 300,
        'encoder_lr': 1e-4,
        'decoder_lr': 2e-4,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'save_freq': 5,
        'sample_freq': 5,
        'early_stopping_patience': 15,
        'early_stopping_delta': 0.001,
        'beta_schedule': {
            'type': 'linear',
            'start_beta': 0.0,
            'end_beta': 1.0,
            'warmup_epochs': 20
        },
        'scheduler': {
            'type': 'cosine',
            'min_lr': 1e-6
        },
        'resume_from': None,
        'save_path': None,
    }
    
    data_config = {
        'batch_size': 16,
        'image_size': 224,
        'train_split': 0.8,
        'augment': True,
        'num_workers': 4,
        'max_images': None
    }
    
    resume_from = training_config.get('resume_from')
    save_path = training_config.get('save_path')

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == '--resume-latest':
                resume_from = 'latest'
            elif arg.startswith('--resume='):
                resume_from = arg.split('=', 1)[1]
            elif arg.startswith('--save-path='):
                save_path = arg.split('=', 1)[1]

    try:
        # Create data loaders
        logger.info(f"Creating data loaders for CelebA dataset...")
        train_loader, val_loader = create_data_loaders(**data_config)
        
        # Initialize trainer with optional resume
        trainer = VAETrainer(
            model_config, 
            training_config, 
            data_config,
            resume_from=resume_from,
            save_path=save_path
        )
        
        # Start training
        trainer.train(train_loader, val_loader)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        logger.info("You can resume training by running:")
        logger.info("python main.py --resume-latest")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
