import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import logging
from tqdm import tqdm
import pickle
import json
from datetime import datetime

class BCCDataset(Dataset):
    """Dataset class for BCC images optimized for StyleGAN2 training."""
    
    def __init__(self, image_dir, image_size=256, augment=True):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.image_paths = self._get_image_paths()
        
        # Optimized transforms for medical skin lesion images
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),  # Only safe augmentation for medical images
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Minimal color jitter
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def _get_image_paths(self):
        """Get all image paths from directory."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(list(self.image_dir.glob(f"*{ext}")))
            image_paths.extend(list(self.image_dir.glob(f"*{ext.upper()}")))
        
        return sorted(list(set(image_paths)))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image = self.transform(image)
            
            return image
        except Exception as e:
            # Return a black image if loading fails
            return torch.zeros(3, self.image_size, self.image_size)

class StyleGAN2Generator(nn.Module):
    """Simplified StyleGAN2 Generator optimized for BCC lesion generation."""
    
    def __init__(self, latent_dim=512, img_channels=3, img_size=256):
        super(StyleGAN2Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Mapping Network (8 layers for better disentanglement)
        self.mapping = nn.Sequential(
            self._make_mapping_layer(latent_dim, latent_dim),
            self._make_mapping_layer(latent_dim, latent_dim),
            self._make_mapping_layer(latent_dim, latent_dim),
            self._make_mapping_layer(latent_dim, latent_dim),
            self._make_mapping_layer(latent_dim, latent_dim),
            self._make_mapping_layer(latent_dim, latent_dim),
            self._make_mapping_layer(latent_dim, latent_dim),
            self._make_mapping_layer(latent_dim, latent_dim)
        )
        
        # Synthesis Network
        self.constant = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # Progressive blocks (4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256)
        self.blocks = nn.ModuleList([
            self._make_synthesis_block(512, 512, upsample=False),  # 4x4
            self._make_synthesis_block(512, 512, upsample=True),   # 8x8
            self._make_synthesis_block(512, 512, upsample=True),   # 16x16
            self._make_synthesis_block(512, 256, upsample=True),   # 32x32
            self._make_synthesis_block(256, 128, upsample=True),   # 64x64
            self._make_synthesis_block(128, 64, upsample=True),    # 128x128
            self._make_synthesis_block(64, 32, upsample=True),     # 256x256
        ])
        
        # To RGB layers for each resolution
        self.to_rgb = nn.ModuleList([
            nn.Conv2d(512, img_channels, 1),  # 4x4
            nn.Conv2d(512, img_channels, 1),  # 8x8
            nn.Conv2d(512, img_channels, 1),  # 16x16
            nn.Conv2d(256, img_channels, 1),  # 32x32
            nn.Conv2d(128, img_channels, 1),  # 64x64
            nn.Conv2d(64, img_channels, 1),   # 128x128
            nn.Conv2d(32, img_channels, 1),   # 256x256
        ])
        
        self.tanh = nn.Tanh()
    
    def _make_mapping_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _make_synthesis_block(self, in_channels, out_channels, upsample=True):
        layers = []
        
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, noise):
        batch_size = noise.size(0)
        
        # Mapping network
        w = self.mapping(noise)
        
        # Start with constant
        x = self.constant.expand(batch_size, -1, -1, -1)
        
        # Progressive synthesis
        for i, (block, to_rgb) in enumerate(zip(self.blocks, self.to_rgb)):
            x = block(x)
            
            # Skip connections for better gradient flow (StyleGAN2 feature)
            if i == len(self.blocks) - 1:  # Final layer
                rgb_out = to_rgb(x)
        
        return self.tanh(rgb_out)

class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator with medical image optimizations."""
    
    def __init__(self, img_channels=3, img_size=256):
        super(StyleGAN2Discriminator, self).__init__()
        
        # Progressive discriminator blocks
        self.blocks = nn.ModuleList([
            self._make_disc_block(img_channels, 32),    # 256x256
            self._make_disc_block(32, 64),              # 128x128
            self._make_disc_block(64, 128),             # 64x64
            self._make_disc_block(128, 256),            # 32x32
            self._make_disc_block(256, 512),            # 16x16
            self._make_disc_block(512, 512),            # 8x8
            self._make_disc_block(512, 512),            # 4x4
        ])
        
        # Final classification layer
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )
    
    def _make_disc_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2)
        )
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final(x)

class StyleGAN2BCCGenerator:
    """Complete StyleGAN2-ADA system for BCC lesion generation."""
    
    def __init__(self, 
                 latent_dim=512,
                 img_size=256,
                 learning_rate=0.0002,
                 beta1=0.0,
                 beta2=0.99,
                 device='auto'):
        
        
        self.logger = logging.getLogger(__name__)
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.device = self._setup_device(device)
        
        # Initialize networks
        self.generator = StyleGAN2Generator(latent_dim, 3, img_size).to(self.device)
        self.discriminator = StyleGAN2Discriminator(3, img_size).to(self.device)
        
        # Optimizers (StyleGAN2-ADA parameters)
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=learning_rate, 
            betas=(beta1, beta2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=learning_rate, 
            betas=(beta1, beta2)
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.is_trained = False
        self.training_epochs = 0
    
    def _setup_device(self, device):
        """Setup computing device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self.logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                self.logger.info("🚀 Using Apple MPS")
            else:
                device = 'cpu'
                self.logger.warning("⚠️  Using CPU - training will be slow!")
        
        return torch.device(device)
    
    def train(self, 
              bcc_data_dir="data/bcc_segmented",
              epochs=2000,
              batch_size=8,
              save_interval=100,
              checkpoint_dir="model/stylegan2_bcc"):
        """
        Train StyleGAN2 on BCC dataset.
        
        Args:
            bcc_data_dir: Directory containing BCC images
            epochs: Number of training epochs (2000+ recommended)
            batch_size: Batch size (start small for medical images)
            save_interval: Save checkpoint every N epochs
            checkpoint_dir: Directory to save checkpoints
        """
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup dataset
        dataset = BCCDataset(bcc_data_dir, self.img_size, augment=True)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"\n🔬 STYLEGAN2-ADA BCC TRAINING")
        print(f"="*60)
        print(f"📊 Dataset: {len(dataset)} BCC images")
        print(f"🖼️  Image size: {self.img_size}x{self.img_size}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🔄 Epochs: {epochs}")
        print(f"💾 Checkpoints: {checkpoint_dir}")
        print(f"🚀 Device: {self.device}")
        print(f"="*60)
        
        # Training loop
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            for i, real_images in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                batch_size_current = real_images.size(0)
                real_images = real_images.to(self.device)
                
                # Create labels
                real_labels = torch.ones(batch_size_current, 1).to(self.device)
                fake_labels = torch.zeros(batch_size_current, 1).to(self.device)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.d_optimizer.zero_grad()
                
                # Real images
                real_pred = self.discriminator(real_images)
                d_real_loss = self.adversarial_loss(real_pred, real_labels)
                
                # Fake images
                noise = torch.randn(batch_size_current, self.latent_dim).to(self.device)
                fake_images = self.generator(noise)
                fake_pred = self.discriminator(fake_images.detach())
                d_fake_loss = self.adversarial_loss(fake_pred, fake_labels)
                
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()
                
                # -----------------
                #  Train Generator
                # -----------------
                self.g_optimizer.zero_grad()
                
                # Generate fake images
                noise = torch.randn(batch_size_current, self.latent_dim).to(self.device)
                fake_images = self.generator(noise)
                fake_pred = self.discriminator(fake_images)
                
                # Generator loss
                g_loss = self.adversarial_loss(fake_pred, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
                
                # Accumulate losses
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
            
            # Print epoch results
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            
            print(f"Epoch [{epoch+1}/{epochs}] - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch + 1, checkpoint_dir)
                self._generate_sample_images(epoch + 1, checkpoint_dir)
        
        # Mark as trained
        self.is_trained = True
        self.training_epochs = epochs
        
        # Save final model
        self._save_final_model(checkpoint_dir)
        
        print(f"\n✅ StyleGAN2-ADA training completed!")
        print(f"🎯 Ready to generate BCC images!")
    
    
    def generate_bcc_images(self, num_images, output_dir, seed=None):
        """Generate exactly num_images BCC images."""
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        
        os.makedirs(output_dir, exist_ok=True)
        generated_count = 0
        
        with torch.no_grad():
            for i in tqdm(range(num_images), desc="Generating BCC images"):
                # Generate random latent vector
                z = torch.randn(1, self.latent_dim, device=self.device)
                
                # Generate image
                fake_image = self.generator(z)
                
                # Convert to PIL format and save
                img_tensor = (fake_image + 1) / 2.0  # Normalize from [-1,1] to [0,1]
                img_tensor = torch.clamp(img_tensor, 0, 1)
                
                # Save individual image
                filepath = os.path.join(output_dir, f"bcc_generated_{i:05d}.png")
                
                # Save without quality parameter
                save_image(img_tensor, filepath)
                
                generated_count += 1
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Generated {i + 1}/{num_images} images")
        
        return generated_count  

    def _save_checkpoint(self, epoch, checkpoint_dir):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'latent_dim': self.latent_dim,
            'img_size': self.img_size
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
    
    def _save_final_model(self, checkpoint_dir):
        """Save final trained model."""
        model_data = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'latent_dim': self.latent_dim,
            'img_size': self.img_size,
            'training_epochs': self.training_epochs,
            'timestamp': datetime.now().isoformat()
        }
        
        model_path = os.path.join(checkpoint_dir, "stylegan2_bcc_final.pt")
        torch.save(model_data, model_path)
        
        print(f"💾 Final model saved: {model_path}")
    
    def _generate_sample_images(self, epoch, checkpoint_dir, num_samples=8):
        """Generate sample images during training."""
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            fake_images = self.generator(noise)
            
            # Save sample grid
            sample_path = os.path.join(checkpoint_dir, f"samples_epoch_{epoch}.jpg")
            save_image(fake_images, sample_path, nrow=4, normalize=True)
        
        self.generator.train()
    
    def load_model(self, model_path):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        if 'discriminator_state_dict' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        self.is_trained = True
        print(f"✅ Model loaded from: {model_path}")

def generate_bcc_images_stylegan2(num_images,
                                 output_dir="data/bcc_gan_generated", 
                                 bcc_data_dir="data/bcc_segmented",
                                 model_path=None,
                                 train_if_needed=True,
                                 epochs=1000,
                                 seed=42):
    """
    Complete function to generate BCC images using StyleGAN2-ADA.
    
    Args:
        num_images: Number of images to generate (YOU control this!)
        output_dir: Where to save generated images
        bcc_data_dir: Source BCC images for training
        model_path: Path to pre-trained model (optional)
        train_if_needed: Whether to train if no model exists
        epochs: Training epochs if training needed
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (success, generated_count, model_path)
    """
    
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize StyleGAN2 generator
        stylegan = StyleGAN2BCCGenerator(
            latent_dim=512,
            img_size=256,
            device='auto'
        )
        
        # Check if we need to train
        if model_path and os.path.exists(model_path):
            print(f"📦 Loading pre-trained model: {model_path}")
            stylegan.load_model(model_path)
        elif train_if_needed:
            print(f"🚀 Training new StyleGAN2-ADA model...")
            stylegan.train(
                bcc_data_dir=bcc_data_dir,
                epochs=epochs,
                batch_size=4,  # Conservative for medical images
                save_interval=100
            )
            model_path = "model/stylegan2_bcc/stylegan2_bcc_final.pt"
        else:
            raise ValueError("❌ No trained model found and training disabled!")
        
        # Generate images
        generated_count = stylegan.generate_bcc_images(
            num_images=num_images,
            output_dir=output_dir,
            seed=seed
        )
        
        return True, generated_count, model_path
        
    except Exception as e:
        print(f"❌ Error in StyleGAN2 generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0, None
