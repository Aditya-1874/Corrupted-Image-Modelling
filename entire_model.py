import os
from pathlib import Path
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
import itertools
import math
import random
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from PIL import Image
from dall_e import load_model
from torchvision.utils import save_image

from torchvision import models

"""Generator"""

# DVAETokenizer Implementation
class DVAETokenizer:
    def __init__(self, device):
        self.encoder = load_model("https://cdn.openai.com/dall-e/encoder.pkl", device)
        self.decoder = load_model("https://cdn.openai.com/dall-e/decoder.pkl", device)
        self.vocab_size = 8192
        self.device = device

    def encode(self, image):
        # Ensure input image has 3 channels
        if image.size(1) != 3:
            image = image[:, :3, :, :]

        with torch.no_grad():
            tokens = self.encoder(image)
            z = torch.argmax(tokens, axis=1)
            return F.one_hot(z, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()

    def decode(self, tokens):
        with torch.no_grad():
            decoded = self.decoder(tokens)
            # Ensure output has exactly 3 channels
            if decoded.size(1) != 3:
                decoded = decoded[:, :3, :, :]
            return decoded

class BEITDatasetCifar(Dataset):
    def __init__(self, cifar_data, device):
        self.cifar_data = cifar_data
        self.device = device
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Lambda(self.normalize_transform)
        ])

    @staticmethod
    def normalize_transform(x):
        return x * 2 - 1

    def __len__(self):
        return len(self.cifar_data)

    def __getitem__(self, idx):
        image, _ = self.cifar_data[idx]
        if not isinstance(image, torch.Tensor):
            image = self.transform(image)
        # Don't move to device here
        return {'image': image}
def setup_data(
    batch_size=32,
    image_size=256,
    num_workers=4,
    data_dir="./data",
    device='cuda'
):
    # Load CIFAR-10 dataset with minimal transforms
    cifar_train = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None
    )

    cifar_test = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=None
    )

    # Wrap with our custom dataset
    train_dataset = BEITDatasetCifar(cifar_train, device)
    test_dataset = BEITDatasetCifar(cifar_test, device)

    # Create data loaders with num_workers=0 if using CUDA
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if device == 'cuda' else num_workers,  # Set to 0 for CUDA
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if device == 'cuda' else num_workers,  # Set to 0 for CUDA
        pin_memory=True
    )

    return train_loader, test_loader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_patches=1024, dropout=0.1):  # Changed max_patches to 1024
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_patches, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return self.pe[:, :x.size(1)]

# BEIT Embedding
class BEITEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=768, patches=196, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Linear(vocab_size, embed_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.position_embedding = PositionalEncoding(embed_size, patches+1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tokens, bool_masked_pos=None):
        batch_size = tokens.size(0)
        x = self.token_embedding(tokens)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply masking if specified
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, x.size(1)-1, -1)
            w = bool_masked_pos.unsqueeze(-1)
            x[:, 1:] = x[:, 1:] * (1 - w) + mask_tokens * w

        # Add positional encoding
        x = x + self.position_embedding(x)
        return self.dropout(x)

#  Transformer Components
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class Generator(nn.Module):
    def __init__(self, device, d_model=384, n_heads=6, n_layers=6, temperature=1.0):
        super().__init__()
        self.d_model = d_model
        self.device = device  # Store device

        # Move all components to device
        self.tokenizer = DVAETokenizer(device)
        self.temperature = temperature
        self.patch_size = 8
        input_dim = 8192
        self.patch_embedding = nn.Linear(input_dim, d_model).to(device)
        self.pos_embedding = PositionalEncoding(d_model, max_patches=1024, dropout=0.1).to(device)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(d_model=d_model, n_heads=n_heads, mlp_ratio=4, dropout=0.1).to(device)
            for _ in range(n_layers)
        ])

        self.token_predictor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 8192)
        ).to(device)

    def to(self, device):
        super().to(device)
        self.patch_embedding = self.patch_embedding.to(device)
        self.pos_embedding = self.pos_embedding.to(device)
        self.transformer_layers = nn.ModuleList([layer.to(device) for layer in self.transformer_layers])
        self.token_predictor = self.token_predictor.to(device)
        return self

    def forward(self, image):

            # Ensure image is float32
            image = image.float()

            # Ensure image is in correct format: [B, C, H, W]
            if image.dim() == 3:
                image = image.unsqueeze(0)

            # Ensure input has 3 channels
            if image.size(1) != 3:
                image = image[:, :3, :, :]

            # Ensure input image has 3 channels
            assert image.size(1) == 3, f"Expected 3 channels for input image, got {image.size(1)}"

            # 1. DALL-E Encoder: Image → Tokens
            with torch.no_grad():
                visual_tokens = self.tokenizer.encode(image)  # [B, 8192, H, W]
                original_tokens = visual_tokens.clone()

            # 2. Patch Embeddings
            B, C, H, W = visual_tokens.shape
            # Reshape tokens to [B, H*W, C] format for linear layer
            visual_tokens = visual_tokens.permute(0, 2, 3, 1).reshape(B, H*W, C)
            x = self.patch_embedding(visual_tokens)  # [B, H*W, d_model]
            x = x + self.pos_embedding(x)

            # 3. Random Masking - 55% masking ratio
            num_patches = H * W
            num_mask = int(0.55 * num_patches)
            mask_indices = torch.randperm(num_patches)[:num_mask]
            bool_masked_pos = torch.zeros(num_patches, dtype=torch.bool, device=image.device)
            bool_masked_pos[mask_indices] = True

            self.last_features = x.clone()

            for layer in self.transformer_layers:
                x = layer(x)

            token_logits = self.token_predictor(x)
            masked_logits = token_logits[:, mask_indices] / self.temperature

            sampled_probs = F.softmax(masked_logits, dim=-1)
            sampled_tokens = torch.multinomial(sampled_probs.reshape(-1, 8192), 1).squeeze(-1)

            # Ensure proper token combination
            combined_tokens = original_tokens.clone()
            combined_tokens = combined_tokens.reshape(B, C, -1)
            combined_tokens[:, :, mask_indices] = F.one_hot(sampled_tokens, num_classes=8192).float().transpose(1, 2)
            combined_tokens = combined_tokens.reshape(B, C, H, W)

            # Ensure decoded image has 3 channels
            with torch.no_grad():
                corrupted_image = self.tokenizer.decode(combined_tokens)
                # Add channel check and correction if needed
                if corrupted_image.size(1) != 3:
                    corrupted_image = corrupted_image[:, :3, :, :]  # Take only first 3 channels

            # Add final assertion to guarantee output shape
            assert corrupted_image.size(1) == 3, f"Generated corrupted image has {corrupted_image.size(1)} channels instead of 3"
            # Ensure output is float32
            corrupted_image = corrupted_image.float()

            return corrupted_image, combined_tokens, bool_masked_pos

#Attention Components
class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
        self.scale = head_size ** -0.5

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(attention, dim=-1)
        return attention @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.head_size = d_model // n_heads
        self.heads = nn.ModuleList([
            AttentionHead(d_model, self.head_size) for _ in range(n_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        out = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Training utilities
class ScheduledOptim():
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return min(
            np.power(self.n_current_steps, -0.5),
            self.n_current_steps * np.power(self.n_warmup_steps, -1.5)
        )

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

# Generator Trainer Implementation
class GeneratorTrainer:
    def __init__(
        self,
        generator,
        train_dataloader,
        test_dataloader=None,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=10000,
        log_freq=10,
        device='cuda'
    ):
        self.device = device
        self.generator = generator.to(device)
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Optimizer for generator
        self.gen_optim = Adam(self.generator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.gen_schedule = ScheduledOptim(self.gen_optim, self.generator.d_model, warmup_steps)

        self.criterion = torch.nn.MSELoss()  # For image reconstruction
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.generator.parameters()]))

    def train_step(self, batch):
        self.generator.train()

        # Move batch to device
        images = batch['image'].to(self.device)

        # Forward pass through generator
        corrupted_image, _, _ = self.generator(images)

        # Compute reconstruction loss
        loss = self.criterion(corrupted_image, images)

        # Update generator
        self.gen_schedule.zero_grad()
        loss.backward()
        self.gen_schedule.step_and_update_lr()

        return loss.item()

    def train_epoch(self, epoch):
        total_loss = 0

        # Main progress bar for the epoch
        epoch_iter = tqdm.tqdm(
            enumerate(self.train_data),
            desc=f"Epoch {epoch+1}",
            total=len(self.train_data),
            leave=True,
            position=0
        )

        # Secondary progress bar for loss tracking
        loss_meter = tqdm.tqdm(
            total=0,
            desc="Loss",
            bar_format="{desc}: {postfix[0]:1.4f}",
            position=1,
            leave=True,
            postfix=[0]
        )

        data_iter = tqdm.tqdm(
            enumerate(self.train_data),
            desc=f"EP_train:{epoch}",
            total=len(self.train_data),
            bar_format="{l_bar}{r_bar}"
        )

        for i, batch in epoch_iter:
            loss = self.train_step(batch)
            total_loss += loss

            # Update the loss meter
            current_avg_loss = total_loss / (i + 1)
            loss_meter.postfix[0] = current_avg_loss
            loss_meter.update()

            if i % self.log_freq == 0:
                epoch_iter.set_postfix(loss=f"{current_avg_loss:.4f}")

        # Close progress bars
        epoch_iter.close()
        loss_meter.close()

        avg_loss = total_loss / len(self.train_data)
        return avg_loss


class CIMEnhancer(nn.Module):
    def __init__(self, num_classes: int = 2, feature_dim: int = 384):
        super().__init__()

        # Store feature_dim as class attribute
        self.feature_dim = feature_dim

        # Base ResNet-50 backbone
        resnet = models.resnet50(pretrained=False)
        resnet = resnet.float()

        # Feature Extraction Layers
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

        # RESPIX Head for pixel-wise prediction
        self.respix_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1)
        )

        # Enhanced REVDET head
        self.revdet_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )


    def compute_revdet_labels(self, original_tokens, generated_tokens):
        """
        Compute binary labels for token replacement detection
        Args:
            original_tokens: Original tokens from DALL-E encoder
            generated_tokens: Tokens after generator sampling
        Returns:
            Binary tensor indicating replaced tokens (1) vs original tokens (0)
        """
        return (original_tokens != generated_tokens).float()

    def sliding_window_normalization(self, x: torch.Tensor, window_size: int = 8) -> torch.Tensor:
        """
        Sliding window normalization with proper padding handling

        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
            window_size (int): Size of the normalization window

        Returns:
            torch.Tensor: Normalized image tensor of same size as input
        """
        b, c, h, w = x.shape

        # Ensure window size is odd for symmetric padding
        if window_size % 2 == 0:
            window_size += 1

        # Create averaging kernel for each channel
        kernel = torch.ones(1, 1, window_size, window_size).to(x.device) / (window_size * window_size)
        kernel = kernel.expand(c, 1, window_size, window_size)

        # Calculate padding
        pad = window_size // 2

        # Create a tensor to store local means and variances
        local_mean = torch.zeros_like(x)
        local_var = torch.zeros_like(x)

        for i in range(c):
            # Process each channel
            channel_data = x[:, i:i+1, :, :]

            # Compute mean using conv2d with reflection padding
            channel_mean = F.conv2d(
                F.pad(channel_data, (pad, pad, pad, pad), mode='reflect'),
                kernel[i:i+1],
                padding=0
            )
            local_mean[:, i:i+1, :, :] = channel_mean

            # Compute variance
            channel_sq = channel_data ** 2
            channel_sq_mean = F.conv2d(
                F.pad(channel_sq, (pad, pad, pad, pad), mode='reflect'),
                kernel[i:i+1],
                padding=0
            )
            local_var[:, i:i+1, :, :] = channel_sq_mean - channel_mean ** 2

        # Add epsilon for numerical stability and compute std
        local_std = torch.sqrt(torch.clamp(local_var, min=1e-8))

        # Normalize
        normalized = (x - local_mean) / local_std

        # Verify output shape
        assert normalized.shape == x.shape, \
            f"Output shape {normalized.shape} doesn't match input shape {x.shape}"

        return normalized

    def forward(self, x, corrupted_x, task='respix'):
        """
        Args:
            x: Original image [B, 3, H, W]
            corrupted_x: Corrupted image [B, 3, H, W]
            task: 'respix' or 'revdet'
        """
        # Ensure inputs are float32
        x = x.float()
        corrupted_x = corrupted_x.float()

        # Print shapes for debugging
        print(f"Original image shape: {x.shape}")
        print(f"Corrupted image shape: {corrupted_x.shape}")

        # Verify input shapes
        assert x.size(1) == 3, f"Expected 3 channels for original image, got {x.size(1)}"
        assert corrupted_x.size(1) == 3, f"Expected 3 channels for corrupted image, got {corrupted_x.size(1)}"

        # Extract features from corrupted image only
        features = self.feature_extractor(corrupted_x)
        print(f"Features shape: {features.shape}")

        if task == 'respix':
            # Process original image for target
            normalized_target = self.sliding_window_normalization(x)
            print(f"Normalized target shape: {normalized_target.shape}")

            # Get predictions from corrupted image features
            predictions = self.respix_head(features)
            print(f"Predictions shape: {predictions.shape}")

            # Ensure predictions and target have same shape
            assert predictions.shape == normalized_target.shape, \
                f"Predictions shape {predictions.shape} doesn't match target shape {normalized_target.shape}"

            return predictions, normalized_target

        elif task == 'revdet':
            return self.revdet_head(features)

        raise ValueError(f"Unknown task: {task}")

    def compute_loss(self, predictions, target, task='respix', weights=None):
        """
        Enhanced loss computation with optional weighting
        """
        if task == 'respix':
            # For RESPIX, predictions and target should already be normalized
            l1_loss = nn.functional.l1_loss(predictions, target, reduction='none')
            l2_loss = nn.functional.mse_loss(predictions, target, reduction='none')

            if weights is not None:
                l1_loss = (l1_loss * weights).mean()
                l2_loss = (l2_loss * weights).mean()
            else:
                l1_loss = l1_loss.mean()
                l2_loss = l2_loss.mean()

            return 0.5 * l1_loss + 0.5 * l2_loss


        elif task == 'revdet':
            loss = nn.functional.binary_cross_entropy_with_logits(
                predictions,
                target,
                reduction='none'
            )

            if weights is not None:
                loss = (loss * weights).mean()
            else:
                loss = loss.mean()

            return loss

"""Synergetical Training Loop"""

class CIMTrainer:
    def __init__(
        self,
        generator,
        enhancer,
        train_dataloader,
        device='cuda',
        lr=1e-4,
        weight_decay=0.01,
        warmup_steps=10000,
        mask_ratio=0.55  # ~110 tokens out of 196 as per paper
    ):
        self.device = device
        self.generator = generator.to(device)
        self.enhancer = enhancer.to(device)
        self.train_data = train_dataloader
        self.mask_ratio = mask_ratio

        # Separate optimizers for generator (BEiT) and enhancer
        self.gen_optim = Adam(
            [p for p in self.generator.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay
        )
        self.enh_optim = Adam(self.enhancer.parameters(), lr=lr, weight_decay=weight_decay)

        # Learning rate schedulers
        self.gen_schedule = ScheduledOptim(self.gen_optim, generator.d_model, warmup_steps)
        self.enh_schedule = ScheduledOptim(self.enh_optim, enhancer.feature_dim, warmup_steps)

    def compute_beit_loss(self, predicted_tokens, golden_tokens, mask_indices):
        """
        Compute MIM loss for generator (BEiT) as per paper
        """
        # Only compute loss for masked positions
        pred_masked = predicted_tokens[:, mask_indices]
        gold_masked = golden_tokens[:, mask_indices]

        # Cross entropy loss for token prediction
        return F.cross_entropy(pred_masked, gold_masked)

    def train_step(self, batch, task='respix'):
        self.generator.train()
        self.enhancer.train()

        # Move batch to device here
        images = batch['image'].to(self.device)
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # Get batch size from images tensor
        B = images.size(0)

        # 1. Generator (BEiT) Forward Pass
        with torch.no_grad():
            golden_tokens = self.generator.tokenizer.encode(images)  # [B, 8192, H/8, W/8]

        # Properly reshape tokens
        B, C, H, W = golden_tokens.shape
        golden_tokens = golden_tokens.permute(0, 2, 3, 1)  # [B, H/8, W/8, 8192]
        golden_tokens = golden_tokens.reshape(B, H*W, C)   # [B, (H/8)*(W/8), 8192]

        # Create patch embeddings
        patch_embeddings = self.generator.patch_embedding(golden_tokens)  # [B, (H/8)*(W/8), d_model]
        patch_embeddings = patch_embeddings + self.generator.pos_embedding(patch_embeddings)

        # Random masking (50-60% as per paper)
        num_patches = patch_embeddings.size(1)
        num_mask = int(0.55 * num_patches)
        mask_indices = torch.randperm(num_patches, device=self.device)[:num_mask]
        bool_masked_pos = torch.zeros(num_patches, dtype=torch.bool, device=self.device)
        bool_masked_pos[mask_indices] = True

        # Forward through transformer layers
        x = patch_embeddings.clone()
        for layer in self.generator.transformer_layers:
            x = layer(x)

        # Token prediction
        token_logits = self.generator.token_predictor(x)  # [B, num_patches, 8192]

        # Get the target tokens for masked positions
        target_tokens = golden_tokens.argmax(dim=-1)  # Convert one-hot to indices [B, num_patches]
        masked_logits = token_logits[:, mask_indices, :]  # [B, num_mask, 8192]
        masked_targets = target_tokens[:, mask_indices]   # [B, num_mask]

        # Compute generator loss
        generator_loss = F.cross_entropy(
            masked_logits.reshape(-1, 8192),  # [B*num_mask, 8192]
            masked_targets.reshape(-1)         # [B*num_mask]
        )

        # Sample new tokens for masked positions
        with torch.no_grad():
            sampled_probs = F.softmax(masked_logits / self.generator.temperature, dim=-1)
            sampled_tokens = torch.multinomial(
                sampled_probs.view(-1, 8192),
                1
            ).view(B, -1)  # [B, num_mask]

        # Create corrupted tokens
        corrupted_tokens = golden_tokens.clone()
        corrupted_tokens[:, mask_indices] = F.one_hot(sampled_tokens, num_classes=8192).float()

        # Reshape back to image format
        corrupted_tokens = corrupted_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Generate corrupted image
        with torch.no_grad():
            corrupted_image = self.generator.tokenizer.decode(corrupted_tokens)
                # 2. Enhancer Forward Pass
        if task == 'respix':
            predictions, normalized_target = self.enhancer(
                images,  # original images
                corrupted_image,  # corrupted images
                task='respix'
            )
            enhancer_loss = self.enhancer.compute_loss(
                predictions,
                normalized_target,
                task='respix'
            )
        else:  # revdet
            predictions = self.enhancer(
                images,
                corrupted_image,
                task='revdet'
            )
            replacement_labels = bool_masked_pos.float()
            enhancer_loss = F.binary_cross_entropy_with_logits(
                predictions.squeeze(),
                replacement_labels
            )
        # Update Generator
        self.gen_optim.zero_grad()
        generator_loss.backward()
        self.gen_optim.step()

        # Update Enhancer
        self.enh_optim.zero_grad()
        enhancer_loss.backward()
        self.enh_optim.step()

        return {
            'generator_loss': generator_loss.item(),
            'enhancer_loss': enhancer_loss.item()
        }

    def train_epoch(self, epoch, task='respix'):
        total_gen_loss = 0
        total_enh_loss = 0

        progress_bar = tqdm.tqdm(
            enumerate(self.train_data),
            desc=f"Epoch {epoch+1}",
            total=len(self.train_data)
        )

        for i, batch in progress_bar:
            losses = self.train_step(batch, task)
            total_gen_loss += losses['generator_loss']
            total_enh_loss += losses['enhancer_loss']

            # Update progress bar
            avg_gen_loss = total_gen_loss / (i + 1)
            avg_enh_loss = total_enh_loss / (i + 1)
            progress_bar.set_postfix({
                'gen_loss': f"{avg_gen_loss:.4f}",
                'enh_loss': f"{avg_enh_loss:.4f}"
            })

        return avg_gen_loss, avg_enh_loss


"""Actual Training Loop"""

def train_cim(
    num_epochs=100,
    batch_size=32,
    image_size=256,  # DALL-E expects 256x256
    d_model=384,     # Generator's BEIT dimension
    feature_dim=384, # Enhancer's feature dimension
    n_heads=6,
    n_layers=6,
    device='cuda',
    save_dir='./checkpoints',
    data_dir='./data',
    log_freq=10,
    save_freq=5,
    task='respix',    # 'respix' or 'revdet'
    train_loader=None,  # Add optional data loader parameters
    test_loader=None
):
    """
    Main training function for Corrupted Image Modeling
    """
    print(f"\nInitializing CIM training with parameters:")
    print(f"num_epochs: {num_epochs}")
    print(f"batch_size: {batch_size}")
    print(f"image_size: {image_size}")
    print(f"d_model: {d_model}")
    print(f"task: {task}")
    print(f"device: {device}\n")

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)

    # Setup data if not provided
    if train_loader is None or test_loader is None:
        print("Setting up data loaders...")
        train_loader, test_loader = setup_data(
            batch_size=batch_size,
            image_size=image_size,
            data_dir=data_dir,
            device=device  # Pass device to setup_data
        )
        print(f"Created data loaders with {len(train_loader)} training batches")

    # Initialize models and ensure they're on the correct device
    print("\nInitializing models...")
    generator = Generator(
        device=device,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    ).to(device)  # Explicitly move to device

    enhancer = CIMEnhancer(
        feature_dim=feature_dim
    ).to(device)  # Explicitly move to device

    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters())}")
    print(f"Enhancer parameters: {sum(p.numel() for p in enhancer.parameters())}")

    # Initialize trainer
    trainer = CIMTrainer(
        generator=generator,
        enhancer=enhancer,
        train_dataloader=train_loader,
        device=device
    )

    # Training loop
    print("\nStarting training loop...")
    best_gen_loss = float('inf')
    best_enh_loss = float('inf')

    epoch_progress = tqdm.tqdm(range(num_epochs), desc="Training Progress")
    for epoch in epoch_progress:
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training step
        avg_gen_loss, avg_enh_loss = trainer.train_epoch(epoch, task=task)

        # Update epoch progress bar
        epoch_progress.set_postfix({
            'gen_loss': f"{avg_gen_loss:.4f}",
            'enh_loss': f"{avg_enh_loss:.4f}"
        })

        print(f"Epoch {epoch+1} Summary:")
        print(f"Generator Loss: {avg_gen_loss:.4f}")
        print(f"Enhancer Loss: {avg_enh_loss:.4f}")

        # Save best models
        if avg_gen_loss < best_gen_loss:
            best_gen_loss = avg_gen_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': trainer.gen_optim.state_dict(),
                'loss': avg_gen_loss,
            }, os.path.join(save_dir, 'best_generator.pt'))

        if avg_enh_loss < best_enh_loss:
            best_enh_loss = avg_enh_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': enhancer.state_dict(),
                'optimizer_state_dict': trainer.enh_optim.state_dict(),
                'loss': avg_enh_loss,
            }, os.path.join(save_dir, 'best_enhancer.pt'))

        # Regular checkpoints
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'enhancer_state_dict': enhancer.state_dict(),
                'gen_optimizer_state_dict': trainer.gen_optim.state_dict(),
                'enh_optimizer_state_dict': trainer.enh_optim.state_dict(),
                'gen_loss': avg_gen_loss,
                'enh_loss': avg_enh_loss,
            }, checkpoint_path)

        # Generate and save sample images
        if (epoch + 1) % log_freq == 0:
            generator.eval()
            enhancer.eval()
            with torch.no_grad():
                # Get a batch of test images
                test_batch = next(iter(test_loader))
                test_images = test_batch['image'].to(device)

                # Generate corrupted images
                corrupted_images, _, _ = generator(test_images)

                # Enhance corrupted images
                if task == 'respix':
                    enhanced_images, _ = enhancer(test_images, corrupted_images, task='respix')
                else:  # revdet
                    detection_map = enhancer(test_images, corrupted_images, task='revdet')
                    enhanced_images = corrupted_images  # Just for visualization

                # Save sample images
                for i in range(min(5, batch_size)):  # Save first 5 images
                    sample_path = os.path.join(save_dir, 'samples', f'epoch_{epoch+1}_sample_{i}.png')
                    # Create a grid of original, corrupted, and enhanced images
                    grid = torch.cat([
                        test_images[i].cpu(),
                        corrupted_images[i].cpu(),
                        enhanced_images[i].cpu()
                    ], dim=2)  # Concatenate horizontally
                    save_image(grid, sample_path, normalize=True)

    print("\nTraining completed!")
    return generator, enhancer

# Usage example:
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Train the models
    generator, enhancer = train_cim(
        num_epochs=100,
        batch_size=32,
        device=device,
        task='respix'  # or 'revdet'
# Given the corrupted image sampled from the auxiliary generator, the enhancer learns either a
# # generative (respix) or a discriminative (revdet) visual pretext task.
    )