# NeuroQuantix Optimized - Real Market Features & Enhanced Architecture
# Advanced Financial Market Analysis with Real Data Integration

import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from typing import Tuple, Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def safe_divide(numerator, denominator, eps=1e-8):
    """Safe division to prevent NaN"""
    return numerator / (denominator + eps)

def safe_log(x, eps=1e-8):
    """Safe logarithm to prevent NaN"""
    return torch.log(torch.clamp(x, min=eps))

def safe_sqrt(x, eps=1e-8):
    """Safe square root to prevent NaN"""
    return torch.sqrt(torch.clamp(x, min=eps))

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss with dynamic class weights"""
    def __init__(self, class_weights=None, label_smoothing=0.1):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        if self.class_weights is not None:
            return F.cross_entropy(inputs, targets, weight=self.class_weights, 
                                 label_smoothing=self.label_smoothing, reduction='none')
        else:
            return F.cross_entropy(inputs, targets, label_smoothing=self.label_smoothing, reduction='none')

class BrierScoreLoss(nn.Module):
    """Brier Score Loss for confidence calibration"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, confidence, targets):
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=3).float()
        # Brier score: MSE between confidence and true labels
        brier_loss = F.mse_loss(confidence, targets_one_hot, reduction='none').mean(dim=-1)
        
        if self.reduction == 'mean':
            return brier_loss.mean()
        elif self.reduction == 'sum':
            return brier_loss.sum()
        else:
            return brier_loss

def calculate_class_weights(y_direction):
    """Calculate class weights based on class distribution"""
    class_counts = torch.bincount(y_direction)
    total_samples = len(y_direction)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    return class_weights

def log_attention_weights(attention_weights, regime_seq, epoch, save_dir='attention_logs'):
    """Log attention weights for each volatility regime"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Separate attention weights by regime
    low_vol_attn = []
    high_vol_attn = []
    
    for layer_idx, layer_attn in enumerate(attention_weights):
        # layer_attn shape: [batch, nhead, seq_len, seq_len]
        batch_size, nhead, seq_len, _ = layer_attn.shape
        
        for batch_idx in range(batch_size):
            regime = regime_seq[batch_idx, -1].item()  # Use last timestep regime
            
            if regime == 0:  # Low volatility
                low_vol_attn.append(layer_attn[batch_idx].mean(dim=0).cpu().numpy())
            else:  # High volatility
                high_vol_attn.append(layer_attn[batch_idx].mean(dim=0).cpu().numpy())
    
    # Save attention weights
    if low_vol_attn:
        low_vol_attn = np.array(low_vol_attn)
        np.save(f'{save_dir}/epoch_{epoch}_low_vol_attention.npy', low_vol_attn)
    
    if high_vol_attn:
        high_vol_attn = np.array(high_vol_attn)
        np.save(f'{save_dir}/epoch_{epoch}_high_vol_attention.npy', high_vol_attn)
    
    return len(low_vol_attn), len(high_vol_attn)

def visualize_attention_heatmap(attention_weights, regime_seq, epoch, save_dir='attention_plots'):
    """Visualize attention distribution heatmaps"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Get average attention weights for each regime
    low_vol_attn = []
    high_vol_attn = []
    
    for layer_attn in attention_weights:
        batch_size, nhead, seq_len, _ = layer_attn.shape
        
        for batch_idx in range(batch_size):
            regime = regime_seq[batch_idx, -1].item()
            avg_attn = layer_attn[batch_idx].mean(dim=0).cpu().numpy()
            
            if regime == 0:
                low_vol_attn.append(avg_attn)
            else:
                high_vol_attn.append(avg_attn)
    
    # Create heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Attention Heatmaps - Epoch {epoch}', fontsize=16)
    
    if low_vol_attn:
        low_vol_avg = np.mean(low_vol_attn, axis=0)
        sns.heatmap(low_vol_avg, ax=axes[0, 0], cmap='Blues', cbar_kws={'label': 'Attention Weight'})
        axes[0, 0].set_title('Low Volatility Regime')
        axes[0, 0].set_xlabel('Key Position')
        axes[0, 0].set_ylabel('Query Position')
    
    if high_vol_attn:
        high_vol_avg = np.mean(high_vol_attn, axis=0)
        sns.heatmap(high_vol_avg, ax=axes[0, 1], cmap='Reds', cbar_kws={'label': 'Attention Weight'})
        axes[0, 1].set_title('High Volatility Regime')
        axes[0, 1].set_xlabel('Key Position')
        axes[0, 1].set_ylabel('Query Position')
    
    # Attention distribution comparison
    if low_vol_attn and high_vol_attn:
        low_flat = np.array(low_vol_attn).flatten()
        high_flat = np.array(high_vol_attn).flatten()
        
        axes[1, 0].hist(low_flat, bins=50, alpha=0.7, label='Low Vol', color='blue')
        axes[1, 0].hist(high_flat, bins=50, alpha=0.7, label='High Vol', color='red')
        axes[1, 0].set_title('Attention Weight Distribution')
        axes[1, 0].set_xlabel('Attention Weight')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Attention entropy comparison
        low_entropy = -np.sum(low_flat * np.log(low_flat + 1e-8))
        high_entropy = -np.sum(high_flat * np.log(high_flat + 1e-8))
        
        axes[1, 1].bar(['Low Vol', 'High Vol'], [low_entropy, high_entropy], 
                      color=['blue', 'red'], alpha=0.7)
        axes[1, 1].set_title('Attention Entropy by Regime')
        axes[1, 1].set_ylabel('Entropy')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/attention_heatmap_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

class LearnableVolatilityPositionalEncoding(nn.Module):
    """Learnable volatility-aware positional encoding"""
    def __init__(self, d_model, max_len=5000, vol_embed_dim=8):
        super(LearnableVolatilityPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.vol_embed_dim = vol_embed_dim
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))
        
        # Volatility-aware modulation
        self.vol_projection = nn.Linear(vol_embed_dim, d_model)
        self.vol_gate = nn.Sequential(
            nn.Linear(vol_embed_dim, d_model),
            nn.Sigmoid()
        )
        
        # Initialize weights
        nn.init.normal_(self.pos_encoding, std=0.02)
    
    def forward(self, x, vol_embedding=None):
        seq_len = x.size(1)
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        if vol_embedding is not None:
            # Volatility-aware modulation
            vol_proj = self.vol_projection(vol_embedding)
            vol_gate = self.vol_gate(vol_embedding)
            
            # Modulate positional encoding with volatility information
            modulated_pos_enc = pos_enc * (1 + vol_gate) + vol_proj
            return x + modulated_pos_enc
        else:
            return x + pos_enc

class EnhancedInputLayer(nn.Module):
    """Enhanced Input Layer with Volatility Integration"""
    def __init__(self, input_dim=37, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature projection layers (fix: use input_dim - 1)
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim - 1, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Volatility-aware feature enhancement
        self.vol_enhancement = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for volatility regime
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Residual connection - match features_without_vol
        self.residual_projection = nn.Linear(input_dim - 1, hidden_dim)
        
    def forward(self, x):
        # Check for NaN in input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        batch_size, seq_len, features = x.shape
        # Extract volatility regime (last feature)
        volatility_regime = x[:, :, -1:].float()  # Shape: (batch, seq, 1)
        features_without_vol = x[:, :, :-1]  # Shape: (batch, seq, features-1)
        # Project features
        projected = self.feature_projection(features_without_vol)
        # Volatility enhancement
        vol_enhanced_input = torch.cat([projected, volatility_regime], dim=-1)
        vol_enhanced = self.vol_enhancement(vol_enhanced_input)
        # Residual connection
        residual = self.residual_projection(features_without_vol)
        # Combine with residual connection
        output = vol_enhanced + residual
        # Check for NaN in output
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)
        return output

class AdvancedMultiScaleConvolution(nn.Module):
    """Advanced Multi-Scale Convolution with Attention"""
    def __init__(self, input_dim=256, kernel_sizes=[3, 7, 15, 31]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        
        # Enhanced convolution layers with attention
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(input_dim),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Conv1d(input_dim, input_dim, kernel_size=1),
                nn.BatchNorm1d(input_dim),
                nn.GELU()
            ) for k in kernel_sizes
        ])
        
        # Projection to d_model for attention
        self.proj_to_d_model = nn.Linear(input_dim * len(kernel_sizes), input_dim)
        
        # Attention mechanism for feature selection
        self.attention = nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True)
        
        # Enhanced fusion
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        # Check for NaN in input
        if torch.isnan(x).any():
            print("⚠️ Warning: NaN detected in input to AdvancedMultiScaleConvolution")
            x = torch.nan_to_num(x, nan=0.0)
        
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        
        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(x)
            conv_outputs.append(conv_out.transpose(1, 2))  # Back to [batch, seq_len, input_dim]
        
        # Concatenate and project to d_model
        combined = torch.cat(conv_outputs, dim=-1)  # [batch, seq_len, input_dim * len(kernel_sizes)]
        combined_proj = self.proj_to_d_model(combined)  # [batch, seq_len, input_dim]
        
        # Apply attention
        attended, _ = self.attention(combined_proj, combined_proj, combined_proj)
        
        # Fusion
        output = self.fusion(attended)
        
        # Check for NaN in output
        if torch.isnan(output).any():
            print("⚠️ Warning: NaN detected in AdvancedMultiScaleConvolution output")
            output = torch.nan_to_num(output, nan=0.0)
        
        return output

class TemporalConvResidualBlock(nn.Module):
    """Enhanced Temporal Convolutional Residual Block with Deep Volatility Integration"""
    def __init__(self, input_dim, hidden_dim=128, kernel_size=5, vol_embed_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        # Main convolution path
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, input_dim, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(input_dim)
        # Volatility integration layers
        self.vol_projection = nn.Linear(vol_embed_dim, hidden_dim)
        self.vol_gate = nn.Sequential(
            nn.Linear(vol_embed_dim, input_dim),
            nn.Sigmoid()
        )
        # Enhanced volatility fusion
        self.vol_fusion = nn.Sequential(
            nn.Linear(input_dim + vol_embed_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, vol_embedding=None):
        # Check for NaN in input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        residual = x
        # Transpose for convolution
        x_conv = x.transpose(1, 2)  # (batch, features, seq)
        # First convolution
        out = self.conv1(x_conv)
        if vol_embedding is not None:
            # Integrate volatility information
            vol_proj = self.vol_projection(vol_embedding).transpose(1, 2)  # (batch, hidden, seq)
            out = out + vol_proj
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        # Transpose back
        out = out.transpose(1, 2)  # (batch, seq, features)
        # Residual connection
        out = out + residual
        # Deep volatility fusion
        if vol_embedding is not None:
            vol_gate = self.vol_gate(vol_embedding)
            out = out * (1 + vol_gate)
            # Additional fusion layer
            vol_fused_input = torch.cat([out, vol_embedding], dim=-1)
            out = self.vol_fusion(vol_fused_input)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class VolatilityEmbedding(nn.Module):
    """Enhanced volatility regime embedding with learnable positional encoding"""
    def __init__(self, embed_dim=8, seq_len=24):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # Basic regime embedding
        self.regime_embed = nn.Embedding(2, embed_dim)  # 2 regimes: low/high
        
        # Learnable volatility positional encoding
        self.vol_pos_encoding = nn.Parameter(torch.randn(seq_len, embed_dim))
        
        # Volatility-aware feature fusion
        self.vol_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
    def forward(self, regime_seq):
        # Check for NaN in input
        if torch.isnan(regime_seq).any():
            print("⚠️ Warning: NaN detected in regime_seq input to VolatilityEmbedding")
            regime_seq = torch.nan_to_num(regime_seq, nan=0.0)
        
        # regime_seq: [batch, seq] of 0/1
        # Ensure regime_seq is integer type for embedding
        regime_seq = regime_seq.long().clamp(0, 1)
        
        # Basic regime embedding
        regime_emb = self.regime_embed(regime_seq)  # [batch, seq, embed_dim]
        
        # Add learnable positional encoding
        batch_size, seq_len = regime_seq.shape
        pos_encoding = self.vol_pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Fuse regime embedding with positional encoding
        combined = torch.cat([regime_emb, pos_encoding], dim=-1)
        fused_embedding = self.vol_fusion(combined)
        
        # Check for NaN in output
        if torch.isnan(fused_embedding).any():
            print("⚠️ Warning: NaN detected in VolatilityEmbedding output")
            fused_embedding = torch.nan_to_num(fused_embedding, nan=0.0)
        
        return fused_embedding

class RegimeAwareMultiheadAttention(nn.Module):
    """Multihead attention with regime-aware scaling of attention scores"""
    def __init__(self, d_model, nhead, dropout=0.15):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.regime_gate = nn.Linear(1, d_model)  # 1=regime, output=d_model (for scaling)
        
    def forward(self, x, regime_seq):
        # Check for NaN in inputs
        if torch.isnan(x).any():
            print("⚠️ Warning: NaN detected in x input to RegimeAwareMultiheadAttention")
            x = torch.nan_to_num(x, nan=0.0)
        if torch.isnan(regime_seq).any():
            print("⚠️ Warning: NaN detected in regime_seq input to RegimeAwareMultiheadAttention")
            regime_seq = torch.nan_to_num(regime_seq, nan=0.0)
        
        # x: [batch, seq, d_model], regime_seq: [batch, seq]
        # Compute regime gate (sigmoid scaling)
        regime_gate = torch.sigmoid(self.regime_gate(regime_seq.unsqueeze(-1).float()))  # [batch, seq, d_model]
        x_mod = x * (1 + regime_gate)  # Emphasize features in high regime
        attn_out, attn_weights = self.attn(x_mod, x_mod, x_mod)
        
        # Check for NaN in output
        if torch.isnan(attn_out).any():
            print("⚠️ Warning: NaN detected in RegimeAwareMultiheadAttention output")
            attn_out = torch.nan_to_num(attn_out, nan=0.0)
        
        return attn_out, attn_weights

class EnhancedTransformerEncoder(nn.Module):
    """Enhanced Transformer with Regime-Aware Attention"""
    def __init__(self, d_model=256, nhead=16, num_layers=6, dropout=0.15, regime_aware=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.regime_aware = regime_aware
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=1000)
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerLayer(d_model, nhead, dropout, regime_aware=regime_aware) 
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.global_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
    def forward(self, x, regime_seq=None, return_attention=False):
        # Check for NaN in input
        if torch.isnan(x).any():
            print("⚠️ Warning: NaN detected in input to EnhancedTransformerEncoder")
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self.pos_encoding(x)
        attn_weights_all = []
        for layer in self.transformer_layers:
            x, attn_weights = layer(x, regime_seq)
            attn_weights_all.append(attn_weights)
        global_attended, _ = self.global_attention(x, x, x)
        output = self.layer_norm(global_attended)
        
        # Check for NaN in output
        if torch.isnan(output).any():
            print("⚠️ Warning: NaN detected in EnhancedTransformerEncoder output")
            output = torch.nan_to_num(output, nan=0.0)
        
        if return_attention:
            return output, attn_weights_all
        return output

class EnhancedTransformerLayer(nn.Module):
    """Enhanced Transformer Layer with Regime-Aware Attention"""
    def __init__(self, d_model, nhead, dropout=0.15, regime_aware=False):
        super().__init__()
        self.regime_aware = regime_aware
        if regime_aware:
            self.self_attention = RegimeAwareMultiheadAttention(d_model, nhead, dropout)
        else:
            self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, regime_seq=None):
        # Check for NaN in input
        if torch.isnan(x).any():
            print("⚠️ Warning: NaN detected in input to EnhancedTransformerLayer")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Self-attention
        if self.regime_aware and regime_seq is not None:
            attn_out, attn_weights = self.self_attention(x, regime_seq)
        else:
            attn_out, attn_weights = self.self_attention(x, x, x)
        
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        # Check for NaN in output
        if torch.isnan(x).any():
            print("⚠️ Warning: NaN detected in EnhancedTransformerLayer output")
            x = torch.nan_to_num(x, nan=0.0)
        
        return x, attn_weights

class AdvancedPredictionHeads(nn.Module):
    """Advanced prediction heads with confidence and volatility estimation"""
    def __init__(self, d_model=256, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        
        # Global attention for feature aggregation
        self.global_attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
        
        # Direction prediction (3 classes: down, sideways, up)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)
        )
        
        # Return prediction
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Confidence estimation (3 values for each class)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 3),  # 3 confidence values for each class
            nn.Sigmoid()
        )
        
        # Volatility estimation
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        # Check for NaN in input
        if torch.isnan(x).any():
            print("⚠️ Warning: NaN detected in input to AdvancedPredictionHeads")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Global pooling with attention
        # Use mean and max pooling
        mean_pooled = torch.mean(x, dim=1)  # [batch, d_model]
        max_pooled = torch.max(x, dim=1)[0]  # [batch, d_model]
        
        # Combine pooling methods
        pooled = mean_pooled + max_pooled
        
        # Apply global attention
        attended, _ = self.global_attention(pooled.unsqueeze(1), pooled.unsqueeze(1), pooled.unsqueeze(1))
        attended = attended.squeeze(1)  # [batch, d_model]
        
        # Generate predictions
        direction_logits = self.direction_head(attended)
        return_pred = self.return_head(attended)
        confidence = self.confidence_head(attended)
        volatility = self.volatility_head(attended)
        
        # Check for NaN in outputs
        outputs = {
            'direction': direction_logits,
            'return': return_pred,
            'confidence': confidence,
            'volatility': volatility
        }
        
        for key, value in outputs.items():
            if torch.isnan(value).any():
                print(f"⚠️ Warning: NaN detected in {key} prediction")
                outputs[key] = torch.nan_to_num(value, nan=0.0)
        
        return outputs

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class OptimizedNeuroQuantixModel(nn.Module):
    """Optimized NeuroQuantix Model with Enhanced Regime-Aware Features"""
    def __init__(self, input_dim=37, d_model=256, nhead=16, num_layers=6, 
                 kernel_sizes=[3, 7, 15, 31], dropout=0.15, vol_embed_dim=8, seq_len=24):
        super().__init__()
        # Ensure d_model is divisible by nhead
        adjusted_d_model = ((d_model + vol_embed_dim) // nhead) * nhead
        self.input_layer = EnhancedInputLayer(input_dim, d_model)
        
        # Learnable volatility positional encoding
        self.vol_pos_encoding = LearnableVolatilityPositionalEncoding(d_model, seq_len, vol_embed_dim)
        
        self.tcrb = TemporalConvResidualBlock(d_model, hidden_dim=128, kernel_size=5, vol_embed_dim=vol_embed_dim)
        self.vol_embed = VolatilityEmbedding(embed_dim=vol_embed_dim, seq_len=seq_len)
        
        # Project to adjusted dimension if needed
        if d_model + vol_embed_dim != adjusted_d_model:
            self.proj_to_adjusted = nn.Linear(d_model + vol_embed_dim, adjusted_d_model)
        else:
            self.proj_to_adjusted = nn.Identity()
        
        self.conv_layer = AdvancedMultiScaleConvolution(d_model + vol_embed_dim, kernel_sizes)
        self.transformer = EnhancedTransformerEncoder(adjusted_d_model, nhead, num_layers, dropout, regime_aware=True)
        self.prediction_heads = AdvancedPredictionHeads(adjusted_d_model, dropout)
        
    def forward(self, x, regime_seq, return_attention=False):
        # Check for NaN in inputs
        if torch.isnan(x).any():
            print("⚠️ Warning: NaN detected in model input x")
            x = torch.nan_to_num(x, nan=0.0)
        if torch.isnan(regime_seq).any():
            print("⚠️ Warning: NaN detected in model input regime_seq")
            regime_seq = torch.nan_to_num(regime_seq, nan=0.0)
        
        # x: [batch, seq, features], regime_seq: [batch, seq] (0/1)
        x = self.input_layer(x)
        
        # Enhanced volatility embedding integration
        regime_seq = regime_seq.round().clamp(0, 1)
        vol_emb = self.vol_embed(regime_seq)
        
        # Apply learnable volatility positional encoding
        x = self.vol_pos_encoding(x, vol_emb)
        
        # Integrate volatility embedding in TCRB
        x = self.tcrb(x, vol_emb)
        
        # Concatenate with volatility embedding for conv layer
        x = torch.cat([x, vol_emb], dim=-1)
        x = self.conv_layer(x)
        
        # Project to adjusted dimension for transformer
        x = self.proj_to_adjusted(x)
        
        # Transformer with attention weights
        out, attn_weights = self.transformer(x, regime_seq, return_attention=True)
        predictions = self.prediction_heads(out)
        
        # Final NaN check
        for key in predictions:
            if torch.isnan(predictions[key]).any():
                print(f"⚠️ Warning: NaN detected in final {key} prediction")
                predictions[key] = torch.nan_to_num(predictions[key], nan=0.0)
        
        if return_attention:
            return predictions, attn_weights
        return predictions

class RealMarketDataProcessor:
    """Real Market Data Processor with Enhanced Feature Engineering"""
    def __init__(self):
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        
    def fetch_real_market_data(self, symbol='BTC-USD', hours=2000):
        """Fetch real market data with strict error handling (no simulation)"""
        print(f"📊 Fetching real market data for {symbol}...")
        
        # Fetch data with 1-hour intervals
        ticker = yf.Ticker(symbol)
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(hours=hours + 24)  # Extra buffer
        
        df = ticker.history(start=start_date, end=end_date, interval='1h')
        
        if df.empty:
            raise RuntimeError(f"No data received from yfinance for {symbol}.")
        
        # Select only the required columns and reset index
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.reset_index()
        # Some yfinance versions return 'Datetime', some 'Date' or 'index' as the first column
        if 'Datetime' not in df.columns:
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'Datetime'})
            elif 'index' in df.columns:
                df = df.rename(columns={'index': 'Datetime'})
            else:
                df.insert(0, 'Datetime', df.index)
        # Now select only the columns we want
        df = df[['Datetime'] + required_cols]
        df = df.dropna()
        
        print(f"✅ Fetched {len(df)} data points for {symbol}")
        return df
    
    def engineer_real_features(self, df):
        """Engineer real market features using manual technical indicators and volatility regime"""
        print("🔄 Engineering real market features...")
        print(f"📈 Input data shape: {df.shape}")
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = df['Close']  # Fallback
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']
        
        # Volume features
        df['Volume_MA_6'] = df['Volume'].rolling(6).mean()
        df['Volume_MA_12'] = df['Volume'].rolling(12).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_6']
        df['Volume_Price_Trend'] = (df['Volume'] * df['Returns']).rolling(6).sum()
        
        # Technical indicators (manual implementation)
        # Moving averages
        df['SMA_6'] = df['Close'].rolling(6).mean()
        df['SMA_12'] = df['Close'].rolling(12).mean()
        df['EMA_6'] = df['Close'].ewm(span=6).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_24'] = df['Close'].ewm(span=24).mean()
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'] = df['EMA_6'] - df['EMA_12']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic
        df['Stoch_K'] = self._calculate_stochastic(df['High'], df['Low'], df['Close'])
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # Williams %R
        df['Williams_R'] = self._calculate_williams_r(df['High'], df['Low'], df['Close'])
        
        # CCI (Commodity Channel Index)
        df['CCI'] = self._calculate_cci(df['High'], df['Low'], df['Close'])
        
        # ADX (Average Directional Index)
        df['ADX'] = self._calculate_adx(df['High'], df['Low'], df['Close'])
        
        # Momentum features
        df['Price_Momentum_6'] = df['Close'] / df['Close'].shift(6) - 1
        df['Price_Momentum_12'] = df['Close'] / df['Close'].shift(12) - 1
        df['Volume_Momentum_6'] = df['Volume'] / df['Volume'].shift(6) - 1
        
        # Volatility features
        df['Volatility_6'] = df['Returns'].rolling(6).std()
        df['Volatility_12'] = df['Returns'].rolling(12).std()
        df['Volatility_24'] = df['Returns'].rolling(24).std()
        
        # Binary volatility regime feature (1 if 24h rolling std > median, else 0)
        vol_24 = df['Volatility_24']
        median_vol_24 = vol_24.median()
        df['Volatility_Regime'] = (vol_24 > median_vol_24).astype(int)
        
        # Advanced features
        df['Price_Acceleration'] = df['Returns'].diff()
        df['Volume_Acceleration'] = df['Volume'].pct_change()
        df['Price_Volume_Correlation'] = df['Returns'].rolling(12).corr(df['Volume'].pct_change())
        
        # Market microstructure features (realistic estimates)
        df['Spread_Estimate'] = df['Price_Range'] * 0.1  # Estimate spread as 10% of price range
        df['Liquidity_Score'] = df['Volume'] / df['Volatility_6']  # Volume per unit volatility
        
        # Trend features
        df['Trend_6'] = np.where(df['SMA_6'] > df['SMA_12'], 1, -1)
        df['Trend_12'] = np.where(df['SMA_12'] > df['EMA_24'], 1, -1)
        df['Trend_Strength'] = abs(df['EMA_6'] - df['EMA_24']) / df['EMA_24']
        
        # Support/Resistance levels (simplified)
        df['Support_Level'] = df['Low'].rolling(24).min()
        df['Resistance_Level'] = df['High'].rolling(24).max()
        df['Price_vs_Support'] = (df['Close'] - df['Support_Level']) / df['Close']
        df['Price_vs_Resistance'] = (df['Resistance_Level'] - df['Close']) / df['Close']
        
        result = df.dropna()
        print(f"📈 Data after feature engineering and dropna: {result.shape}")
        print(f"📈 NaN count before dropna: {df.isna().sum().sum()}")
        
        return result
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI manually"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, high, low, close, period=14):
        """Calculate Stochastic %K"""
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        return k_percent
    
    def _calculate_williams_r(self, high, low, close, period=14):
        """Calculate Williams %R"""
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def _calculate_cci(self, high, low, close, period=20):
        """Calculate CCI (Commodity Channel Index)"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def _calculate_adx(self, high, low, close, period=14):
        """Calculate ADX (Average Directional Index) - simplified"""
        # Simplified ADX calculation
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / pd.Series(tr).rolling(period).mean())
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / pd.Series(tr).rolling(period).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.fillna(25)  # Fill NaN with neutral value
    
    def preprocess_optimized_data(self, df, seq_len=24):
        """Preprocess data with percentile-based direction labeling and rolling z-score normalization"""
        print("🔄 Preprocessing optimized data...")
        
        # Select the most predictive features
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'LogReturns', 'Price_Range', 'Body_Size',
            'Volume_MA_6', 'Volume_Ratio', 'Volume_Price_Trend',
            'SMA_6', 'SMA_12', 'EMA_6', 'EMA_12', 'EMA_24',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Position', 'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI', 'ADX',
            'Price_Momentum_6', 'Price_Momentum_12', 'Volume_Momentum_6',
            'Volatility_6', 'Volatility_12', 'Volatility_24', 'Volatility_Regime',
            'Price_Acceleration', 'Volume_Acceleration', 'Price_Volume_Correlation'
        ]
        
        # Ensure all columns exist
        available_columns = [col for col in feature_columns if col in df.columns]
        missing_columns = [col for col in feature_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
            for col in missing_columns:
                df[col] = 0
        
        # Handle NaN values in features before scaling
        features = df[feature_columns].copy()
        
        # Fill NaN values with forward fill, then backward fill, then 0
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Check for infinite values and replace them
        features = features.replace([np.inf, -np.inf], 0)
        
        # Rolling z-score normalization for high volatility periods
        zscore_features = features.copy()
        rolling_window = 24
        for col in features.columns:
            rolling_mean = features[col].rolling(rolling_window, min_periods=1).mean()
            rolling_std = features[col].rolling(rolling_window, min_periods=1).std()
            # Add small epsilon to prevent division by zero
            rolling_std = rolling_std + 1e-8
            zscore = (features[col] - rolling_mean) / rolling_std
            # Only apply z-score if volatility regime is high
            zscore_features[col] = np.where(df['Volatility_Regime'] == 1, zscore, features[col])
        
        features = zscore_features
        
        # Scale features using robust scaler
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Check for NaN after scaling
        if np.isnan(features_scaled).any():
            print("⚠️ Warning: NaN detected after scaling. Replacing with zeros.")
            features_scaled = np.nan_to_num(features_scaled, nan=0.0)
        
        # Generate sequences and labels with proper time alignment
        X, regime_seq, y_direction, y_return = [], [], [], []
        
        # Compute future returns for all samples
        all_future_returns = features_scaled[seq_len:, 5]  # Returns column
        
        # Handle NaN in future returns
        if np.isnan(all_future_returns).any():
            print("⚠️ Warning: NaN detected in future returns. Replacing with zeros.")
            all_future_returns = np.nan_to_num(all_future_returns, nan=0.0)
        
        # Enhanced return normalization using log-scaling
        returns_abs = np.abs(all_future_returns)
        returns_sign = np.sign(all_future_returns)
        # Log-scale the absolute values to reduce extreme values
        log_returns_abs = np.log(returns_abs + 1e-8)
        normalized_returns = returns_sign * log_returns_abs
        
        # Use percentiles for thresholds on normalized returns
        down_thres = np.percentile(normalized_returns, 33)
        up_thres = np.percentile(normalized_returns, 66)
        
        # Get the volatility regime column (last one in features)
        vol_regime_col = features['Volatility_Regime'].values.astype(int)
        
        for i in range(seq_len, len(features_scaled) - 1):  # -1 to avoid look-ahead bias
            X.append(features_scaled[i-seq_len:i])
            # Regime sequence for this window
            regime_seq.append(vol_regime_col[i-seq_len:i])
            # Future return (next period) - use normalized returns
            future_return = normalized_returns[i-seq_len]  # Index adjusted for sequence
            
            # Handle NaN in future return
            if np.isnan(future_return):
                future_return = 0.0
            
            # Percentile-based direction labeling
            if future_return > up_thres:
                direction = 2  # Up
            elif future_return < down_thres:
                direction = 0  # Down
            else:
                direction = 1  # Sideways
            y_direction.append(direction)
            y_return.append(future_return)
        
        # Analyze class distribution
        y_direction_np = np.array(y_direction)
        unique_classes, class_counts = np.unique(y_direction_np, return_counts=True)
        total_samples = len(y_direction_np)
        
        print("\n📊 CLASS DISTRIBUTION ANALYSIS:")
        print("-" * 40)
        for class_idx, count in zip(unique_classes, class_counts):
            class_name = ['Down', 'Sideways', 'Up'][class_idx]
            percentage = (count / total_samples) * 100
            print(f"Class {class_idx} ({class_name}): {count} samples ({percentage:.1f}%)")
        
        # Calculate class imbalance ratio
        max_count = np.max(class_counts)
        min_count = np.min(class_counts)
        imbalance_ratio = max_count / min_count
        print(f"Class Imbalance Ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2.0:
            print("⚠️ Warning: Significant class imbalance detected. Consider using weighted loss.")
        
        # Store return normalization parameters for denormalization
        self.return_sign = returns_sign
        self.return_abs = returns_abs
        self.normalized_returns = normalized_returns
        
        if len(X) == 0:
            raise RuntimeError("No valid sequences generated after preprocessing. Try increasing the 'hours' parameter or check feature engineering.")
        
        X = torch.tensor(X, dtype=torch.float32)
        regime_seq = torch.tensor(regime_seq, dtype=torch.long)
        y_direction = torch.tensor(y_direction, dtype=torch.long)
        y_return = torch.tensor(y_return, dtype=torch.float32).unsqueeze(1)
        
        # Final NaN check for tensors
        if torch.isnan(X).any():
            print("⚠️ Warning: NaN detected in X tensor. Replacing with zeros.")
            X = torch.nan_to_num(X, nan=0.0)
        
        if torch.isnan(y_return).any():
            print("⚠️ Warning: NaN detected in y_return tensor. Replacing with zeros.")
            y_return = torch.nan_to_num(y_return, nan=0.0)
        
        # Robust print for shape
        if X.ndim == 3:
            print(f"✅ Generated {len(X)} sequences with {X.shape[2]} features")
        else:
            print(f"✅ Generated {len(X)} sequences (tensor shape: {tuple(X.shape)})")
        
        print(f"✅ X shape: {X.shape}, regime_seq shape: {regime_seq.shape}")
        print(f"✅ y_direction shape: {y_direction.shape}, y_return shape: {y_return.shape}")
        
        return X, regime_seq, y_direction, y_return, available_columns

class OptimizedMarketAnalyzer:
    """Optimized Market Analyzer with Enhanced Training"""
    def __init__(self):
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_names = None
        self.metrics = None
        
    def train_optimized_model(self, X, regime_seq, y_direction, y_return, epochs=150, lr=0.0005):
        """Enhanced training with improved generalization and class balance"""
        print("🔄 Training Enhanced NeuroQuantix Model with Improved Generalization...")
        
        # Input validation
        if torch.isnan(X).any():
            print("⚠️ Warning: NaN detected in input X. Replacing with zeros.")
            X = torch.nan_to_num(X, nan=0.0)
        
        if torch.isnan(regime_seq).any():
            print("⚠️ Warning: NaN detected in input regime_seq. Replacing with zeros.")
            regime_seq = torch.nan_to_num(regime_seq, nan=0.0)
        
        if torch.isnan(y_return).any():
            print("⚠️ Warning: NaN detected in input y_return. Replacing with zeros.")
            y_return = torch.nan_to_num(y_return, nan=0.0)
        
        # Analyze class distribution and calculate weights
        print("\n📊 ANALYZING CLASS DISTRIBUTION:")
        print("-" * 40)
        y_direction_np = y_direction.numpy()
        unique_classes, class_counts = np.unique(y_direction_np, return_counts=True)
        total_samples = len(y_direction_np)
        
        for class_idx, count in zip(unique_classes, class_counts):
            class_name = ['Down', 'Sideways', 'Up'][class_idx]
            percentage = (count / total_samples) * 100
            print(f"Class {class_idx} ({class_name}): {count} samples ({percentage:.1f}%)")
        
        # Calculate dynamic class weights
        class_weights = calculate_class_weights(y_direction)
        print(f"Class Weights: {class_weights.numpy()}")
        
        # Initialize model with enhanced features
        input_dim = X.shape[2]
        seq_len = X.shape[1]
        self.model = OptimizedNeuroQuantixModel(
            input_dim=input_dim,
            d_model=256,
            nhead=16,
            num_layers=6,
            kernel_sizes=[3, 7, 15, 31],
            dropout=0.15,
            vol_embed_dim=8,
            seq_len=seq_len
        )
        
        # Enhanced optimizer with better numerical stability
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        
        # Enhanced loss functions with better class balancing
        direction_criterion = WeightedCrossEntropyLoss(class_weights=class_weights, label_smoothing=0.15)
        focal_criterion = FocalLoss(alpha=1, gamma=2.5, reduction='none')  # Increased gamma for harder examples
        return_criterion = nn.HuberLoss(delta=0.05, reduction='none')  # Smaller delta for better precision
        brier_criterion = BrierScoreLoss(reduction='none')
        
        # Data splitting
        split_idx = int(0.75 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        regime_train, regime_val = regime_seq[:split_idx], regime_seq[split_idx:]
        y_dir_train, y_dir_val = y_direction[:split_idx], y_direction[split_idx:]
        y_ret_train, y_ret_val = y_return[:split_idx], y_return[split_idx:]
        
        print(f"📊 Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
        
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        # Enhanced loss weights with better balance
        direction_weight = 1.5  # Increased for better direction accuracy
        focal_weight = 0.8  # Increased focal loss weight for hard examples
        return_weight = 1.5  # Reduced to balance with direction
        sharpe_weight = 0.2  # Increased for better risk-adjusted returns
        brier_weight = 0.4  # Increased confidence calibration weight
        alpha = 1.2  # Increased volatility penalty strength
        
        # Attention logging
        attention_log_interval = 25  # Log attention every 25 epochs
        
        for epoch in range(epochs):
            try:
                self.model.train()
                optimizer.zero_grad()
                
                # Forward pass with attention weights
                predictions, attention_weights = self.model(X_train, regime_train, return_attention=True)
                
                # Check for NaN in predictions
                nan_detected = False
                for key in predictions:
                    if torch.isnan(predictions[key]).any():
                        print(f"⚠️ Warning: NaN detected in {key} during training. Skipping epoch.")
                        nan_detected = True
                        break
                
                if nan_detected:
                    continue
                
                # Volatility weights: 1 + alpha * regime (high=2, low=1)
                vol_weights = 1.0 + alpha * regime_train[:, -1].float()
                
                # Enhanced direction loss with both weighted CE and focal loss
                dir_loss_weighted = direction_criterion(predictions['direction'], y_dir_train)
                dir_loss_focal = focal_criterion(predictions['direction'], y_dir_train)
                dir_loss = (dir_loss_weighted * vol_weights).mean() + focal_weight * (dir_loss_focal * vol_weights).mean()
                
                # Enhanced return loss with improved confidence gating
                confidence = predictions['confidence'].squeeze()
                # Use class-specific confidence thresholds
                class_confidence_thresholds = torch.tensor([0.6, 0.5, 0.6], device=confidence.device)  # Higher for down/up
                confidence_mask = torch.zeros_like(confidence, dtype=torch.bool)
                
                for class_idx in range(3):
                    class_mask = (y_dir_train == class_idx)
                    class_conf = confidence[class_mask]
                    if class_conf.numel() > 0:
                        threshold = class_confidence_thresholds[class_idx]
                        class_confidence_mask = class_conf > threshold
                        confidence_mask[class_mask] = class_confidence_mask
                
                if confidence_mask.sum() > 0:
                    ret_loss = return_criterion(predictions['return'].squeeze(), y_ret_train.squeeze())
                    ret_loss = (ret_loss * vol_weights * confidence_mask.float()).sum() / confidence_mask.sum()
                else:
                    ret_loss = return_criterion(predictions['return'].squeeze(), y_ret_train.squeeze())
                    ret_loss = (ret_loss * vol_weights).mean()
                
                # Sharpe proxy (maximize mean/std of predicted returns)
                pred_returns = predictions['return'].squeeze()
                pred_std = pred_returns.std() + 1e-8
                sharpe_proxy = pred_returns.mean() / pred_std
                
                # Brier score for confidence calibration
                brier_loss = brier_criterion(confidence, y_dir_train)
                brier_loss = (brier_loss * vol_weights).mean()
                
                # Check for NaN in losses
                if torch.isnan(dir_loss) or torch.isnan(ret_loss) or torch.isnan(sharpe_proxy) or torch.isnan(brier_loss):
                    print("⚠️ Warning: NaN detected in loss calculation. Skipping epoch.")
                    continue
                
                # Enhanced multi-objective loss
                total_loss = (direction_weight * dir_loss + 
                            return_weight * ret_loss - 
                            sharpe_weight * sharpe_proxy + 
                            brier_weight * brier_loss)
                
                # Check for NaN in total loss
                if torch.isnan(total_loss):
                    print("⚠️ Warning: NaN detected in total loss. Skipping epoch.")
                    continue
                
                # Backward pass with gradient clipping
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()
                
                # Log attention weights periodically
                if (epoch + 1) % attention_log_interval == 0:
                    low_vol_count, high_vol_count = log_attention_weights(
                        attention_weights, regime_train, epoch + 1
                    )
                    visualize_attention_heatmap(attention_weights, regime_train, epoch + 1)
                    print(f"📊 Attention logged: {low_vol_count} low-vol, {high_vol_count} high-vol samples")
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_predictions, val_attention_weights = self.model(X_val, regime_val, return_attention=True)
                    
                    # Check for NaN in validation predictions
                    val_nan_detected = False
                    for key in val_predictions:
                        if torch.isnan(val_predictions[key]).any():
                            print(f"⚠️ Warning: NaN detected in validation {key}. Skipping validation.")
                            val_nan_detected = True
                            break
                    
                    if not val_nan_detected:
                        val_vol_weights = 1.0 + alpha * regime_val[:, -1].float()
                        
                        # Validation losses
                        val_dir_loss_weighted = direction_criterion(val_predictions['direction'], y_dir_val)
                        val_dir_loss_focal = focal_criterion(val_predictions['direction'], y_dir_val)
                        val_dir_loss = (val_dir_loss_weighted * val_vol_weights).mean() + focal_weight * (val_dir_loss_focal * val_vol_weights).mean()
                        
                        val_confidence = val_predictions['confidence'].squeeze()
                        val_confidence_mask = val_confidence > 0.5
                        
                        if val_confidence_mask.sum() > 0:
                            val_ret_loss = return_criterion(val_predictions['return'].squeeze(), y_ret_val.squeeze())
                            val_ret_loss = (val_ret_loss * val_vol_weights * val_confidence_mask).sum() / val_confidence_mask.sum()
                        else:
                            val_ret_loss = return_criterion(val_predictions['return'].squeeze(), y_ret_val.squeeze())
                            val_ret_loss = (val_ret_loss * val_vol_weights).mean()
                        
                        val_pred_returns = val_predictions['return'].squeeze()
                        val_pred_std = val_pred_returns.std() + 1e-8
                        val_sharpe_proxy = val_pred_returns.mean() / val_pred_std
                        
                        val_brier_loss = brier_criterion(val_confidence, y_dir_val)
                        val_brier_loss = (val_brier_loss * val_vol_weights).mean()
                        
                        # Check for NaN in validation losses
                        if not (torch.isnan(val_dir_loss) or torch.isnan(val_ret_loss) or 
                               torch.isnan(val_sharpe_proxy) or torch.isnan(val_brier_loss)):
                            val_total_loss = (direction_weight * val_dir_loss + 
                                            return_weight * val_ret_loss - 
                                            sharpe_weight * val_sharpe_proxy + 
                                            brier_weight * val_brier_loss)
                        else:
                            val_total_loss = float('inf')
                    else:
                        val_total_loss = float('inf')
                
                train_losses.append(total_loss.item())
                val_losses.append(val_total_loss)
                
                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if (epoch + 1) % 25 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss.item():.6f}, Val Loss: {val_total_loss:.6f}")
                    print(f"  Direction Loss: {dir_loss.item():.6f}, Return Loss: {ret_loss.item():.6f}")
                    print(f"  Sharpe Proxy: {sharpe_proxy.item():.6f}, Brier Loss: {brier_loss.item():.6f}")
                    
            except Exception as e:
                print(f"⚠️ Error in epoch {epoch + 1}: {str(e)}")
                continue
        
        # Calculate enhanced metrics
        self._calculate_enhanced_metrics(X_val, regime_val, y_dir_val, y_ret_val)
        return train_losses, val_losses
    
    def _calculate_enhanced_metrics(self, X_val, regime_val, y_dir_val, y_ret_val):
        """Calculate enhanced accuracy metrics with detailed per-class analysis"""
        if self.model is None:
            raise ValueError("Model not initialized. Please train the model first.")
        
        print("🔄 Calculating Enhanced Metrics with Per-Class Analysis...")
        
        # Input validation
        if torch.isnan(X_val).any():
            print("⚠️ Warning: NaN detected in X_val. Replacing with zeros.")
            X_val = torch.nan_to_num(X_val, nan=0.0)
        
        if torch.isnan(regime_val).any():
            print("⚠️ Warning: NaN detected in regime_val. Replacing with zeros.")
            regime_val = torch.nan_to_num(regime_val, nan=0.0)
        
        if torch.isnan(y_ret_val).any():
            print("⚠️ Warning: NaN detected in y_ret_val. Replacing with zeros.")
            y_ret_val = torch.nan_to_num(y_ret_val, nan=0.0)
        
        self.model.eval()
        with torch.no_grad():
            try:
                model_output = self.model(X_val, regime_val, return_attention=True)
                if isinstance(model_output, tuple):
                    predictions = model_output[0]
                    attention_weights = model_output[1] if len(model_output) > 1 else None
                else:
                    predictions = model_output
                    attention_weights = None
            except Exception as e:
                print(f"⚠️ Error during prediction: {str(e)}")
                # Return default metrics
                self.metrics = {
                    'direction_accuracy': 0.0,
                    'down_accuracy': 0.0, 'down_precision': 0.0, 'down_recall': 0.0, 'down_f1': 0.0,
                    'sideways_accuracy': 0.0, 'sideways_precision': 0.0, 'sideways_recall': 0.0, 'sideways_f1': 0.0,
                    'up_accuracy': 0.0, 'up_precision': 0.0, 'up_recall': 0.0, 'up_f1': 0.0,
                    'mse': 1.0, 'mae': 1.0, 'r2': 0.0,
                    'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'cumulative_return': 0.0,
                    'confidence_mean': 0.5, 'volatility_mean': 1.0,
                    'confusion_matrix': np.zeros((3, 3))
                }
                return
        
        # Handle NaN values in predictions with robust extraction
        try:
            for key in predictions:
                if torch.isnan(predictions[key]).any():
                    print(f"⚠️ Warning: NaN values detected in {key} predictions. Replacing with zeros.")
                    predictions[key] = torch.nan_to_num(predictions[key], nan=0.0)
        except Exception as e:
            print(f"⚠️ Error handling NaN in predictions: {e}")
        
        # Direction accuracy and per-class metrics with robust handling
        try:
            direction_pred = torch.argmax(predictions['direction'], dim=1)
            y_dir_val_np = y_dir_val.numpy() if hasattr(y_dir_val, 'numpy') else y_dir_val
            direction_pred_np = direction_pred.detach().cpu().numpy()
            direction_accuracy = accuracy_score(y_dir_val_np, direction_pred_np)
            
            # Per-class precision, recall, F1-score
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_dir_val_np, direction_pred_np, average=None, zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_dir_val_np, direction_pred_np)
        except Exception as e:
            print(f"⚠️ Error calculating direction metrics: {e}")
            direction_accuracy = 0.0
            precision = recall = f1 = np.array([0.0, 0.0, 0.0])
            cm = np.zeros((3, 3))
        
        # Return prediction metrics with robust handling
        try:
            return_pred = predictions['return'].detach().cpu().numpy().flatten()
            return_true = y_ret_val.numpy().flatten() if hasattr(y_ret_val, 'numpy') else y_ret_val.flatten()
        except Exception as e:
            print(f"⚠️ Error extracting return predictions: {e}")
            return_pred = np.zeros(len(y_ret_val))
            return_true = y_ret_val.numpy().flatten() if hasattr(y_ret_val, 'numpy') else y_ret_val.flatten()
        
        # Handle NaN values in return predictions
        if np.isnan(return_pred).any():
            print("⚠️ Warning: NaN values in return predictions. Replacing with zeros.")
            return_pred = np.nan_to_num(return_pred, nan=0.0)
        
        if np.isnan(return_true).any():
            print("⚠️ Warning: NaN values in return true values. Replacing with zeros.")
            return_true = np.nan_to_num(return_true, nan=0.0)
        
        mse = mean_squared_error(return_true, return_pred)
        mae = mean_absolute_error(return_true, return_pred)
        
        # Safe R² calculation
        try:
            r2 = r2_score(return_true, return_pred)
        except:
            r2 = 0.0
        
        # Enhanced financial metrics
        returns_diff = return_pred - return_true
        returns_std = np.std(returns_diff)
        if returns_std > 1e-8:
            sharpe_ratio = np.mean(returns_diff) / returns_std
        else:
            sharpe_ratio = 0.0
        
        # Calculate Max Drawdown
        cumulative_returns = np.cumsum(return_pred)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown)
        
        # Calculate Cumulative Return
        cumulative_return = np.sum(return_pred)
        
        # Direction-specific accuracy (legacy)
        direction_pred_np = direction_pred.numpy()
        y_dir_val_np = y_dir_val.numpy()
        
        down_mask = (y_dir_val_np == 0)
        sideways_mask = (y_dir_val_np == 1)
        up_mask = (y_dir_val_np == 2)
        
        down_accuracy = np.mean((direction_pred_np == 0) & down_mask) / max(np.mean(down_mask), 1e-6)
        sideways_accuracy = np.mean((direction_pred_np == 1) & sideways_mask) / max(np.mean(sideways_mask), 1e-6)
        up_accuracy = np.mean((direction_pred_np == 2) & up_mask) / max(np.mean(up_mask), 1e-6)
        
        # Confidence and volatility with robust handling
        try:
            confidence = predictions['confidence'].detach().cpu().numpy().flatten()
            confidence = np.nan_to_num(confidence, nan=0.5, posinf=1.0, neginf=0.0)
        except Exception as e:
            print(f"⚠️ Error extracting confidence: {e}")
            confidence = np.full(len(return_pred), 0.5)
        
        try:
            volatility = predictions['volatility'].detach().cpu().numpy().flatten()
            volatility = np.nan_to_num(volatility, nan=1.0, posinf=2.0, neginf=0.1)
        except Exception as e:
            print(f"⚠️ Error extracting volatility: {e}")
            volatility = np.full(len(return_pred), 1.0)
        
        # Store enhanced metrics
        self.metrics = {
            'direction_accuracy': direction_accuracy,
            'down_accuracy': down_accuracy, 'down_precision': precision[0], 'down_recall': recall[0], 'down_f1': f1[0],
            'sideways_accuracy': sideways_accuracy, 'sideways_precision': precision[1], 'sideways_recall': recall[1], 'sideways_f1': f1[1],
            'up_accuracy': up_accuracy, 'up_precision': precision[2], 'up_recall': recall[2], 'up_f1': f1[2],
            'mse': mse, 'mae': mae, 'r2': r2,
            'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown, 'cumulative_return': cumulative_return,
            'confidence_mean': confidence.mean(), 'volatility_mean': volatility.mean(),
            'confusion_matrix': cm
        }
        
        print("\n" + "="*80)
        print("🎯 ENHANCED NEUROQUANTIX METRICS WITH PER-CLASS ANALYSIS")
        print("="*80)
        
        # Overall metrics
        print(f"📊 Overall Direction Accuracy: {direction_accuracy:.4f}")
        print(f"📊 Return MSE: {mse:.6f}")
        print(f"📊 Return MAE: {mae:.6f}")
        print(f"📊 Return R² Score: {r2:.4f}")
        print(f"📊 Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"📊 Max Drawdown: {max_drawdown:.4f}")
        print(f"📊 Cumulative Return: {cumulative_return:.4f}")
        print(f"📊 Average Confidence: {confidence.mean():.4f}")
        print(f"📊 Average Volatility: {volatility.mean():.4f}")
        
        # Per-class detailed metrics
        print("\n📈 PER-CLASS PERFORMANCE BREAKDOWN:")
        print("-" * 60)
        class_names = ['Down', 'Sideways', 'Up']
        for i, class_name in enumerate(class_names):
            print(f"\n{class_name} Class (Class {i}):")
            print(f"  Accuracy: {[down_accuracy, sideways_accuracy, up_accuracy][i]:.4f}")
            print(f"  Precision: {precision[i]:.4f}")
            print(f"  Recall: {recall[i]:.4f}")
            print(f"  F1-Score: {f1[i]:.4f}")
            print(f"  Support: {cm[i].sum()} samples")
        
        # Confusion matrix
        print("\n📊 CONFUSION MATRIX:")
        print("-" * 40)
        print("Predicted →")
        print("Actual ↓")
        print("           Down  Sideways  Up")
        for i, class_name in enumerate(class_names):
            print(f"{class_name:9} {cm[i][0]:5d} {cm[i][1]:8d} {cm[i][2]:3d}")
        
        # Enhanced performance assessment
        print("\n📈 ENHANCED PERFORMANCE ASSESSMENT:")
        print("-" * 50)
        
        # Class balance analysis
        class_counts = cm.sum(axis=1)
        total_samples = class_counts.sum()
        class_balance = class_counts / total_samples
        print(f"Class Distribution: Down {class_balance[0]:.1%}, Sideways {class_balance[1]:.1%}, Up {class_balance[2]:.1%}")
        
        # Performance evaluation
        if direction_accuracy > 0.6:
            print("✅ Excellent overall direction prediction performance")
        elif direction_accuracy > 0.5:
            print("✅ Good overall direction prediction performance")
        else:
            print("⚠️ Overall direction prediction needs improvement")
        
        # Per-class performance evaluation
        for i, class_name in enumerate(class_names):
            f1_score = f1[i]
            if f1_score > 0.6:
                print(f"✅ Strong {class_name} prediction performance (F1: {f1_score:.3f})")
            elif f1_score > 0.4:
                print(f"✅ Moderate {class_name} prediction performance (F1: {f1_score:.3f})")
            else:
                print(f"⚠️ {class_name} prediction needs improvement (F1: {f1_score:.3f})")
        
        if r2 > 0.3:
            print("✅ Strong return prediction performance")
        elif r2 > 0.1:
            print("✅ Moderate return prediction performance")
        else:
            print("⚠️ Return prediction needs improvement")
        
        if sharpe_ratio > 0.5:
            print("✅ Excellent risk-adjusted returns")
        elif sharpe_ratio > 0.2:
            print("✅ Good risk-adjusted returns")
        else:
            print("⚠️ Risk-adjusted returns need improvement")
        
        if max_drawdown > -0.1:
            print("✅ Low maximum drawdown")
        elif max_drawdown > -0.2:
            print("✅ Moderate maximum drawdown")
        else:
            print("⚠️ High maximum drawdown")
        
        # Attention analysis with robust handling
        print("\n🧠 ATTENTION ANALYSIS:")
        print("-" * 30)
        try:
            if attention_weights is not None:
                # Analyze attention weights by regime
                low_vol_attn = []
                high_vol_attn = []
                
                for layer_attn in attention_weights:
                    if layer_attn is not None and len(layer_attn.shape) == 4:
                        batch_size, nhead, seq_len, _ = layer_attn.shape
                        for batch_idx in range(min(batch_size, len(regime_val))):
                            try:
                                regime = regime_val[batch_idx, -1].item()
                                avg_attn = layer_attn[batch_idx].mean(dim=0).detach().cpu().numpy()
                                
                                if regime == 0:
                                    low_vol_attn.append(avg_attn)
                                else:
                                    high_vol_attn.append(avg_attn)
                            except Exception as e:
                                print(f"⚠️ Error processing attention for batch {batch_idx}: {e}")
                                continue
                
                if low_vol_attn and high_vol_attn:
                    try:
                        low_entropy = -np.sum(np.array(low_vol_attn).flatten() * np.log(np.array(low_vol_attn).flatten() + 1e-8))
                        high_entropy = -np.sum(np.array(high_vol_attn).flatten() * np.log(np.array(high_vol_attn).flatten() + 1e-8))
                        print(f"Low Volatility Attention Entropy: {low_entropy:.4f}")
                        print(f"High Volatility Attention Entropy: {high_entropy:.4f}")
                        print(f"Attention Focus Difference: {abs(high_entropy - low_entropy):.4f}")
                    except Exception as e:
                        print(f"⚠️ Error calculating attention entropy: {e}")
                else:
                    print("⚠️ Insufficient attention data for analysis")
            else:
                print("⚠️ No attention weights available for analysis")
        except Exception as e:
            print(f"⚠️ Error in attention analysis: {e}")
        
        print("="*80)

def main():
    """Main function to run the optimized NeuroQuantix model"""
    print("🚀 Starting Optimized NeuroQuantix with Real Market Features")
    print("=" * 70)
    
    try:
        # Initialize components
        data_processor = RealMarketDataProcessor()
        analyzer = OptimizedMarketAnalyzer()
        
        # Fetch real market data with larger window
        print("📊 Fetching real market data...")
        df = data_processor.fetch_real_market_data('BTC-USD', hours=5000)  # ~7 months
        print(f"📈 Raw data shape: {df.shape}")
        print(f"📈 Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
        
        # Check for NaN in raw data
        if df.isna().any().any():
            print("⚠️ Warning: NaN detected in raw data. Cleaning...")
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Engineer real features
        print("🔄 Engineering real market features...")
        df_with_features = data_processor.engineer_real_features(df)
        print(f"📈 Data after feature engineering: {df_with_features.shape}")
        
        if len(df_with_features) == 0:
            raise RuntimeError("No data left after feature engineering. Try increasing the 'hours' parameter to fetch more data.")
        
        # Check for NaN after feature engineering
        if df_with_features.isna().any().any():
            print("⚠️ Warning: NaN detected after feature engineering. Cleaning...")
            df_with_features = df_with_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Preprocess data
        print("🔄 Preprocessing optimized data...")
        X, regime_seq, y_direction, y_return, feature_names = data_processor.preprocess_optimized_data(df_with_features, seq_len=24)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"Total sequences: {len(X)}")
        print(f"Features: {len(feature_names)}")
        print(f"Direction classes: {len(np.unique(y_direction))}")
        print(f"Return range: {y_return.min():.4f} to {y_return.max():.4f}")
        
        # Final NaN check before training
        if torch.isnan(X).any():
            print("⚠️ Warning: NaN detected in X before training. Replacing with zeros.")
            X = torch.nan_to_num(X, nan=0.0)
        
        if torch.isnan(regime_seq).any():
            print("⚠️ Warning: NaN detected in regime_seq before training. Replacing with zeros.")
            regime_seq = torch.nan_to_num(regime_seq, nan=0.0)
        
        if torch.isnan(y_return).any():
            print("⚠️ Warning: NaN detected in y_return before training. Replacing with zeros.")
            y_return = torch.nan_to_num(y_return, nan=0.0)
        
        # Train optimized model
        print("🔄 Training Optimized NeuroQuantix Model...")
        train_losses, val_losses = analyzer.train_optimized_model(X, regime_seq, y_direction, y_return, epochs=100, lr=0.0005)
        
        # Visualize results
        print("📊 Generating visualizations...")
        visualize_results(train_losses, val_losses, analyzer.metrics, df_with_features)
        
        print("\n🎉 Training completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
        print("Please check the data source and try again.")
        return

def visualize_results(train_losses, val_losses, metrics, df_with_features):
    """Generate comprehensive visualizations of results"""
    try:
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('NeuroQuantix Optimized Model Results', fontsize=16, fontweight='bold')
        
        # 1. Training Loss
        if train_losses and val_losses:
            axes[0, 0].plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
            axes[0, 0].plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
            axes[0, 0].set_title('Training Progress')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No training data available', ha='center', va='center')
            axes[0, 0].set_title('Training Progress')
        
        # 2. Price Chart with Technical Indicators
        if len(df_with_features) > 500:
            axes[0, 1].plot(df_with_features.index[-500:], df_with_features['Close'].iloc[-500:], 
                            label='BTC Price', color='green', linewidth=2)
            if 'SMA_12' in df_with_features.columns:
                axes[0, 1].plot(df_with_features.index[-500:], df_with_features['SMA_12'].iloc[-500:], 
                                label='SMA 12', color='orange', alpha=0.7)
            if 'EMA_24' in df_with_features.columns:
                axes[0, 1].plot(df_with_features.index[-500:], df_with_features['EMA_24'].iloc[-500:], 
                                label='EMA 24', color='red', alpha=0.7)
            axes[0, 1].set_title('BTC Price with Moving Averages')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Price (USD)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Insufficient price data', ha='center', va='center')
            axes[0, 1].set_title('BTC Price with Moving Averages')
        
        # 3. RSI
        if 'RSI' in df_with_features.columns and len(df_with_features) > 500:
            axes[0, 2].plot(df_with_features.index[-500:], df_with_features['RSI'].iloc[-500:], 
                            color='purple', linewidth=2)
            axes[0, 2].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
            axes[0, 2].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
            axes[0, 2].set_title('RSI Indicator')
            axes[0, 2].set_xlabel('Time')
            axes[0, 2].set_ylabel('RSI')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'RSI data not available', ha='center', va='center')
            axes[0, 2].set_title('RSI Indicator')
        
        # 4. Performance Metrics Bar Chart
        if metrics:
            metric_names = ['Direction Acc', 'R² Score', 'MSE', 'MAE', 'Sharpe']
            metric_values = [
                metrics.get('direction_accuracy', 0.0),
                metrics.get('r2', 0.0),
                metrics.get('mse', 1.0),
                metrics.get('mae', 1.0),
                metrics.get('sharpe_ratio', 0.0)
            ]
            
            # Handle NaN values in metrics
            metric_values = [0.0 if np.isnan(v) else v for v in metric_values]
            
            colors = ['green' if v > 0.5 else 'orange' if v > 0.1 else 'red' for v in metric_values]
            bars = axes[1, 0].bar(metric_names, metric_values, color=colors, alpha=0.7)
            axes[1, 0].set_title('Model Performance Metrics')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{value:.3f}', ha='center', va='bottom')
        else:
            axes[1, 0].text(0.5, 0.5, 'No metrics available', ha='center', va='center')
            axes[1, 0].set_title('Model Performance Metrics')
        
        # 5. Direction Accuracy Breakdown
        if metrics:
            direction_names = ['Down', 'Sideways', 'Up']
            direction_values = [
                metrics.get('down_accuracy', 0.0),
                metrics.get('sideways_accuracy', 0.0),
                metrics.get('up_accuracy', 0.0)
            ]
            
            # Handle NaN values
            direction_values = [0.0 if np.isnan(v) else v for v in direction_values]
            
            axes[1, 1].bar(direction_names, direction_values, 
                           color=['red', 'gray', 'green'], alpha=0.7)
            axes[1, 1].set_title('Direction Accuracy by Class')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(direction_values):
                axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        else:
            axes[1, 1].text(0.5, 0.5, 'No direction metrics available', ha='center', va='center')
            axes[1, 1].set_title('Direction Accuracy by Class')
        
        # 6. Volume and Volatility
        ax6 = axes[1, 2]
        ax6_twin = ax6.twinx()
        
        if len(df_with_features) > 500:
            # Volume
            if 'Volume' in df_with_features.columns:
                ax6.plot(df_with_features.index[-500:], df_with_features['Volume'].iloc[-500:], 
                         color='blue', alpha=0.7, label='Volume')
            ax6.set_xlabel('Time')
            ax6.set_ylabel('Volume', color='blue')
            ax6.tick_params(axis='y', labelcolor='blue')
            
            # Volatility
            if 'Volatility_6' in df_with_features.columns:
                ax6_twin.plot(df_with_features.index[-500:], df_with_features['Volatility_6'].iloc[-500:], 
                              color='red', alpha=0.7, label='Volatility')
            ax6_twin.set_ylabel('Volatility', color='red')
            ax6_twin.tick_params(axis='y', labelcolor='red')
            
            ax6.set_title('Volume and Volatility')
            ax6.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax6.get_legend_handles_labels()
            lines2, labels2 = ax6_twin.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax6.text(0.5, 0.5, 'Insufficient data for volume/volatility', ha='center', va='center')
            ax6.set_title('Volume and Volatility')
        
        plt.tight_layout()
        plt.savefig('NeuroQuantix_Optimized_Results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed results
        if metrics:
            print("\n" + "="*80)
            print("🎯 DETAILED NEUROQUANTIX RESULTS")
            print("="*80)
            print(f"📊 Overall Direction Accuracy: {metrics.get('direction_accuracy', 0.0):.4f}")
            print(f"📊 Down Direction Accuracy: {metrics.get('down_accuracy', 0.0):.4f}")
            print(f"📊 Sideways Direction Accuracy: {metrics.get('sideways_accuracy', 0.0):.4f}")
            print(f"📊 Up Direction Accuracy: {metrics.get('up_accuracy', 0.0):.4f}")
            print(f"📊 Return MSE: {metrics.get('mse', 1.0):.6f}")
            print(f"📊 Return MAE: {metrics.get('mae', 1.0):.6f}")
            print(f"📊 Return R² Score: {metrics.get('r2', 0.0):.4f}")
            print(f"📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.4f}")
            print(f"📊 Max Drawdown: {metrics.get('max_drawdown', 0.0):.4f}")
            print(f"📊 Cumulative Return: {metrics.get('cumulative_return', 0.0):.4f}")
            print(f"📊 Average Confidence: {metrics.get('confidence_mean', 0.5):.4f}")
            print(f"📊 Average Volatility: {metrics.get('volatility_mean', 1.0):.4f}")
            
            # Save results to file
            with open('NeuroQuantix_Optimized_Results.txt', 'w') as f:
                f.write("NEUROQUANTIX OPTIMIZED MODEL RESULTS\n")
                f.write("="*50 + "\n\n")
                f.write(f"Model Performance:\n")
                f.write(f"- Overall Direction Accuracy: {metrics.get('direction_accuracy', 0.0):.4f}\n")
                f.write(f"- R² Score: {metrics.get('r2', 0.0):.4f}\n")
                f.write(f"- MSE: {metrics.get('mse', 1.0):.6f}\n")
                f.write(f"- MAE: {metrics.get('mae', 1.0):.6f}\n")
                f.write(f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.4f}\n")
                f.write(f"- Max Drawdown: {metrics.get('max_drawdown', 0.0):.4f}\n")
                f.write(f"- Cumulative Return: {metrics.get('cumulative_return', 0.0):.4f}\n\n")
                
                f.write(f"Direction Accuracy Breakdown:\n")
                f.write(f"- Down: {metrics.get('down_accuracy', 0.0):.4f}\n")
                f.write(f"- Sideways: {metrics.get('sideways_accuracy', 0.0):.4f}\n")
                f.write(f"- Up: {metrics.get('up_accuracy', 0.0):.4f}\n\n")
                
                f.write(f"Model Architecture:\n")
                f.write(f"- Input Features: 37\n")
                f.write(f"- Hidden Dimension: 256\n")
                f.write(f"- Attention Heads: 16\n")
                f.write(f"- Transformer Layers: 6\n")
                f.write(f"- Multi-scale Convolutions: [3, 7, 15, 31]\n")
            
            print(f"\n💾 Results saved to 'NeuroQuantix_Optimized_Results.txt'")
            print(f"📊 Visualization saved to 'NeuroQuantix_Optimized_Results.png'")
        else:
            print("\n⚠️ No metrics available for detailed results.")
            
    except Exception as e:
        print(f"❌ Error in visualization: {str(e)}")
        print("Skipping visualization generation.")

if __name__ == "__main__":
    main() 