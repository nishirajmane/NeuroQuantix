
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
from typing import List, Tuple

# --- 1. Model Architecture ---


class EnhancedInputLayer(nn.Module):
    """
    Processes the initial 37 features into a d_model dimensional space (256).
    Includes Layer Normalization and Dropout for stability.
    """

    def __init__(self, input_features: int, d_model: int):
        super().__init__()
        self.layer = nn.Linear(input_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_features]
        x = self.layer(x)
        x = self.norm(x)
        x = self.activation(x)
        return torch.nan_to_num(x)


class VolatilityEmbedding(nn.Module):
    """
    Embeds the binary volatility regime sequence into a dense vector space.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # 2 possible regimes (0 or 1), so 2 embeddings.
        self.embedding = nn.Embedding(2, embed_dim)

    def forward(self, regime_seq: torch.Tensor) -> torch.Tensor:
        # regime_seq shape: [batch_size, seq_len]
        embedded_regime = self.embedding(regime_seq.long())
        return torch.nan_to_num(embedded_regime)


class TemporalConvResidualBlock(nn.Module):
    """
    A residual block with a 1D temporal convolution for early-stage feature smoothing.
    """

    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding='same')
        self.norm1 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding='same')
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        residual = x
        x = x.permute(0, 2, 1)  # [batch_size, d_model, seq_len] for Conv1d
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, d_model]
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)

        return torch.nan_to_num(residual + x)


class MultiScaleConvolution(nn.Module):
    """
    Applies multiple 1D convolutions with different kernel sizes to capture features
    at various temporal scales.
    """

    def __init__(self, d_model: int, kernel_sizes: List[int]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, k, padding='same') for k in kernel_sizes
        ])
        self.norm = nn.LayerNorm(d_model * len(kernel_sizes))
        self.activation = nn.GELU()
        self.projection = nn.Linear(d_model * len(kernel_sizes), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        x = x.permute(0, 2, 1)  # [batch_size, d_model, seq_len]
        convolved_features = [conv(x) for conv in self.convs]
        # Concat along channel dim
        x_cat = torch.cat(convolved_features, dim=1)
        # [batch_size, seq_len, d_model * num_kernels]
        x_cat = x_cat.permute(0, 2, 1)

        x_cat = self.norm(x_cat)
        x_cat = self.activation(x_cat)
        projected = self.projection(x_cat)
        return torch.nan_to_num(projected)


class RegimeAwareAttention(nn.Module):
    """
    Custom attention mechanism that incorporates volatility regime information.
    The regime embedding creates a bias in the attention scores, guiding the
    model to focus on specific time steps based on the market regime.
    """

    def __init__(self, d_model: int, n_heads: int, d_regime: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.regime_bias_layer = nn.Linear(d_regime, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, regime_embedding: torch.Tensor, mask: torch.Tensor = None):
        # x shape: [batch_size, seq_len, d_model]
        # regime_embedding shape: [batch_size, seq_len, d_regime]
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_layer(x)  # [batch_size, seq_len, 3 * d_model]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads,
                   self.d_head).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads,
                   self.d_head).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads,
                   self.d_head).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)

        # --- Regime-Aware Bias ---
        regime_bias = self.regime_bias_layer(
            regime_embedding)  # [batch_size, seq_len, n_heads]
        regime_bias = regime_bias.permute(0, 2, 1).unsqueeze(
            2)  # [batch_size, n_heads, 1, seq_len]
        scores += regime_bias  # Add bias to attention scores

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(
            batch_size, seq_len, -1)

        output = self.out_layer(context)
        return torch.nan_to_num(output), attention_weights


class RegimeAwareTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_regime: int, d_ff: int = 1024, dropout: float = 0.15):
        super().__init__()
        self.self_attn = RegimeAwareAttention(
            d_model, n_heads, d_regime, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, regime_embedding: torch.Tensor):
        attn_output, attn_weights = self.self_attn(x, regime_embedding)
        x = self.norm1(x + self.dropout1(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return torch.nan_to_num(x), attn_weights


class RegimeAwareTransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_regime: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            RegimeAwareTransformerEncoderLayer(
                d_model, n_heads, d_regime, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, regime_embedding: torch.Tensor):
        all_attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, regime_embedding)
            all_attention_weights.append(attn_weights)
        return x, all_attention_weights


class AdvancedPredictionHeads(nn.Module):
    """
    Four separate prediction heads for the multi-objective task.
    - Direction: 3-class classification (Up, Down, Sideways)
    - Return: Regression of future returns
    - Confidence: Model's confidence in its direction prediction
    - Volatility: Estimation of future volatility
    """

    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)

        # 1. Direction Head (Classification)
        self.direction_head = nn.Linear(d_model, 3)

        # 2. Return Head (Regression)
        self.return_head = nn.Linear(d_model, 1)

        # 3. Confidence Head (Regression, bounded between 0 and 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # 4. Volatility Head (Regression, must be non-negative)
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softplus()  # Ensures output is positive
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: [batch_size, seq_len, d_model]
        # Pool features across the time dimension
        pooled_x = self.pooling_layer(
            x.permute(0, 2, 1)).squeeze(-1)  # [batch_size, d_model]

        direction_logits = self.direction_head(pooled_x)
        predicted_return = self.return_head(pooled_x)
        confidence_score = self.confidence_head(pooled_x)
        predicted_volatility = self.volatility_head(pooled_x)

        return (
            torch.nan_to_num(direction_logits),
            torch.nan_to_num(predicted_return),
            torch.nan_to_num(confidence_score),
            torch.nan_to_num(predicted_volatility)
        )


class NeuroQuantixV2Model(nn.Module):
    """
    The complete NeuroQuantix v2 model, integrating all custom components.
    This model is designed for explainability, with hooks to return attention weights.
    """

    def __init__(self, input_features: int = 37, seq_len: int = 24, d_model: int = 256,
                 d_regime: int = 8, n_heads: int = 16, num_encoder_layers: int = 6,
                 dropout: float = 0.15):
        super().__init__()
        self.input_layer = EnhancedInputLayer(input_features, d_model)
        self.volatility_embedding = VolatilityEmbedding(d_regime)

        # Feature Fusion: Concatenate base features and volatility embeddings
        fused_dim = d_model + d_regime
        self.fusion_projection = nn.Linear(fused_dim, d_model)

        self.temporal_conv = TemporalConvResidualBlock(d_model, kernel_size=5)
        self.multiscale_conv = MultiScaleConvolution(
            d_model, kernel_sizes=[3, 7, 15, 31])

        self.transformer_encoder = RegimeAwareTransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_regime=d_regime,
            d_ff=d_model * 4,
            dropout=dropout
        )

        self.prediction_heads = AdvancedPredictionHeads(d_model, seq_len)

    def forward(self, x: torch.Tensor, regime_seq: torch.Tensor, return_attention: bool = False):
        # x: [batch_size, seq_len, input_features]
        # regime_seq: [batch_size, seq_len]

        # 1. Initial Processing
        x_processed = self.input_layer(x)
        regime_embedded = self.volatility_embedding(regime_seq)

        # 2. Enhanced Feature Fusion
        fused_features = torch.cat([x_processed, regime_embedded], dim=-1)
        fused_features = self.fusion_projection(fused_features)

        # 3. Convolutional Layers
        conv_out = self.temporal_conv(fused_features)
        multiscale_out = self.multiscale_conv(conv_out)

        # 4. Transformer Encoder
        encoder_out, attention_weights = self.transformer_encoder(
            multiscale_out, regime_embedded)

        # 5. Prediction Heads
        direction, returns, confidence, volatility = self.prediction_heads(
            encoder_out)

        if return_attention:
            return direction, returns, confidence, volatility, attention_weights
        return direction, returns, confidence, volatility

# --- 2. Loss Functions ---


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor = None):
        super().__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        if self.weight is not None:
            self.weight = self.weight.to(inputs.device)
        return F.cross_entropy(inputs, targets, weight=self.weight)


class BrierScore(nn.Module):
    """
    Calculates a Brier-like score to train the dedicated confidence head.
    The loss encourages the confidence score to be high when the model's direction
    prediction is correct and low when it is incorrect.
    """
    def forward(self, confidence: torch.Tensor, direction_logits: torch.Tensor, y_direction: torch.Tensor):
        with torch.no_grad():
            predicted_classes = torch.argmax(direction_logits, dim=1)
            is_correct = (predicted_classes == y_direction).float()
        
        # Train confidence to predict the probability of being correct
        return F.mse_loss(confidence.squeeze(), is_correct)

class SharpeProxyLoss(nn.Module):
    """
    A proxy for the Sharpe Ratio, stabilized to prevent exploding gradients.
    It uses tanh to bound predicted returns before calculation.
    """
    def forward(self, predicted_returns: torch.Tensor, epsilon: float = 1e-7):
        # We want to maximize this, so we return its negative
        # Stabilize returns to a [-1, 1] range to prevent explosion
        stabilized_returns = torch.tanh(predicted_returns)
        
        mean_return = torch.mean(stabilized_returns)
        std_return = torch.std(stabilized_returns) + epsilon
        
        return -(mean_return / std_return)

class NeuroQuantixLoss(nn.Module):
    def __init__(self, weights: dict = None):
        super().__init__()
        self.weights = weights or {'dir': 1.0, 'ret': 0.5, 'shp': 0.02, 'br': 0.5}
        self.dir_loss_fn = WeightedCrossEntropyLoss()
        self.ret_loss_fn = nn.HuberLoss()
        self.brier_loss_fn = BrierScore()
        self.sharpe_loss_fn = SharpeProxyLoss()

    def forward(self, outputs, targets):
        dir_logits, pred_ret, conf, _ = outputs
        y_dir, y_ret = targets

        dir_loss = self.dir_loss_fn(dir_logits, y_dir)
        ret_loss = self.ret_loss_fn(pred_ret.squeeze(), y_ret)
        # Pass logits to brier score to determine correctness
        brier_loss = self.brier_loss_fn(conf, dir_logits, y_dir)
        sharpe_loss = self.sharpe_loss_fn(pred_ret)

        total_loss = (self.weights['dir'] * dir_loss +
                      self.weights['ret'] * ret_loss +
                      self.weights['br'] * brier_loss +
                      self.weights['shp'] * sharpe_loss)
        
        return total_loss, {'dir': dir_loss.item(), 'ret': ret_loss.item(), 'brier': brier_loss.item(), 'sharpe': sharpe_loss.item()}

# --- 3. Trainer and Evaluator ---


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(
        ), lr=config['lr'], weight_decay=config['weight_decay'])
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=15, T_mult=2) # Increased T_0
        self.loss_fn = NeuroQuantixLoss(config['loss_weights']).to(self.device)

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            X, regime, y_dir, y_ret = [b.to(self.device) for b in batch]

            self.optimizer.zero_grad()

            outputs = self.model(X, regime)
            loss, _ = self.loss_fn(outputs, (y_dir, y_ret))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        all_pred_rets, all_target_rets = [], []
        all_conf, all_vol = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                X, regime, y_dir, y_ret = [b.to(self.device) for b in batch]

                outputs = self.model(X, regime)
                loss, _ = self.loss_fn(outputs, (y_dir, y_ret))
                total_loss += loss.item()

                dir_logits, pred_ret, conf, vol = outputs
                all_preds.extend(torch.argmax(dir_logits, dim=1).cpu().numpy())
                all_targets.extend(y_dir.cpu().numpy())
                all_pred_rets.extend(pred_ret.squeeze().cpu().numpy())
                all_target_rets.extend(y_ret.cpu().numpy())
                all_conf.extend(conf.squeeze().cpu().numpy())
                all_vol.extend(vol.squeeze().cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, (all_preds, all_targets, all_pred_rets, all_target_rets, all_conf, all_vol)

    def train(self):
        print(f"Starting training on {self.device}...")
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch()
            val_loss, metrics_data = self.validate()
            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{self.config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), 'neuroquantix_v2_best.pth')
                print("Validation loss improved. Model saved.")
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.config['early_stopping_patience']:
                print(
                    f"Early stopping triggered after {self.config['early_stopping_patience']} epochs with no improvement.")
                break

        # Load best model for final evaluation
        print("\nLoading best model for final evaluation...")
        self.model.load_state_dict(torch.load('neuroquantix_v2_best.pth'))
        self.evaluate()

    def evaluate(self):
        print("\n--- Final Evaluation on Validation Set ---")
        _, metrics_data = self.validate()
        preds, targets, pred_rets, target_rets, confs, vols = metrics_data

        # Direction Metrics
        print("\n--- Direction Performance ---")
        accuracy = accuracy_score(targets, preds)
        print(f"Direction Accuracy: {accuracy:.4f}")
        print(classification_report(targets, preds,
              target_names=['Down', 'Sideways', 'Up'], zero_division=0))

        # Return Metrics
        print("\n--- Return Performance ---")
        print(f"MSE: {mean_squared_error(target_rets, pred_rets):.6f}")
        print(f"MAE: {mean_absolute_error(target_rets, pred_rets):.6f}")
        print(f"R² Score: {r2_score(target_rets, pred_rets):.4f}")

        # Financial & Model Metrics
        print("\n--- Financial & Model Metrics ---")
        pred_rets_series = pd.Series(pred_rets)
        target_rets_series = pd.Series(target_rets)
        
        # Calculate metrics on target returns for a realistic baseline
        sharpe_ratio_real = (target_rets_series.mean() / target_rets_series.std()) * np.sqrt(252) if target_rets_series.std() != 0 else 0
        
        # Calculate financial metrics based on a simple strategy: long if predicted up, short if predicted down
        strategy_returns = pd.Series(np.zeros(len(preds)))
        strategy_returns[pd.Series(preds) == 2] = target_rets_series[pd.Series(preds) == 2] # Go long
        strategy_returns[pd.Series(preds) == 0] = -target_rets_series[pd.Series(preds) == 0] # Go short

        cumulative_return = (1 + strategy_returns).cumprod() - 1
        sharpe_ratio_strategy = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() != 0 else 0
        
        rolling_max = (1 + cumulative_return).cummax()
        drawdown = (1 + cumulative_return) / rolling_max - 1.0
        max_drawdown = drawdown.min()

        print(f"Annualized Sharpe Ratio (Strategy): {sharpe_ratio_strategy:.4f}")
        print(f"Annualized Sharpe Ratio (Buy & Hold): {sharpe_ratio_real:.4f}")
        print(f"Max Drawdown (Strategy): {max_drawdown:.4f}")
        print(f"Total Cumulative Return (Strategy): {cumulative_return.iloc[-1]:.4f}")
        print(f"Average Predicted Confidence: {np.mean(confs):.4f}")
        print(f"Average Predicted Volatility: {np.mean(vols):.4f}")

# --- Main Execution ---


if __name__ == '__main__':
    # Configuration
    config = {
        'lr': 1e-4, # Reduced learning rate
        'weight_decay': 0.01,
        'epochs': 100,  # Will be stopped early if no improvement
        'early_stopping_patience': 15, # Increased patience
        'loss_weights': {'dir': 1.0, 'ret': 0.5, 'shp': 0.02, 'br': 0.5} # Adjusted weights
    }

    # --- Dummy Data Generation (Replace with your DataLoader) ---
    # This part demonstrates the pipeline. In a real scenario, you would
    # use a torch.utils.data.Dataset and DataLoader to feed real data.
    BATCH_SIZE = 32
    SEQ_LEN = 24
    INPUT_FEATURES = 37

    # Create dummy data tensors
    dummy_X = torch.randn(BATCH_SIZE * 10, SEQ_LEN, INPUT_FEATURES)
    dummy_regime = torch.randint(0, 2, (BATCH_SIZE * 10, SEQ_LEN))
    dummy_y_dir = torch.randint(0, 3, (BATCH_SIZE * 10,))
    dummy_y_ret = torch.randn(BATCH_SIZE * 10,) * 0.02 # Slightly higher variance

    # Create a TensorDataset and DataLoader
    dataset = torch.utils.data.TensorDataset(
        dummy_X, dummy_regime, dummy_y_dir, dummy_y_ret)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE)
    # --- End of Dummy Data Section ---

    # Initialize Model
    model = NeuroQuantixV2Model(
        input_features=INPUT_FEATURES,
        seq_len=SEQ_LEN,
        d_model=256,
        d_regime=8,
        n_heads=16,
        num_encoder_layers=6,
        dropout=0.15
    )

    # Initialize and run Trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

    # Example of running inference and getting attention weights
    print("\n--- Example Inference with Attention Weights ---")
    model.eval()
    with torch.no_grad():
        # Use a batch from the validation loader for a consistent example
        sample_x, sample_regime, _, _ = next(iter(val_loader))
        sample_x = sample_x.to(trainer.device)
        sample_regime = sample_regime.to(trainer.device)

        outputs = model(sample_x, sample_regime, return_attention=True)
        dir_logits, pred_ret, conf, vol, attention_weights = outputs

        print(f"Inference - Direction Logits Shape: {dir_logits.shape}")
        print(
            f"Inference - Returned Attention Layers: {len(attention_weights)}")
        print(
            f"Inference - Shape of Attention from one layer: {attention_weights[0].shape}")
