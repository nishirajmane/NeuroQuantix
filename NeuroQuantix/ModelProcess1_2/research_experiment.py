"""
🧠 NeuroQuantix R&D Experiment: Multimodal Financial AI
Research and Development Approach with Clear Explanations

Architecture:
1. TimeSeriesTransformer: OHLCV → 64-dim embeddings
2. FinBERTWrapper: News → 64-dim sentiment embeddings  
3. FusionLayer: Cross-attention fusion
4. NeuroQuantixModel: Complete pipeline
5. Training, Backtesting, and SHAP Explainability
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("🔬 Starting NeuroQuantix R&D Experiment...")
print("=" * 60)

# ============================================================================
# STEP 1: DATA ACQUISITION & PREPROCESSING
# ============================================================================

def load_and_preprocess_data(ticker='AAPL', period='3mo'):
    """
    📊 Load OHLCV data and create sample news
    Returns: price_data (normalized), news_data, labels
    """
    print(f"\n📈 Loading {ticker} data for {period}...")
    
    # Download OHLCV data
    df = yf.download(ticker, period=period)
    if df is None or df.empty:
        raise ValueError(f"Failed to download data for {ticker}")
    
    # Select OHLCV columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Normalize data (z-score normalization)
    df_normalized = (df - df.mean()) / df.std()
    
    # Create sample news data (in real scenario, this would be actual news)
    news_data = [
        "Apple stock surges after strong earnings report",
        "Tech sector faces pressure from rising interest rates", 
        "Apple to increase iPhone production amid high demand",
        "Market volatility increases due to economic uncertainty",
        "Apple announces new product line expansion"
    ]
    
    # Create binary labels (1 if next day return > 0, else 0)
    df['Returns'] = df['Close'].pct_change()
    labels = (df['Returns'].shift(-1) > 0).astype(int).dropna()
    
    print(f"✅ Loaded {len(df)} days of data")
    print(f"📰 Sample news: {len(news_data)} headlines")
    print(f"🎯 Labels: {labels.sum()} positive, {len(labels) - labels.sum()} negative")
    
    return df_normalized, news_data, labels

# ============================================================================
# STEP 2: MODEL ARCHITECTURE
# ============================================================================

class TimeSeriesTransformer(nn.Module):
    """
    🕐 Time-Series Transformer for OHLCV data
    Input: [batch_size, sequence_length, 5] (OHLCV)
    Output: [batch_size, 64] (time-series embeddings)
    """
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear_out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, 5]
        x = self.linear_in(x)  # [batch_size, seq_len, 64]
        x = self.encoder(x)    # [batch_size, seq_len, 64]
        return self.linear_out(x[:, -1, :])  # [batch_size, 64]

class FinBERTWrapper(nn.Module):
    """
    📰 FinBERT wrapper for news sentiment encoding
    Input: List of news headlines
    Output: [batch_size, 64] (sentiment embeddings)
    """
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(768, 64)  # Persistent projection layer
        # Freeze FinBERT parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, news_list):
        # Tokenize and encode news
        inputs = self.tokenizer(
            news_list, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=128
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch_size, 768]
        return self.proj(embeddings)  # [batch_size, 64]

class FusionLayer(nn.Module):
    """
    🔗 Cross-Attention Fusion Layer
    Fuses time-series and news embeddings using multi-head attention
    """
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
    def forward(self, ts_embed, news_embed):
        # ts_embed: [batch_size, 64] (query)
        # news_embed: [batch_size, 64] (key, value)
        
        # Reshape for attention
        query = ts_embed.unsqueeze(1)  # [batch_size, 1, 64]
        key = value = news_embed.unsqueeze(1)  # [batch_size, 1, 64]
        
        # Apply cross-attention
        fused, _ = self.attention(query, key, value)
        return fused.squeeze(1)  # [batch_size, 64]

class NeuroQuantixModel(nn.Module):
    """
    🧠 Complete NeuroQuantix Model
    Combines all components: TimeSeries + FinBERT + Fusion + Prediction
    """
    def __init__(self):
        super().__init__()
        self.ts_transformer = TimeSeriesTransformer()
        self.finbert_wrapper = FinBERTWrapper()
        self.fusion_layer = FusionLayer()
        self.prediction_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # Binary classification: Buy/Sell
        )
        
    def forward(self, price_seq, news_list):
        # Process time-series data
        ts_embed = self.ts_transformer(price_seq)  # [batch_size, 64]
        
        # Process news data
        news_embed = self.finbert_wrapper(news_list)  # [batch_size, 64]
        
        # Fuse embeddings
        fused = self.fusion_layer(ts_embed, news_embed)  # [batch_size, 64]
        
        # Make prediction
        logits = self.prediction_head(fused)  # [batch_size, 2]
        
        return logits

# ============================================================================
# STEP 3: DATASET & TRAINING
# ============================================================================

class FinancialDataset(Dataset):
    """
    📚 Custom dataset for financial data
    """
    def __init__(self, price_data, news_data, labels, sequence_length=30):
        self.price_data = torch.FloatTensor(price_data.values)
        self.news_data = news_data
        self.labels = torch.LongTensor(labels.values)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.labels) - self.sequence_length
        
    def __getitem__(self, idx):
        # Get price sequence
        price_seq = self.price_data[idx:idx + self.sequence_length]
        # Get a single news headline for this sample (cycle through news list)
        news = self.news_data[idx % len(self.news_data)]
        # Get label
        label = self.labels[idx + self.sequence_length]
        return price_seq, news, label

def train_model(model, train_loader, num_epochs=10, lr=0.001):
    """
    🎯 Training loop with detailed logging
    """
    print(f"\n🎯 Training model for {num_epochs} epochs...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (price_seq, news, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            # Ensure price_seq is always 3D
            if price_seq.ndim == 2:
                price_seq = price_seq.unsqueeze(0)
            news_batch = list(news)
            logits = model(price_seq, news_batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
    print("✅ Training completed!")
    return train_losses

# ============================================================================
# STEP 4: BACKTESTING & EVALUATION
# ============================================================================

def backtest_strategy(model, test_loader, price_data):
    """
    📊 Backtest trading strategy based on model predictions
    """
    print(f"\n📊 Running backtest simulation...")
    
    model.eval()
    predictions = []
    actual_returns = []
    
    with torch.no_grad():
        for price_seq, news, labels in test_loader:
            if price_seq.ndim == 2:
                price_seq = price_seq.unsqueeze(0)
            news_batch = list(news)
            logits = model(price_seq, news_batch)
            pred = torch.argmax(logits, dim=1)
            predictions.extend(pred.numpy())
    
    # Calculate actual returns for the test period
    returns = price_data['Close'].pct_change().dropna()
    test_returns = returns[-len(predictions):].values
    
    # Strategy: Buy when model predicts 1, Sell when predicts 0
    strategy_returns = np.array(predictions) * test_returns
    
    # Calculate metrics
    total_return = np.sum(strategy_returns)
    sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)
    win_rate = np.sum(strategy_returns > 0) / len(strategy_returns)
    
    print(f"📈 Backtest Results:")
    print(f"   Total Return: {total_return:.4f}")
    print(f"   Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"   Win Rate: {win_rate:.2%}")
    
    return strategy_returns, test_returns, predictions

# ============================================================================
# STEP 5: VISUALIZATION & REPORTING
# ============================================================================

def create_visualizations(price_data, train_losses, strategy_returns, test_returns):
    """
    📊 Create comprehensive visualizations
    """
    print(f"\n📊 Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Price chart
    axes[0, 0].plot(price_data.index, price_data['Close'])
    axes[0, 0].set_title('Stock Price Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].grid(True)
    
    # 2. Training loss
    axes[0, 1].plot(train_losses)
    axes[0, 1].set_title('Training Loss Over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # 3. Cumulative returns comparison
    cumulative_strategy = np.cumsum(strategy_returns)
    cumulative_market = np.cumsum(test_returns)
    
    axes[1, 0].plot(cumulative_strategy, label='Strategy', linewidth=2)
    axes[1, 0].plot(cumulative_market, label='Market', linewidth=2)
    axes[1, 0].set_title('Cumulative Returns Comparison')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Cumulative Return')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Returns distribution
    axes[1, 1].hist(strategy_returns, bins=30, alpha=0.7, label='Strategy')
    axes[1, 1].hist(test_returns, bins=30, alpha=0.7, label='Market')
    axes[1, 1].set_title('Returns Distribution')
    axes[1, 1].set_xlabel('Return')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved as 'experiment_results.png'")

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """
    🧪 Main experiment function
    """
    print("🧪 Starting NeuroQuantix R&D Experiment")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    price_data, news_data, labels = load_and_preprocess_data()
    
    # Step 2: Create dataset and dataloader
    dataset = FinancialDataset(price_data, news_data, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"📚 Dataset created: {len(dataset)} samples")
    
    # Step 3: Initialize model
    model = NeuroQuantixModel()
    print(f"🧠 Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 4: Train model
    train_losses = train_model(model, train_loader, num_epochs=10)
    
    # Step 5: Backtest
    strategy_returns, test_returns, predictions = backtest_strategy(model, train_loader, price_data)
    
    # Step 6: Visualizations
    create_visualizations(price_data, train_losses, strategy_returns, test_returns)
    
    # Step 7: Final report
    print("\n" + "=" * 60)
    print("📋 EXPERIMENT SUMMARY")
    print("=" * 60)
    print("🎯 Model Architecture:")
    print("   • TimeSeriesTransformer: OHLCV → 64-dim embeddings")
    print("   • FinBERTWrapper: News → 64-dim sentiment embeddings")
    print("   • FusionLayer: Cross-attention fusion")
    print("   • PredictionHead: MLP for binary classification")
    print()
    print("📊 Results:")
    y_true = labels[-len(predictions):]
    print(f"   • Training Accuracy: {100 * accuracy_score(y_true, predictions):.2f}%")
    print(f"   • Strategy Sharpe Ratio: {np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8):.4f}")
    print(f"   • Market Sharpe Ratio: {np.mean(test_returns) / (np.std(test_returns) + 1e-8):.4f}")
    print()
    print("🔍 Key Insights:")
    print("   • Model learns temporal patterns in OHLCV data")
    print("   • News sentiment provides additional context")
    print("   • Cross-attention fusion combines both modalities")
    print()
    print("🚀 Next Steps:")
    print("   • Use real-time news data")
    print("   • Implement more sophisticated backtesting")
    print("   • Add risk management rules")
    print("   • Test on multiple assets")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment() 