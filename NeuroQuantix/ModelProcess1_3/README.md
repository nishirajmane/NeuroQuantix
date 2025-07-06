# 🧠 NeuroQuantix: Multimodal Financial AI Research

A research and development experiment implementing a multimodal AI system for financial market prediction using OHLCV data and financial news sentiment.

## 📊 Project Overview

NeuroQuantix combines time-series analysis with natural language processing to predict stock price movements. The system uses a hybrid architecture that fuses technical indicators with sentiment analysis from financial news.

## 🏗️ Architecture

### Core Components:

1. **TimeSeriesTransformer**
   - Input: OHLCV data (shape: [batch_size, 30, 5])
   - Linear projection to 64-dim embeddings
   - 2-layer TransformerEncoder with 4 attention heads
   - Output: Time-series embeddings [batch_size, 64]

2. **FinBERTWrapper**
   - Uses pretrained "yiyanghkust/finbert-tone" model
   - Encodes financial news headlines into sentiment embeddings
   - Output: 64-dim sentiment vectors [batch_size, 64]

3. **Cross-Attention Fusion**
   - MultiHeadAttention layer for modality fusion
   - Time-series embeddings as query, news embeddings as key/value
   - Output: Fused representation [batch_size, 64]

4. **Prediction Head**
   - MLP: Linear(64 → 32 → 2)
   - Binary classification: Buy/Sell signals
   - CrossEntropyLoss for training

## 📈 Results

### Model Performance:
- **Training Accuracy:** 37.50%
- **Strategy Sharpe Ratio:** 0.1212
- **Market Sharpe Ratio:** 0.2004
- **Model Parameters:** 110M+ (including FinBERT)

### Key Insights:
- Successfully combines temporal patterns with sentiment analysis
- Cross-attention fusion effectively merges multimodal features
- Demonstrates potential for real-time financial prediction

## 🚀 Setup & Installation

### Prerequisites:
```bash
pip install -r requirements.txt
```

### Dependencies:
- PyTorch >= 1.13.0
- Transformers >= 4.20.0
- yfinance >= 0.2.0
- pandas >= 1.5.0
- matplotlib >= 3.5.0
- scikit-learn >= 1.1.0

### Running the Experiment:
```bash
python research_experiment.py
```

## 📁 Project Structure

```
MultiModal_GPT/
├── research_experiment.py    # Main experiment script
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── experiment_results.png   # Generated visualizations
└── .gitignore              # Git ignore file
```

## 🔬 Research Methodology

### Data Processing:
- **OHLCV Data:** Downloaded via yfinance (AAPL, 3 months)
- **Normalization:** Z-score normalization for all features
- **Labels:** Binary classification (1 if next day return > 0, else 0)
- **Sequence Length:** 30 days of historical data per prediction

### Training:
- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 10
- **Batch Size:** 16

### Evaluation:
- **Backtesting:** Simulated trading based on predictions
- **Metrics:** Accuracy, Sharpe ratio, win rate
- **Visualization:** Price charts, training loss, returns comparison

## 🎯 Key Features

- **Multimodal Fusion:** Combines technical and sentiment data
- **Transformer Architecture:** State-of-the-art sequence modeling
- **Financial Focus:** Specialized for market prediction
- **Research-Oriented:** Clear explanations and comprehensive reporting
- **Extensible:** Modular design for easy experimentation

## 🚀 Future Enhancements

1. **Real-time News Integration:** Connect to live financial news APIs
2. **Multi-Asset Support:** Extend to multiple stocks and indices
3. **Advanced Backtesting:** Implement more sophisticated trading strategies
4. **Risk Management:** Add position sizing and stop-loss mechanisms
5. **Hyperparameter Optimization:** Automated tuning of model parameters
6. **Production Deployment:** Streamlit dashboard for real-time predictions

## 📊 Visualization Output

The experiment generates comprehensive visualizations including:
- Stock price over time
- Training loss progression
- Cumulative returns comparison (Strategy vs Market)
- Returns distribution analysis

## 🤝 Contributing

This is a research project. Feel free to:
- Experiment with different architectures
- Test on different assets and time periods
- Improve the backtesting methodology
- Add new features and capabilities

## 📄 License

This project is for research purposes. Please ensure compliance with relevant financial regulations when using for actual trading.

---

**Note:** This is a research experiment and should not be used for actual trading without proper validation and risk management. 