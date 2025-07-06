# Advanced NeuroQuantix - Market State Encoder

## 🧠 Next-Generation Financial Market Analysis with Multi-Scale Architecture

Advanced NeuroQuantix is a cutting-edge market state encoder that combines multiple advanced techniques for comprehensive financial market analysis. The system features a sophisticated multi-scale architecture with state-of-the-art transformer components.

## 🏗️ Architecture Overview

### 1. Input Layer
- **25+ Market Features**: Comprehensive feature engineering
- **Core OHLCV**: Open, High, Low, Close, Volume
- **Market Microstructure**: Spread, Order Depth, Implied Volatility
- **Contextual Zones**: Demand Zone, Supply Zone
- **Technical Indicators**: RSI, MACD, EMA, Bollinger Bands
- **Macro Indicators**: Interest Rates, Inflation, Market Sentiment

### 2. Multi-Scale Convolution
- **Kernel Sizes**: [3, 7, 15] for different temporal scales
- **Feature Extraction**: Local and trend dependencies
- **Signal Denoising**: Advanced filtering and smoothing
- **Receptive Fields**: Multiple scales for comprehensive analysis

### 3. PatchTST Encoder
- **Time-Series Transformer**: Inspired by Vision Transformers
- **Fixed-Length Patches**: Temporal pattern preservation
- **Gated Residuals**: Advanced regularization
- **Positional Embeddings**: Temporal context awareness
- **Masked Attention**: Enhanced pattern recognition

### 4. Cross-Time Attention
- **Temporal Fusion**: Historical pattern emphasis
- **Regime Detection**: Volatility clustering identification
- **Interpretability**: Clear attention weight visualization
- **Pattern Recognition**: Market regime switching detection

### 5. Multi-Task Prediction Heads
- **Direction Classifier**: Up/Down movement prediction
- **Return Regressor**: Percentage movement forecasting
- **Confidence Estimator**: Prediction reliability scoring
- **Volatility Forecaster**: Future volatility prediction

## 🚀 Features

- **Advanced Architecture**: Multi-scale convolution + transformer
- **Comprehensive Features**: 25+ engineered market indicators
- **Multi-Task Learning**: Direction, return, confidence, volatility
- **Regime Detection**: Market state classification
- **High Accuracy**: Optimized for maximum prediction performance
- **Beautiful Visualizations**: 16-panel advanced analysis dashboard
- **Real-Time Processing**: Live market data integration

## 📊 Model Specifications

```
Input Features: 25+
Model Dimension: 128
Attention Heads: 8
Transformer Layers: 4
Patch Size: 4
Kernel Sizes: [3, 7, 15]
Sequence Length: 24 hours
```

## 🛠️ Installation

1. Navigate to the advanced directory:
```bash
cd NeuroQuantix_Advanced
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the advanced analysis:
```bash
python advanced_market_encoder.py
```

## 📈 Performance Metrics

- **Direction Accuracy**: Binary classification performance
- **Return MSE/MAE**: Regression prediction accuracy
- **R² Score**: Overall model fit quality
- **Confidence Score**: Prediction reliability
- **Volatility Forecast**: Risk assessment
- **Regime Classification**: Market state identification

## 🎯 Output

The system generates:
- **Real-time predictions**: Direction, return, confidence, volatility
- **Market regime classification**: Low/Medium/High volatility states
- **Attention weight analysis**: Interpretable pattern recognition
- **Comprehensive visualizations**: 16-panel analysis dashboard
- **Detailed results**: Complete analysis in text format

## 📊 Visualization Dashboard

The advanced system creates a comprehensive 16-panel visualization including:
1. Direction prediction accuracy
2. Return prediction scatter plot
3. Confidence distribution
4. Volatility forecast
5. Cross-time attention weights
6. Market regime probabilities
7. Embeddings PCA projection
8. Performance metrics
9. Return distribution comparison
10. Cumulative returns
11. Feature importance ranking
12. Model architecture summary
13. Training loss curves
14. Multi-scale convolution output
15. Patch embeddings visualization
16. Final prediction summary

## 🔬 Technical Details

### Advanced Features Used:
- **Price Data**: OHLCV with advanced derivatives
- **Technical Indicators**: RSI, MACD, EMA, Bollinger Bands
- **Market Microstructure**: Spread, order depth, implied volatility
- **Macro Indicators**: Interest rates, inflation, sentiment
- **Volatility Measures**: Multiple timeframes and regimes
- **Volume Analysis**: Advanced volume-based indicators

### Model Architecture:
- **Input Dimension**: 25+ features
- **Model Dimension**: 128
- **Attention Heads**: 8
- **Transformer Layers**: 4
- **Patch Size**: 4
- **Kernel Sizes**: [3, 7, 15]
- **Activation**: GELU
- **Optimizer**: AdamW with cosine annealing

## 🎉 Results

The Advanced NeuroQuantix system achieves:
- **High Direction Accuracy** for market movement prediction
- **Robust Return Forecasting** with confidence intervals
- **Accurate Volatility Prediction** for risk management
- **Effective Regime Detection** for market state classification
- **Real-time Processing** capabilities with live data

## 🔮 Future Enhancements

- **Multi-Asset Support**: Cross-asset correlation analysis
- **Real-time Streaming**: Live data processing pipeline
- **Advanced Risk Metrics**: VaR, CVaR, stress testing
- **Portfolio Optimization**: Multi-asset allocation
- **API Integration**: RESTful API for external access
- **Cloud Deployment**: Scalable cloud infrastructure

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

---

**Advanced NeuroQuantix** - Transforming Market Analysis with AI 🚀 