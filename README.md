# News-based Trading Strategies

This repository contains an implementation of the research paper **"News-based Trading Strategies"** by Stefan Feuerriegel and Helmut Prendinger. The paper explores how financial news sentiment can be leveraged to design trading strategies for predicting stock price movements and generating profits. The authors propose approaches based on supervised learning and reinforcement learning to automate trading decisions using textual news data.

## Core Concepts

### Financial Markets and News Sentiment
Financial markets are highly efficient in processing information, including textual news, which often contains valuable sentiment information that influences stock prices. News sentiment analysis involves automatically determining the sentiment (positive, negative, or neutral) of news articles to predict stock returns. 

### Trading Strategies
The paper introduces trading strategies that utilize machine learning models to:
1. Process and analyze financial news text.
2. Predict stock price movements based on news sentiment.
3. Make automated investment decisions using supervised learning and reinforcement learning techniques.

### Key Contributions
1. **Sentiment Analysis for Stock Price Prediction**: The paper demonstrates how analyzing news sentiment can help explain stock price changes.
2. **Automated Decision-Making**: The authors design trading strategies that rely on machine learning models to generate buy/sell signals based on news data.
3. **Reinforcement Learning**: The paper proposes reinforcement learning as a method for learning optimal trading policies directly from data.

## Repository Overview

This repository provides a Python implementation of the trading strategies described in the paper, using the PyTorch framework. The code demonstrates how to preprocess financial news data, extract sentiment scores, and integrate them into trading models.

### Main Features
1. **Text Preprocessing**: Convert raw financial news into a format suitable for analysis. This includes tokenization, stopword removal, and vectorization.
2. **Sentiment Analysis Model**: Utilize supervised learning to train a sentiment analysis model that predicts the sentiment of news articles.
3. **Reinforcement Learning-Based Trading Agent**: Implement a reinforcement learning agent that learns to make trading decisions based on news sentiment and historical stock data.
4. **Backtesting Framework**: Simulate trading strategies using historical data to evaluate performance metrics like profit, loss, and risk.

## Code Structure

```
.
├── data/
│   ├── raw_news.csv        # Raw financial news data
│   ├── stock_prices.csv    # Historical stock price data
├── models/
│   ├── sentiment_model.py  # Supervised sentiment analysis model
│   ├── trading_agent.py    # Reinforcement learning agent
├── utils/
│   ├── data_preprocessing.py  # Utilities for data cleaning and feature extraction
│   ├── evaluation.py          # Functions for backtesting and performance evaluation
├── notebooks/
│   ├── exploratory_analysis.ipynb  # Exploratory data analysis on news and stock data
│   ├── model_training.ipynb        # Training the sentiment analysis model
│   ├── trading_simulation.ipynb    # Simulations of the trading strategies
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── LICENSE                 # License file
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.10 or higher
- Additional dependencies listed in `requirements.txt`

### Installation
1. Clone this repository:
   ```
   git clone https://github.com/your_username/news-based-trading-strategies.git
   cd news-based-trading-strategies
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download or prepare the dataset and place it in the `data/` directory.

### Usage
1. **Data Preprocessing**:
   Run the `data_preprocessing.py` script to clean and preprocess the raw news and stock price data:
   ```
   python utils/data_preprocessing.py
   ```
2. **Train Sentiment Analysis Model**:
   Use the `model_training.ipynb` notebook to train the sentiment analysis model on the preprocessed data.
3. **Run Trading Simulations**:
   Use the `trading_simulation.ipynb` notebook to simulate the trading strategies and evaluate their performance using backtesting.

## Results and Findings

The implementation demonstrates that integrating news sentiment into trading strategies can improve the prediction of stock price movements and enhance profitability. The reinforcement learning agent is particularly effective in learning optimal trading policies in dynamic market environments.

### Example Outputs
- **Sentiment Analysis Model**: Accuracy and F1-score on the validation dataset.
- **Trading Strategy Backtesting**: 
  - Annualized return.
  - Sharpe ratio.
  - Maximum drawdown.

## Future Work
- Extend the sentiment analysis model to incorporate additional features like market trends and macroeconomic indicators.
- Experiment with different reinforcement learning algorithms and reward structures.
- Apply the strategies to other asset classes like cryptocurrencies or commodities.

## Citation
If you use this repository in your research or work, please cite the original paper:
```
@article{feuerriegel2018news,
  title={News-based trading strategies},
  author={Feuerriegel, Stefan and Prendinger, Helmut},
  journal={arXiv preprint arXiv:1807.06824},
  year={2018}
}
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.