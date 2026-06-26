import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define a simple feedforward neural network for sentiment analysis
class SentimentNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Simulate trading strategy based on sentiment predictions
def trading_strategy(predictions, stock_prices):
    cash = 10000  # Starting cash
    shares = 0    # Starting shares
    for i in range(len(predictions)):
        if predictions[i] == 1:  # Buy signal
            shares += cash / stock_prices[i]
            cash = 0
        elif predictions[i] == 0 and shares > 0:  # Sell signal
            cash += shares * stock_prices[i]
            shares = 0
    # Final portfolio value
    return cash + shares * stock_prices[-1]

if __name__ == '__main__':
    # Dummy data
    news_data = [
        "Stock prices are expected to rise due to strong earnings.",
        "The market is experiencing a downturn.",
        "Positive outlook for the tech industry.",
        "Economic slowdown is causing concerns.",
        "Company reports record profits.",
        "Market volatility is increasing."
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1 for positive sentiment, 0 for negative sentiment
    stock_prices = np.array([100, 102, 105, 103, 110, 108])  # Simulated stock prices

    # Preprocess text data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(news_data).toarray()
    y = np.array(labels)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Model, loss, and optimizer
    input_dim = X_train.shape[1]
    hidden_dim = 16
    output_dim = 2  # Positive or negative sentiment
    model = SentimentNet(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predictions = torch.max(test_outputs, 1)
        accuracy = (predictions == y_test).float().mean()
        print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

    # Simulate trading strategy
    all_outputs = model(torch.tensor(scaler.transform(X), dtype=torch.float32))
    _, all_predictions = torch.max(all_outputs, 1)
    final_portfolio_value = trading_strategy(all_predictions.numpy(), stock_prices)
    print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")