from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

def get_stock_data(ticker, days_history=365, days_recent=7):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days_history)
    df_daily = yf.download(ticker, start=start, end=end, interval='1d')
    df_minute = yf.download(ticker, start=end - datetime.timedelta(days=days_recent), end=end, interval='1m')
    return df_daily, df_minute

def calculate_technical_indicators(df):
    # RSI
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # MACD
    def calculate_macd(data, fast=12, slow=26, signal=9):
        fast_ema = data.ewm(span=fast, min_periods=1).mean()
        slow_ema = data.ewm(span=slow, min_periods=1).mean()
        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(span=signal, min_periods=1).mean()
        return macd, macd_signal

    # Bollinger Bands
    def calculate_bollinger_bands(data, window=20, num_std=2):
        sma = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)
        return upper_band, lower_band

    # EMA
    def calculate_ema(data, window=14):
        return data.ewm(span=window, min_periods=1).mean()

    # SMA
    def calculate_sma(data, window=14):
        return data.rolling(window=window).mean()

    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['EMA'] = calculate_ema(df['Close'])
    df['SMA'] = calculate_sma(df['Close'])
    return df.dropna()

def create_sequences(data, seq_length=60):
    X, y_open, y_close = [], [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y_open.append(data[i][0])  # Open
        y_close.append(data[i][3])  # Close
    return np.array(X), np.array(y_open), np.array(y_close)

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(2)  # Predict open and close
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def create_price_chart(df):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], color='#5fba7d', linewidth=2)
    ax.set_title('Stock Price History', color='white')
    ax.set_xlabel('Date', color='white')
    ax.set_ylabel('Price (USD)', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(color='#2A3459')
    return plot_to_base64(fig)

def create_rsi_chart(df):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df['RSI'], color='#ff6b6b', linewidth=1)
    ax.axhline(70, color='#ff0000', linestyle='--', linewidth=0.7)
    ax.axhline(30, color='#00ff00', linestyle='--', linewidth=0.7)
    ax.set_title('RSI', color='white')
    ax.set_xlabel('Date', color='white')
    ax.set_ylabel('RSI', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(color='#2A3459')
    return plot_to_base64(fig)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form.get('ticker', 'TATAMOTORS.NS')
        
        try:
            # Get and prepare data
            df_daily, df_minute = get_stock_data(ticker)
            df = calculate_technical_indicators(df_daily.copy())
            
            # Scale data
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'EMA', 'SMA']
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[features])
            
            # Create sequences
            X, y_open, y_close = create_sequences(scaled_data)
            
            # Build and train model
            model = build_model((X.shape[1], X.shape[2]))
            split = int(0.9 * len(X))
            model.fit(X[:split], np.column_stack((y_open[:split], y_close[:split])), 
                     epochs=20, batch_size=32, verbose=0)
            
            # Make prediction
            last_seq = X[-1:]
            pred = model.predict(last_seq)[0]
            inv_pred = scaler.inverse_transform([[pred[0], 0, 0, pred[1], 0, 0, 0, 0, 0, 0, 0, 0]])
            
            pred_open = inv_pred[0][0]
            pred_close = inv_pred[0][3]
            current_close = df['Close'].iloc[-1].item()
            trend = "UP" if pred_close > current_close else "DOWN"
            change_percent = ((pred_close - current_close) / current_close) * 100
            
            # Create charts
            price_chart = create_price_chart(df)
            rsi_chart = create_rsi_chart(df)
            
            return render_template('index.html', 
                                 ticker=ticker,
                                 current_close=f"{current_close:.2f}",
                                 pred_open=f"{pred_open:.2f}",
                                 pred_close=f"{pred_close:.2f}",
                                 trend=trend,
                                 change_percent=f"{change_percent:.2f}",
                                 price_chart=price_chart,
                                 rsi_chart=rsi_chart,
                                 success=True)
        
        except Exception as e:
            return render_template('index.html', error=str(e), ticker=ticker)
    
    return render_template('index.html', ticker='TATAMOTORS.NS')

if __name__ == '__main__':
    app.run(debug=True)