🔹 1. Objective
Build an LSTM (Long Short-Term Memory) deep learning model that:

Uses historical stock data (daily and minute)

Computes custom technical indicators

Predicts tomorrow’s open and close prices

Predicts the trend direction (UP/DOWN)

Displays the current price based on live minute-level data

🔹 2. Data Collection
Use the yfinance library to fetch stock data:

1-year daily data (interval='1d') → for model training

Last 30 days of minute-level data (interval='1m') → for real-time price and trends

🔹 3. Feature Engineering
Generate 5 custom technical indicators manually, for example:

Simple Moving Average (SMA)

Exponential Moving Average (EMA)

Relative Strength Index (RSI)

Momentum

Volatility (Rolling Std Dev)

These indicators help the model capture:

Trends

Momentum

Overbought/oversold conditions

Price volatility

🔹 4. Data Preprocessing
Drop missing values

Normalize/scale data (e.g., using MinMaxScaler)

Create sliding windows of time series data suitable for LSTM:

Input: last n timesteps (e.g., 60 days)

Output: next day’s open and close price

🔹 5. Model Building (LSTM)
Use a sequential LSTM model:

One or more LSTM layers

Dense layer for predicting two outputs: open and close

Compile using mean_squared_error loss and adam optimizer

Train the model on the prepared time series data

🔹 6. Prediction
Predict tomorrow’s open and close prices using the last n timesteps

Compare the predicted close with the latest close to determine trend:

If predicted_close > last_close → UP

Else → DOWN

🔹 7. Live Price Retrieval
Use minute-level data to show the current/latest price

Extract the last close from 1-minute interval data

🔹 8. Output Display
Print or render:

Predicted Open Price

Predicted Close Price

Trend Direction (UP/DOWN)

Current Market Price



![{96C9B44D-E444-439C-9E2E-E98F406E3988}](https://github.com/user-attachments/assets/2cddb140-0f24-4a32-a1f9-e3a4472a58e2)
![{9CC55C2C-7904-48E1-84F7-556B773604C9}](https://github.com/user-attachments/assets/b9fdbe20-1172-40ef-9f6b-85c95e6cf913)
![{5B46A3A0-55DC-470C-B233-8A5C88290734}](https://github.com/user-attachments/assets/2748a22f-63fa-43f0-9994-4c6a5c1f161c)
![{BD03CC2E-8AD9-41A9-8220-3A974BD98E2D}](https://github.com/user-attachments/assets/61eb8aad-a819-4150-8f62-0c5a2db2e213)
