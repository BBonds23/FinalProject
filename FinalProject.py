import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model  # Import the Keras load_model function
from alpha_vantage.timeseries import TimeSeries
import yfinance as yf
import datetime
import diskcache as dc

# --- Configuration ---
API_KEY = "9EOUHM0E1T9OK4U3"  # Replace with your Alpha Vantage API key
DEFAULT_SYMBOL = 'AAPL'
TIME_STEPS = 60  # Lookback period
FUTURE_STEPS = 30  # Shorter forecast
EPOCHS = 20
BATCH_SIZE = 32
RECENT_WEIGHT = 0.3  # How much should recent volatility adjust the forecast

# Caching setup
cache = dc.Cache('/path/to/cache/directory')

# Create Dash app instance
app = dash.Dash(__name__)

# --- Fetch Data with Fallback ---
def fetch_stock_data(symbol):
    # Check cache first
    if symbol in cache:
        print(f"Data for {symbol} found in cache.")
        return cache[symbol]

    ts = TimeSeries(key=API_KEY, output_format='pandas')
    try:
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        data = data.sort_index()
        cache[symbol] = data  # Store in cache
        return data
    except Exception as e:
        print(f"Alpha Vantage failed: {e}")
        try:
            yf_data = yf.download(symbol, period="max", progress=False)
            if yf_data is not None and not yf_data.empty:
                # Rename the columns to a more readable format
                yf_data.rename(columns={
                    "Open": "1. open",
                    "High": "2. high",
                    "Low": "3. low",
                    "Close": "4. close",
                    "Volume": "5. volume"
                }, inplace=True)
                # Save to cache for future use
                cache[symbol] = yf_data
                return yf_data
            else:
                print("yfinance returned no data.")
                return None
        except Exception as yf_e:
            print(f"yfinance fetch failed: {str(yf_e)}")
            return None

# ---- PREPARE FEATURES ----
def prepare_features(df):
    # Print the column names to debug the issue
    print(f"Columns in the data: {df.columns}")

    # Ensure the correct column name is used
    if '4. close' not in df.columns:
        raise ValueError("'4. close' column not found in the data.")
    
    df['Day'] = np.arange(len(df))
    X = df[['Day']]
    y = df['4. close']  # Use '4. close' as the column for closing prices
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ---- TRAIN MODELS ----
def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# --- Load Pre-trained LSTM Model ---
def load_pretrained_model():
    try:
        model = load_model('lstm_model.h5')  # Load the pre-trained model
        print("Pre-trained LSTM model loaded.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# --- Preprocessing for LSTM ---
def preprocess_lstm_data(data, time_steps=TIME_STEPS, feature_cols=None):
    if feature_cols is None:
        feature_cols = ['1. open', '2. high', '3. low', '4. close', '5. volume']
    data = data[feature_cols]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(time_steps, len(scaled)):
        X.append(scaled[i-time_steps:i])
        y.append(scaled[i, feature_cols.index('4. close')])  # Predict close price
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# --- Forecast Future Prices ---
def forecast_future_prices(data, time_steps=TIME_STEPS, future_steps=FUTURE_STEPS, feature_cols=None):
    X, y, scaler = preprocess_lstm_data(data, time_steps, feature_cols)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = load_pretrained_model()  # Use the pre-trained model instead of training a new one
    if model is None:
        print("Model not found. Skipping prediction.")
        return [], []

    # Predict future prices
    last_sequence = data[feature_cols].values[-time_steps:]
    current_input = scaler.transform(last_sequence).reshape(1, time_steps, len(feature_cols))
    future_predictions = []
    for _ in range(future_steps):
        next_pred_scaled = model.predict(current_input, verbose=0)
        next_input_features = np.copy(current_input[:, -1, :])
        next_input_features[0, feature_cols.index('4. close')] = next_pred_scaled[0, 0]
        current_input = np.concatenate([current_input[:, 1:, :], next_input_features.reshape(1, 1, len(feature_cols))], axis=1)

        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, :] = next_input_features
        next_pred = scaler.inverse_transform(dummy)[0, feature_cols.index('4. close')]
        future_predictions.append(next_pred)

    future_dates = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_steps)

    # Adjust forecast for recent volatility
    if len(data) >= 2:
        recent_change = data['4. close'].iloc[-1] - data['4. close'].iloc[-2]
        weighted_change = recent_change * RECENT_WEIGHT
        future_predictions = [pred + weighted_change for pred in future_predictions]

    return future_predictions, future_dates

# --- Layout and Interactivity ---
app.layout = html.Div([
    html.H1("Stock Market Prediction with LSTM and Traditional Models"),
    dcc.Input(id='stock-ticker', type='text', value=DEFAULT_SYMBOL, debounce=True),
    html.Div(id='ticker-output'),
    dcc.Graph(id='price-history-graph'),
    dcc.Graph(id='prediction-graph'),
    dcc.Graph(id='comparison-graph'),
    html.Div(id='loading-spinner', children=[dcc.Loading(type="circle", children="Processing...")], style={'text-align': 'center'})
])

@app.callback(
    [Output('ticker-output', 'children'),
     Output('price-history-graph', 'figure'),
     Output('prediction-graph', 'figure'),
     Output('comparison-graph', 'figure'),
     Output('loading-spinner', 'children')],
    [Input('stock-ticker', 'value')]
)
def update_graph(symbol):
    print(f"Received ticker symbol: {symbol}")

    # Fetch stock data
    data = fetch_stock_data(symbol)
    if data is None or data.empty:
        print("No data fetched or empty data.")
        return "No data available for this symbol. Please try another ticker.", {}, {}, {}, "Data is processing..."

    print(f"Columns in the fetched data: {data.columns}")

    # Historical Price Plot
    historical_plot = {
        'data': [
            {'x': data.index, 'y': data['4. close'], 'type': 'line', 'name': 'Historical Close', 'line': {'color': 'blue'}}
        ],
        'layout': {
            'title': f'{symbol} Historical Closing Prices',
            'xaxis': {
                'title': 'Date',
                'tickformat': '%Y-%m-%d',
                'tickangle': 45,  # Rotate date labels for better readability
            },
            'yaxis': {'title': 'Price ($)', 'tickformat': '.2f'},
            'legend': {'title': 'Data Series'},
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'
        }
    }

    # Train models for Linear Regression and Random Forest
    X_train, X_test, y_train, y_test = prepare_features(data)
    linear_model = train_linear_model(X_train, y_train)
    rf_model = train_random_forest_model(X_train, y_train)

    # LSTM Prediction
    future_predictions, future_dates = forecast_future_prices(data, TIME_STEPS, FUTURE_STEPS, ['1. open', '2. high', '3. low', '4. close', '5. volume'])

    prediction_plot = {
        'data': [
            {'x': future_dates, 'y': future_predictions, 'type': 'line', 'name': 'Predicted Close', 'line': {'color': 'red'}}
        ],
        'layout': {
            'title': f'{symbol} Predicted Closing Prices for the Next {FUTURE_STEPS} Days',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Price ($)', 'tickformat': '.2f'},
            'legend': {'title': 'Predictions'},
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'
        }
    }

    # Comparison Plot (Actual vs Predicted)
    linear_preds = linear_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)

    comparison_plot = {
        'data': [
            {'x': data.index[-60:], 'y': data['4. close'][-60:], 'type': 'line', 'name': 'Actual Close', 'line': {'color': 'blue'}},
            {'x': future_dates, 'y': future_predictions, 'type': 'line', 'name': 'Predicted Close', 'line': {'color': 'red'}}
        ],
        'layout': {
            'title': f'Actual vs Predicted Closing Prices for {symbol}',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Price ($)', 'tickformat': '.2f'},
            'legend': {'title': 'Data Series'},
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'
        }
    }

    # Hide loading spinner
    return f"Data for {symbol}:", historical_plot, prediction_plot, comparison_plot, None

# --- Run the app ---
if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)
