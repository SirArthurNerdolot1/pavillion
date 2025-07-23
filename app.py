import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="BTC Strategy Dashboard", page_icon="📈")
st.title("📈 BTC-USD Technical Backtest Dashboard")

# --- Sidebar Input ---
st.sidebar.header("📅 Time & Capital Settings")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
initial_capital = st.sidebar.number_input("💰 Initial Capital (USD)", min_value=1000, value=10000000, step=1000)

# --- Load Data ---
@st.cache_data

def load_data(start, end):
    df = yf.download("BTC-USD", start=start, end=end, interval="1d")
    df.dropna(inplace=True)
    df.columns = df.columns.get_level_values(0)
    return df.reset_index()

df = load_data(start_date, end_date)

# --- Indicator Functions ---
def ichimoku_cloud(df):
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()

    df['Tenkan'] = (high_9 + low_9) / 2
    df['Kijun'] = (high_26 + low_26) / 2
    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
    df['Chikou'] = df['Close'].shift(-26)
    return df

def fibonacci_levels(df):
    recent_low = df['Low'].min()
    recent_high = df['High'].max()
    diff = recent_high - recent_low

    levels = {
        '0.0%': recent_high,
        '23.6%': recent_high - 0.236 * diff,
        '38.2%': recent_high - 0.382 * diff,
        '50.0%': recent_high - 0.500 * diff,
        '61.8%': recent_high - 0.618 * diff,
        '78.6%': recent_high - 0.786 * diff,
        '100.0%': recent_low
    }
    return levels

def generate_signals(df, fib_levels, tolerance=0.01):
    signals = []
    for i in range(len(df)):
        row = df.iloc[i]
        price = row['Close']
        senkou_a = row['Senkou_A']
        senkou_b = row['Senkou_B']
        tenkan = row['Tenkan']
        kijun = row['Kijun']

        if any(pd.isna([senkou_a, senkou_b, tenkan, kijun])):
            signals.append(0)
            continue

        cloud_top = max(float(senkou_a), float(senkou_b))
        cloud_bottom = min(float(senkou_a), float(senkou_b))

        near_support = any(abs(price - level)/price <= tolerance for label, level in fib_levels.items() if float(label.strip('%')) > 50)
        near_resistance = any(abs(price - level)/price <= tolerance for label, level in fib_levels.items() if float(label.strip('%')) < 50)

        if price > cloud_top or (tenkan > kijun and near_support):
            signal = 1
        elif price < cloud_bottom or (tenkan < kijun and near_resistance):
            signal = -1
        else:
            signal = 0

        signals.append(signal)
    df['Signal'] = signals
    return df

def macd(df):
    df['6_day_EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['13_day_EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['6_day_EMA'] - df['13_day_EMA']
    df['signal_line'] = df['MACD'].ewm(span=4, adjust=False).mean()
    return df

def clean_signals(df):
    signal = list(df['Signal'])
    current = 0
    for i in range(len(signal)):
        if current:
            if signal[i] and signal[i] != current:
                current = signal[i]
            else:
                signal[i] = 0
        else:
            current = signal[i]
    df['Signal'] = signal
    return df

def atr(df):
    tr = [df['High'][0] - df['Low'][0]]
    for i in range(1, len(df)):
        tr.append(max(df['High'][i] - df['Low'][i], abs(df['High'][i] - df['Close'][i-1]), abs(df['Low'][i] - df['Close'][i-1])))
    df['true_range'] = tr
    df['atr'] = df['true_range'].ewm(span=14, adjust=True).mean()
    del df['true_range']
    return df

def sharpe_ratio(return_series, N, rf):
    mean = return_series.mean() * N - rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma

def sortino_ratio(series, N, rf):
    mean = series.mean() * N - rf
    std_neg = series[series < 0].std() * np.sqrt(N)
    return mean / std_neg

def backtest(df):
    capt = initial_capital
    opening_capital = initial_capital
    signals = list(df['Signal'])
    close = list(df['Close'])
    no_of_trades, long_trades, winning_trades = 0, 0, 0
    atr_vals = list(df['atr'])
    current_trade = 0
    entry_price = 0
    shares = 0
    portfolio_value = []
    ht = []
    entry_index = 0
    for i in range(len(df)):
        if current_trade:
            if current_trade == 1:
                if close[i] < (entry_price - atr_vals[i]):
                    capt += shares * (close[i] - entry_price)
                    current_trade = 0
                    shares = 0
                    no_of_trades += 1
                    long_trades += 1
                    ht.append(i - entry_index)
                portfolio_value.append(capt + (shares * (close[i] - entry_price)) if current_trade else capt)
            else:
                if close[i] > (entry_price + atr_vals[i]):
                    capt += shares * (entry_price - close[i])
                    current_trade = 0
                    shares = 0
                    no_of_trades += 1
                    ht.append(i - entry_index)
                portfolio_value.append(capt + (shares * (entry_price - close[i])) if current_trade else capt)
        else:
            portfolio_value.append(capt)
        if signals[i] == 1:
            if current_trade == -1:
                capt += shares * (entry_price - close[i])
                winning_trades += int(entry_price > close[i])
                current_trade = 0
                shares = 0
                no_of_trades += 1
                ht.append(i - entry_index)
            current_trade = 1
            shares = int(capt / close[i])
            entry_price = close[i]
            entry_index = i
        if signals[i] == -1:
            if current_trade == 1:
                capt += shares * (close[i] - entry_price)
                current_trade = 0
                shares = 0
                no_of_trades += 1
                long_trades += 1
                ht.append(i - entry_index)
                winning_trades += int(close[i] > entry_price)
            current_trade = -1
            entry_price = close[i]
            shares = int(capt / close[i])
            entry_index = i
    df['portfolio_value'] = portfolio_value
    dd = []
    peak = portfolio_value[0]
    for value in portfolio_value:
        peak = max(peak, value)
        dd.append((peak - value) / peak)
    max_drawdown = max(dd)
    average_drawdown = sum(dd) / len(dd)
    excess_returns = [0] + [(portfolio_value[i] - portfolio_value[i - 1]) * 100 / portfolio_value[i - 1] for i in range(1, len(portfolio_value))]
    df['excess_returns'] = excess_returns
    sr1 = sharpe_ratio(df['excess_returns'], len(df), 0.04)
    sr2 = sortino_ratio(df['excess_returns'], len(df), 0.04)

    st.subheader("📊 Backtest Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Benchmark Return", f"{(close[-1] - close[0]) * 100 / close[0]:.2f}%")
    col2.metric("Portfolio Return", f"{(portfolio_value[-1] - portfolio_value[0]) * 100 / portfolio_value[0]:.2f}%")
    col3.metric("Sharpe Ratio", f"{sr1:.2f}")

    st.markdown(f"**Win Rate:** {winning_trades * 100 / no_of_trades:.2f}%")
    st.markdown(f"**Total Trades:** {no_of_trades} (Long: {long_trades}, Short: {no_of_trades - long_trades})")
    st.markdown(f"**Max Drawdown:** -{max_drawdown * 100:.2f}%")
    st.markdown(f"**Sortino Ratio:** {sr2:.2f}")
    st.markdown(f"**Average Holding Time:** {np.mean(ht):.2f} days, **Max:** {max(ht)} days")

# --- Indicator Pipeline ---
df = ichimoku_cloud(df)
fib_levels = fibonacci_levels(df)
df = generate_signals(df, fib_levels)
df = macd(df)

signals = list(df['Signal'])
macd_signals = [0]
signal_line = list(df['signal_line'])
for i in range(1, len(df)):
    if signal_line[i - 1] > 0 and signal_line[i] < 0:
        macd_signals.append(-1)
    elif signal_line[i - 1] < 0 and signal_line[i] > 0:
        macd_signals.append(1)
    else:
        macd_signals.append(0)
final_signals = [signals[i] if signals[i] or macd_signals[i] else 0 for i in range(len(df))]
df['Signal'] = final_signals
df = clean_signals(df)
df = atr(df)


# --- Plotting ---
def plot_ichimoku_fib(df, fib_levels):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Tenkan'], name='Tenkan'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Kijun'], name='Kijun'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Senkou_A'], name='Senkou A'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Senkou_B'], name='Senkou B'))
    buy = df[df['Signal'] == 1]
    sell = df[df['Signal'] == -1]
    fig.add_trace(go.Scatter(x=buy['Date'], y=buy['Close'], mode='markers', name='Buy', marker=dict(color='green', symbol='triangle-up', size=10)))
    fig.add_trace(go.Scatter(x=sell['Date'], y=sell['Close'], mode='markers', name='Sell', marker=dict(color='red', symbol='triangle-down', size=10)))
    for label, level in fib_levels.items():
        fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]], y=[level, level], mode='lines', line=dict(dash='dash', color='gray'), name=f'Fib {label}'))
    fig.update_layout(title="Ichimoku + Fibonacci + Signals")
    return fig

# --- Show Plots ---
st.plotly_chart(plot_ichimoku_fib(df, fib_levels), use_container_width=True)
backtest(df)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['portfolio_value'], name='Portfolio Value', line=dict(color='blue')))
fig.update_layout(title='📈 Portfolio Value Over Time')
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='gray')))
fig2.add_trace(go.Scatter(x=df['Date'], y=df['atr'] * 10, name='ATR x10', line=dict(color='orange')))
st.plotly_chart(fig2, use_container_width=True)

with st.expander("📄 Show Final Data"):
    st.dataframe(df.tail(100), use_container_width=True)
