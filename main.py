import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
import io
from scipy.stats import norm

# greeks
def calculate_greeks(df, S, r=0.05, T=30/365):
    K = df['strike']
    sigma = df['impliedVolatility'].fillna(0.2)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df['delta'] = norm.cdf(d1)
    df['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    df['vega'] = S * norm.pdf(d1) * np.sqrt(T)
    df['theta'] = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    df['rho'] = K * T * np.exp(-r*T) * norm.cdf(d2)
    return df

# bs price
def calculate_theoretical_price(df, S, r=0.05, option_type='Call'):
    today = datetime.today()
    df['T'] = df['expiry'].apply(lambda x: (pd.to_datetime(x) - today).days / 365)
    K = df['strike']
    sigma = df['impliedVolatility'].fillna(0.2)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * df['T']) / (sigma * np.sqrt(df['T']))
    d2 = d1 - sigma * np.sqrt(df['T'])
    if option_type == "Call":
        df['theoretical_price'] = S * norm.cdf(d1) - K * np.exp(-r * df['T']) * norm.cdf(d2)
    else:
        df['theoretical_price'] = K * np.exp(-r * df['T']) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return df

# moneyness
def calculate_moneyness(df, S, option_type):
    if option_type == "Call":
        df['moneyness'] = S / df['strike']
    else:
        df['moneyness'] = df['strike'] / S
    return df

# execute SQL query
def execute_query(df: pd.DataFrame, query: str):
    return duckdb.query(query).to_df()

# download function 
def get_image_download_link(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

st.title("\U0001F4C8 Option Scraper YFinance")

st.markdown("**Choose a ticker**")

# choosing box
popular_tickers = ["SPY", "QQQ", "DIA", "IWM", "^VIX", "GLD", "USO", "Other (enter manually)"]
ticker_selection = st.selectbox("Choose a ticker or enter it manually", popular_tickers)

if ticker_selection == "Other (enter manually)":
    ticker_symbol = st.text_input("Enter the ticker symbol", value="AAPL")
else:
    ticker_symbol = ticker_selection

ticker = yf.Ticker(ticker_symbol)

exp_dates = ticker.options
selected_dates = st.multiselect("Select one or more expiration dates", exp_dates)
option_type = st.radio("Option type", ["Call", "Put"])
min_volume = st.slider("Minimum volume", min_value=0, max_value=1000, value=100, step=50)
filter_itm = st.checkbox("Only in-the-money", value=False)

if selected_dates:
    all_data = []
    for date in selected_dates:
        chain = ticker.option_chain(date)
        df = chain.calls if option_type == "Call" else chain.puts
        df['expiry'] = date
        if filter_itm:
            df = df[df['inTheMoney'] == True]
        df = df[df['volume'] >= min_volume]
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        df = calculate_greeks(df, S=current_price)
        df = calculate_theoretical_price(df, S=current_price, option_type=option_type)
        df = calculate_moneyness(df, S=current_price, option_type=option_type)
        all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)
    st.subheader("Filtered Options Data")
    st.dataframe(df)

    graph_type = st.selectbox("Graph to display", ["Open Interest", "Implied Volatility", "Theoretical vs Market Price", "Moneyness Distribution", "Greeks Charts"])

    st.subheader(f"{graph_type} vs Strike Chart")
    if graph_type == "Greeks Charts":
        for greek in ["delta", "gamma", "theta", "rho"]:
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x='strike', y=greek, hue='expiry', marker='o', ax=ax)
            ax.set_xlabel("Strike")
            ax.set_ylabel(greek.capitalize())
            ax.set_title(f"{greek.capitalize()} - {option_type} (multi-expiry)")
            st.pyplot(fig)
            buf = get_image_download_link(fig, f"{ticker_symbol}_{greek}.png")
            st.download_button(f"\U0001F4F7 Download {greek} chart", data=buf, file_name=f"{ticker_symbol}_{greek}.png", mime="image/png")

    elif graph_type == "Theoretical vs Market Price":
        fig, ax = plt.subplots()
        color_map = plt.get_cmap("tab10")
        for i, expiry_date in enumerate(df['expiry'].unique()):
            subset = df[df['expiry'] == expiry_date].sort_values(by='strike')
            color = color_map(i % 10)
            ax.plot(subset['strike'], subset['lastPrice'], label=f"Market - {expiry_date}", linestyle='-', marker='o', color=color_map(i * 2 % 10))
            ax.plot(subset['strike'], subset['theoretical_price'], label=f"Theoretical - {expiry_date}", linestyle='--', marker='x', color=color_map((i * 2 + 1) % 10))
        ax.set_xlabel("Strike")
        ax.set_ylabel("Price")
        ax.set_title(f"Theoretical vs Market Price - {option_type}")
        ax.legend()
        st.pyplot(fig)

    elif graph_type == "Moneyness Distribution":
        fig, ax = plt.subplots()
        sns.histplot(data=df, x='moneyness', bins=30, hue='expiry', ax=ax, kde=True)
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Moneyness Distribution - {option_type}")
        st.pyplot(fig)

    else:
        fig, ax = plt.subplots()
        if graph_type == "Open Interest":
            sns.barplot(data=df, x='strike', y='openInterest', hue='expiry', ax=ax)
            ax.set_ylabel("Open Interest")
        elif graph_type == "Implied Volatility":
            sns.lineplot(data=df, x='strike', y='impliedVolatility', hue='expiry', marker='o', ax=ax)
            ax.set_ylabel("Implied Volatility")
        ax.set_xlabel("Strike")
        ax.set_title(f"{graph_type} - {option_type} (multi-expiry)")
        st.pyplot(fig)
        buf = get_image_download_link(fig, f"{ticker_symbol}_{graph_type}.png")
        st.download_button(f"\U0001F4F7 Download {graph_type} chart", data=buf, file_name=f"{ticker_symbol}_{graph_type}.png", mime="image/png")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("\U0001F4E5 Download CSV", data=csv, file_name=f"{ticker_symbol}_{option_type}_multi_expiry.csv", mime='text/csv')

    with st.expander("\U0001F4BB Analyze data with an SQL query"):
        st.markdown("""
        ℹ️ **Quick guide**:
        - Use `df` as the table name.
        - Examples:
            - SELECT * FROM df WHERE volume > 100
            - SELECT strike, volume FROM df WHERE inTheMoney = TRUE
            - SELECT * FROM df ORDER BY impliedVolatility DESC LIMIT 10
            - SELECT expiry, AVG(impliedVolatility) FROM df GROUP BY expiry
        """)
        query_input = st.text_area("Write an SQL query (use 'df' as the table)",
                                   value="SELECT * FROM df WHERE impliedVolatility > 0.3 ORDER BY volume DESC")
        if st.button("Execute Query"):
            try:
                result = execute_query(df, query_input)
                st.dataframe(result)
            except Exception as e:
                st.error(f"Query error: {e}")

    st.markdown("""
        <div style='text-align: center; margin-top: 50px;'>
            <a href="https://www.linkedin.com/in/mascapuano" target="_blank">
                <button style="background-color: #0077B5; color: white; padding: 10px 20px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer;">
                    🔗 Connect on LinkedIn
                </button>
            </a>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='text-align: center; margin-top: 10px;'>
            <a href="https://github.com/capuanomassimo" target="_blank">
                <button style="background-color: #333; color: white; padding: 10px 20px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer;">
                    🌐 Visit my GitHub
                </button>
            </a>
        </div>
    """, unsafe_allow_html=True)