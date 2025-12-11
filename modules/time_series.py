import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def show():
    st.title("📈 Time Series Forecasting")
    
    st.markdown("""
    Time series data has a temporal ordering. Learn to predict future values based on historical patterns!
    """)
    
    tabs = st.tabs(["📚 Concepts", "📊 Components", "🔮 Moving Averages", "📉 Exponential Smoothing", "🎮 Interactive Forecast"])
    
    # TAB 1: Concepts
    with tabs[0]:
        st.header("What is Time Series?")
        
        st.markdown("""
        **Time Series** is a sequence of data points indexed in time order.
        
        **Examples:**
        - Stock prices (daily)
        - Temperature readings (hourly)
        - Website traffic (monthly)
        - Sales data (quarterly)
        """)
        
        st.subheader("Key Concepts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Components of Time Series
            
            1. **Trend** 📈: Long-term increase or decrease
            2. **Seasonality** 🔄: Repeating patterns (daily, weekly, yearly)
            3. **Cyclical** 🌊: Non-fixed patterns (economic cycles)
            4. **Noise** 📡: Random variations
            """)
            
        with col2:
            st.markdown("""
            ### Common Methods
            
            - **Moving Average**: Smooth out noise
            - **Exponential Smoothing**: Recent values matter more
            - **ARIMA**: Statistical forecasting
            - **Prophet**: Facebook's tool for trends + seasonality
            - **LSTM**: Deep learning for sequences
            """)
        
        st.subheader("Stationarity")
        st.info("""
        A **stationary** series has constant mean and variance over time.
        Most forecasting methods require stationarity!
        
        **How to make stationary:**
        - Differencing (subtract previous value)
        - Log transformation
        - Detrending
        """)
    
    # TAB 2: Components
    with tabs[1]:
        st.header("Visualizing Time Series Components")
        
        # Generate synthetic data with trend, seasonality, and noise
        np.random.seed(42)
        n_points = 365
        dates = pd.date_range(start='2023-01-01', periods=n_points, freq='D')
        
        # Components
        trend = np.linspace(100, 150, n_points)  # Upward trend
        seasonality = 20 * np.sin(2 * np.pi * np.arange(n_points) / 30)  # Monthly pattern
        noise = np.random.randn(n_points) * 5
        
        # Combined
        values = trend + seasonality + noise
        
        df = pd.DataFrame({
            'Date': dates,
            'Value': values,
            'Trend': trend,
            'Seasonality': seasonality,
            'Noise': noise
        })
        
        st.subheader("Combined Time Series")
        fig1 = px.line(df, x='Date', y='Value', title='Original Time Series (Trend + Seasonality + Noise)')
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Decomposed Components")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig2 = px.line(df, x='Date', y='Trend', title='Trend Component')
            fig2.update_traces(line_color='green')
            st.plotly_chart(fig2, use_container_width=True)
            
        with col2:
            fig3 = px.line(df, x='Date', y='Seasonality', title='Seasonality Component')
            fig3.update_traces(line_color='orange')
            st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = px.scatter(df, x='Date', y='Noise', title='Noise/Residual Component')
        fig4.update_traces(marker=dict(size=3))
        st.plotly_chart(fig4, use_container_width=True)
    
    # TAB 3: Moving Averages
    with tabs[2]:
        st.header("Moving Averages")
        
        st.markdown("""
        **Moving Average** smooths data by averaging nearby values.
        
        $$MA_t = \\frac{1}{k} \\sum_{i=0}^{k-1} y_{t-i}$$
        
        Where $k$ is the window size.
        """)
        
        # Generate data
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
        trend = np.linspace(50, 100, n)
        noise = np.random.randn(n) * 15
        values = trend + noise
        
        df = pd.DataFrame({'Date': dates, 'Value': values})
        
        # Interactive window size
        window = st.slider("Window Size (k):", 3, 30, 7)
        
        # Calculate MA
        df['MA'] = df['Value'].rolling(window=window).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Value'], mode='lines', name='Original', opacity=0.5))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA'], mode='lines', name=f'{window}-day MA', line=dict(width=3, color='red')))
        fig.update_layout(title=f"Moving Average (Window={window})", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Observations:**
        - Larger window = Smoother curve, but loses detail
        - Smaller window = More responsive, but noisier
        - MA lags behind the actual data!
        """)
    
    # TAB 4: Exponential Smoothing
    with tabs[3]:
        st.header("Exponential Smoothing")
        
        st.markdown("""
        **Exponential Smoothing** gives more weight to recent observations.
        
        $$S_t = \\alpha \\cdot y_t + (1-\\alpha) \\cdot S_{t-1}$$
        
        Where $\\alpha$ (alpha) is the smoothing factor (0 to 1).
        - $\\alpha \\approx 1$: More weight to recent data (responsive)
        - $\\alpha \\approx 0$: More weight to history (smooth)
        """)
        
        # Generate data
        np.random.seed(42)
        n = 100
        trend = np.linspace(50, 100, n)
        noise = np.random.randn(n) * 15
        values = trend + noise
        
        # Interactive alpha
        alpha = st.slider("Smoothing Factor (α):", 0.1, 0.9, 0.3, 0.1)
        
        # Calculate Exponential Smoothing
        def exp_smooth(values, alpha):
            result = [values[0]]
            for i in range(1, len(values)):
                result.append(alpha * values[i] + (1 - alpha) * result[-1])
            return result
        
        smoothed = exp_smooth(values, alpha)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=values, mode='lines', name='Original', opacity=0.5))
        fig.add_trace(go.Scatter(y=smoothed, mode='lines', name=f'Exp. Smooth (α={alpha})', line=dict(width=3, color='green')))
        fig.update_layout(title=f"Exponential Smoothing (α={alpha})", xaxis_title="Time", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Simple Exponential Smoothing")
            st.write("Best for: Data with no trend or seasonality")
        with col2:
            st.markdown("### Holt-Winters")
            st.write("Handles: Trend + Seasonality")
    
    # TAB 5: Interactive Forecast
    with tabs[4]:
        st.header("🎮 Build Your Own Forecast")
        
        st.subheader("1. Generate Synthetic Stock Price")
        
        col1, col2 = st.columns(2)
        with col1:
            initial_price = st.number_input("Initial Price ($):", 50, 500, 100)
            volatility = st.slider("Volatility:", 0.5, 5.0, 2.0)
        with col2:
            trend_strength = st.slider("Trend Strength:", -0.5, 0.5, 0.1)
            n_days = st.slider("Number of Days:", 30, 365, 100)
        
        # Generate stock-like data (random walk with drift)
        np.random.seed(st.session_state.get('ts_seed', 42))
        returns = np.random.randn(n_days) * volatility + trend_strength
        prices = initial_price * np.cumprod(1 + returns / 100)
        
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        df = pd.DataFrame({'Date': dates, 'Price': prices})
        
        if st.button("🔀 Randomize Data"):
            st.session_state.ts_seed = np.random.randint(0, 10000)
            st.rerun()
        
        st.subheader("2. Choose Forecasting Method")
        
        method = st.selectbox("Method:", ["Moving Average", "Exponential Smoothing", "Linear Trend"])
        forecast_days = st.slider("Forecast Horizon (days):", 7, 60, 14)
        
        # Train/Test Split
        train_size = len(df) - forecast_days
        train = df[:train_size]
        test = df[train_size:]
        
        # Generate forecast
        if method == "Moving Average":
            window = st.slider("MA Window:", 5, 20, 10)
            last_ma = train['Price'].tail(window).mean()
            forecast = [last_ma] * forecast_days
            
        elif method == "Exponential Smoothing":
            alpha = st.slider("Alpha (α):", 0.1, 0.9, 0.3, 0.1)
            last_value = train['Price'].iloc[-1]
            forecast = [last_value]
            for i in range(forecast_days - 1):
                forecast.append(alpha * forecast[-1] + (1 - alpha) * forecast[-1])
                
        else:  # Linear Trend
            from sklearn.linear_model import LinearRegression
            X_train = np.arange(len(train)).reshape(-1, 1)
            y_train = train['Price'].values
            model = LinearRegression()
            model.fit(X_train, y_train)
            X_forecast = np.arange(len(train), len(train) + forecast_days).reshape(-1, 1)
            forecast = model.predict(X_forecast)
        
        # Plot
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(x=train['Date'], y=train['Price'], mode='lines', name='Historical', line=dict(color='blue')))
        
        # Actual (test)
        fig.add_trace(go.Scatter(x=test['Date'], y=test['Price'], mode='lines', name='Actual', line=dict(color='green', dash='dot')))
        
        # Forecast
        fig.add_trace(go.Scatter(x=test['Date'], y=forecast, mode='lines', name='Forecast', line=dict(color='red', width=3)))
        
        fig.update_layout(title=f"Stock Price Forecast ({method})", xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Error Metrics
        actual = test['Price'].values
        forecast_arr = np.array(forecast)
        
        mae = np.mean(np.abs(actual - forecast_arr))
        mape = np.mean(np.abs((actual - forecast_arr) / actual)) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
        with col2:
            st.metric("Mean Absolute % Error (MAPE)", f"{mape:.1f}%")
        
        st.info("""
        **Pro Tips:**
        - MAPE < 10%: Excellent forecast
        - MAPE 10-20%: Good forecast
        - MAPE > 30%: Poor forecast
        """)
