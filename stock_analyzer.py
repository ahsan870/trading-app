import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(page_title="MarketMaster AI", layout="wide")

# Add custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def get_stock_data(ticker, period='2y'):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        
        # For intraday periods, check market hours
        if period in ['5m', '10m', '15m', '30m', '1h']:
            current_time = datetime.now().time()
            market_open = datetime.strptime('09:30', '%H:%M').time()
            market_close = datetime.strptime('16:00', '%H:%M').time()
            
            # If outside market hours, show message and return daily data
            if not (market_open <= current_time <= market_close):
                st.info("""
                🕒 Market is currently closed. 
                
                Market Hours (EST):
                - Monday to Friday
                - 9:30 AM to 4:00 PM
                
                Showing daily data instead of intraday data.
                """)
                # Get daily data instead
                df = stock.history(period='5d', interval='1d')
                if df.empty:
                    st.warning("No recent data available for this stock.")
                    return None, None
                return stock, df
        
        # Normal data fetch
        df = stock.history(period=period)
        
        # Check if data is empty
        if df.empty:
            st.warning(f"No data available for {ticker} in the selected time period.")
            return None, None
            
        return stock, df
        
    except Exception as e:
        st.info("Unable to fetch data. Please try again with a different time period.")
        return None, None

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # Moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def predict_stock_price(df, prediction_days=90):
    """Predict future stock prices using simple moving average"""
    try:
        # Calculate the average daily price change
        daily_returns = df['Close'].pct_change().mean()
        
        # Get the last closing price
        last_price = df['Close'].iloc[-1]
        last_date = df.index[-1]
        
        # Create future dates based on selected timeline
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=prediction_days
        )
        
        # Calculate predicted prices
        predicted_prices = [last_price * (1 + daily_returns) ** i 
                          for i in range(1, prediction_days + 1)]
        
        # Create forecast dataframe
        forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': predicted_prices,
            'yhat_lower': [price * 0.9 for price in predicted_prices],
            'yhat_upper': [price * 1.1 for price in predicted_prices]
        })
        
        return forecast
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def calculate_fair_value(stock):
    """Calculate a simple fair value estimate and provide valuation signal"""
    try:
        # Get financial data
        pe_ratio = stock.info.get('forwardPE', None)
        eps = stock.info.get('forwardEps', None)
        current_price = stock.history(period='1d')['Close'][-1]

        # Simple fair value calculation
        if pe_ratio and eps:
            industry_avg_pe = 15  # This is a simplified assumption
            fair_value = eps * industry_avg_pe

            # Determine valuation signal
            if current_price < fair_value:
                signal = "Undervalued"
            elif current_price > fair_value:
                signal = "Overvalued"
            else:
                signal = "Fairly Valued"

            return fair_value, signal

        return None, "Data not available"
    except Exception as e:
        return None, str(e)

def main():
    # Add a sidebar for controls
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/stocks.png", width=100)
        st.title("Controls")
        
        # Move period selector to sidebar
        period = st.selectbox(
            "Select Time Period:",
            ['5m', '10m', '15m', '30m', '1h', '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'],
            index=11,  # Default to 2y
            help="Choose the historical data timeframe"
        )
        
        # Add market hours warning for intraday selections
        if period in ['5m', '10m', '15m', '30m', '1h']:
            current_time = datetime.now().time()
            market_open = datetime.strptime('09:30', '%H:%M').time()
            market_close = datetime.strptime('16:00', '%H:%M').time()
            
            if not (market_open <= current_time <= market_close):
                st.info("📊 Market hours: 9:30 AM - 4:00 PM EST. Historical daily data is shown outside market hours.")
        
        # Add theme selector
        theme = st.selectbox(
            "Chart Theme:",
            ["Light", "Dark"],
            help="Select chart color theme"
        )
    
    # Add footer to sidebar (add this at the end of your sidebar section)
    st.sidebar.markdown("<br>" * 10, unsafe_allow_html=True)  # Add some spacing
    
    sidebar_footer = """
    <div style="position: fixed; 
                bottom: 0; 
                left: 0; 
                width: 17rem; 
                background-color: transparent;
                padding: 10px 0; 
                text-align: center;">
        <div style="margin: 0 auto;
                    padding: 10px;
                    color: rgba(250, 250, 250, 0.6);
                    font-size: 14px;">
            © 2024 MarketMaster AI<br>
            Developed by Cache Ahmed<br>
            <a href="mailto:ahmed.bitsandbytes@gmail.com" 
               style="color: rgba(250, 250, 250, 0.6); 
                      text-decoration: none;
                      transition: color 0.3s;">
                📧 ahmed.bitsandbytes@gmail.com
            </a>
        </div>
    </div>
    """
    st.sidebar.markdown(sidebar_footer, unsafe_allow_html=True)

    # Main content
    st.title(" MarketMaster AI ")
    st.markdown("---")
    
    # Create a container for the input section
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker = st.text_input(
                "Enter Stock Ticker:",
                "AAPL",
                help="Enter the stock symbol (e.g., AAPL for Apple)"
            ).upper()
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align with input
            analyze_button = st.button(
                "📈 Let's Get Rich!",
                use_container_width=True
            )

    if analyze_button:
        # Add a progress bar
        progress_bar = st.progress(0)
        
        with st.spinner('Fetching stock data...'):
            stock, df = get_stock_data(ticker, period=period)
            progress_bar.progress(33)
            
            df = calculate_technical_indicators(df)
            progress_bar.progress(66)
            
            # Move progress bar completion and success message before the tabs
            progress_bar.progress(100)
            st.success('Analysis completed!')
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📊 Technical Analysis", 
                "🔮 Price Prediction", 
                "📈 Statistics", 
                "💰 Valuation",
                "🏢 Company Info",
                "📰 News & Sentiment",
                "🧠 Smart Analysis",
                "🤖 Backtesting"
            ])
            
            with tab1:
                # Technical Analysis Chart
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  vertical_spacing=0.03, row_heights=[0.7, 0.3])

                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='OHLC'
                    ),
                    row=1, col=1
                )

                # Add Moving Averages
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA20'],
                        name='MA20',
                        line=dict(color='orange', width=2)
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA50'],
                        name='MA50',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )

                # Add RSI
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['RSI'],
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ),
                    row=2, col=1
                )

                # Add RSI overbought/oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                # Update layout based on theme
                bg_color = "white" if theme == "Light" else "black"
                text_color = "black" if theme == "Light" else "white"
                grid_color = "lightgrey" if theme == "Light" else "grey"

                fig.update_layout(
                    title=f'{ticker} Technical Analysis',
                    yaxis_title='Stock Price (USD)',
                    yaxis2_title='RSI',
                    xaxis_rangeslider_visible=False,
                    height=800,
                    showlegend=True,
                    paper_bgcolor=bg_color,
                    plot_bgcolor=bg_color,
                    font=dict(color=text_color)
                )

                # Update axes
                fig.update_xaxes(showgrid=True, gridcolor=grid_color)
                fig.update_yaxes(showgrid=True, gridcolor=grid_color)

                # Show the chart
                st.plotly_chart(fig, use_container_width=True)

                # Add some technical analysis insights
                st.subheader("Technical Analysis Insights")
                col1, col2, col3 = st.columns(3)
                
                # Current price vs Moving Averages
                current_price = df['Close'].iloc[-1]
                ma20_current = df['MA20'].iloc[-1]
                ma50_current = df['MA50'].iloc[-1]
                rsi_current = df['RSI'].iloc[-1]

                with col1:
                    ma20_signal = "ABOVE" if current_price > ma20_current else "BELOW"
                    st.metric(
                        "Price vs MA20",
                        f"{ma20_signal} MA20",
                        f"{((current_price/ma20_current)-1)*100:.2f}%"
                    )

                with col2:
                    ma50_signal = "ABOVE" if current_price > ma50_current else "BELOW"
                    st.metric(
                        "Price vs MA50",
                        f"{ma50_signal} MA50",
                        f"{((current_price/ma50_current)-1)*100:.2f}%"
                    )

                with col3:
                    rsi_signal = "OVERBOUGHT" if rsi_current > 70 else "OVERSOLD" if rsi_current < 30 else "NEUTRAL"
                    st.metric(
                        "RSI Signal",
                        rsi_signal,
                        f"{rsi_current:.2f}"
                    )
            
            with tab2:
                # Simplified Prediction section
                st.subheader("Price Prediction")
                
                # Fixed prediction for 2 years (730 days)
                prediction_days = 730
                
                try:
                    with st.spinner('Generating long-term prediction...'):
                        forecast = predict_stock_price(df, prediction_days)
                        
                        if forecast is not None:
                            # Create prediction chart
                            fig_pred = go.Figure()
                            
                            # Add historical price line
                            fig_pred.add_trace(go.Scatter(
                                x=df.index,
                                y=df['Close'],
                                name='Historical Price',
                                line=dict(color='blue')
                            ))
                            
                            # Add prediction line
                            fig_pred.add_trace(go.Scatter(
                                x=forecast['ds'],
                                y=forecast['yhat'],
                                name='Predicted Price',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            # Add confidence interval
                            fig_pred.add_trace(go.Scatter(
                                x=forecast['ds'],
                                y=forecast['yhat_upper'],
                                y0=forecast['yhat_lower'],
                                fill='tonexty',
                                fillcolor='rgba(255,0,0,0.1)',
                                line=dict(color='rgba(255,0,0,0)'),
                                name='Prediction Range'
                            ))
                            
                            # Update layout
                            fig_pred.update_layout(
                                title=f'{ticker} 2-Year Price Prediction',
                                yaxis_title='Price (USD)',
                                xaxis_title='Date',
                                height=600,
                                showlegend=True,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                            # Add a simple disclaimer
                            st.caption("Note: This is a simplified prediction model based on historical trends. Always conduct thorough research before making investment decisions.")
                            
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
            
            with tab3:
                # Statistics section
                st.subheader("Key Statistics")
                
                # Calculate additional statistics
                returns = df['Close'].pct_change()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                annual_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (252/len(df)) - 1
                max_drawdown = (df['Close'] / df['Close'].cummax() - 1).min()
                
                # Create two rows of metrics
                col1, col2, col3 = st.columns(3)
                col4, col5, col6 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Current Price",
                        f"${df['Close'].iloc[-1]:.2f}",
                        f"{returns.iloc[-1]:.2%} today"
                    )
                
                with col2:
                    st.metric(
                        "Annual Volatility",
                        f"{volatility:.2%}"
                    )
                
                with col3:
                    st.metric(
                        "Annual Return",
                        f"{annual_return:.2%}"
                    )
                
                with col4:
                    st.metric(
                        "52-Week High",
                        f"${df['High'][-252:].max():.2f}"
                    )
                
                with col5:
                    st.metric(
                        "52-Week Low",
                        f"${df['Low'][-252:].min():.2f}"
                    )
                
                with col6:
                    st.metric(
                        "Max Drawdown",
                        f"{max_drawdown:.2%}"
                    )
                
                # Add explanation of metrics
                st.markdown("""
                ### Understanding These Metrics:
                
                * **Current Price**: Latest trading price with today's percentage change
                * **Annual Volatility**: Measures price fluctuation risk (higher = more volatile)
                * **Annual Return**: Yearly return rate if held for the entire period
                * **52-Week High/Low**: Highest and lowest prices in the past year
                * **Max Drawdown**: Largest peak-to-trough decline, measuring downside risk
                
                These statistics help assess:
                - Risk level (through volatility and drawdown)
                - Historical performance (through returns)
                - Price ranges (through highs and lows)
                """)
            
            with tab4:
                # Valuation section
                st.subheader("Stock Valuation Analysis")
                
                # Calculate fair value
                fair_value, valuation_signal = calculate_fair_value(stock)
                current_price = df['Close'].iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Current Market Price",
                        f"${current_price:.2f}"
                    )
                
                with col2:
                    if fair_value:
                        st.metric(
                            "Estimated Fair Value",
                            f"${fair_value:.2f}",
                            f"{((fair_value/current_price)-1)*100:.2f}%"
                        )
                    else:
                        st.metric("Estimated Fair Value", "N/A")
                
                with col3:
                    if valuation_signal:
                        color = ""
                        if valuation_signal == "Undervalued":
                            color = "green"
                        elif valuation_signal == "Overvalued":
                            color = "red"
                        st.markdown(f"<h3 style='text-align: center; color: {color};'>{valuation_signal}</h3>", 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown("<h3 style='text-align: center;'>No Signal Available</h3>", 
                                  unsafe_allow_html=True)
                
                # Add additional valuation metrics if available
                if fair_value:
                    st.markdown("---")
                    st.markdown(f"""
                        ### Valuation Details
                        - Price difference from fair value: ${abs(current_price - fair_value):.2f}
                        - Potential return to fair value: {((fair_value/current_price)-1)*100:.2f}%
                    """)

            with tab5:
                st.subheader("Company Information")
                
                # Get company info
                info = stock.info
                
                # Company Profile
                col1, col2 = st.columns([2,1])
                with col1:
                    st.markdown(f"### {info.get('longName', '')}")
                    st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.markdown(f"**Website:** [{info.get('website', 'N/A')}]({info.get('website', '#')})")
                    st.markdown("### Business Summary")
                    st.markdown(info.get('longBusinessSummary', 'No description available.'))
                
                with col2:
                    # Key company metrics
                    metrics = {
                        "Market Cap": info.get('marketCap', 'N/A'),
                        "P/E Ratio": info.get('forwardPE', 'N/A'),
                        "EPS": info.get('forwardEps', 'N/A'),
                        "Dividend Yield": info.get('dividendYield', 'N/A'),
                        "Beta": info.get('beta', 'N/A')
                    }
                    
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            if key == "Market Cap":
                                value = f"${value/1e9:.2f}B"
                            elif key == "Dividend Yield":
                                value = f"{value*100:.2f}%"
                            elif key == "Beta":
                                value = f"{value:.2f}"
                        st.metric(key, value)

            with tab6:
                st.subheader("News & Sentiment Analysis")
                
                try:
                    # Get news from yfinance with error handling
                    try:
                        news = stock.news if stock is not None else []
                    except:
                        news = []

                    if not news:
                        st.info("""
                        ### No Recent News Available
                        
                        This could be due to:
                        - No recent news for this stock
                        - Market being closed
                        - API limitations
                        
                        Try:
                        - Checking major stocks (e.g., AAPL, GOOGL)
                        - Refreshing in a few minutes
                        - Viewing during market hours
                        """)
                    else:
                        try:
                            # Calculate sentiment scores with validation
                            sentiment_scores = []
                            valid_news = []
                            
                            for article in news[:10]:
                                if article and article.get('title'):
                                    analysis = TextBlob(article.get('title', ''))
                                    sentiment_scores.append(analysis.sentiment.polarity)
                                    valid_news.append(article)
                            
                            if sentiment_scores:
                                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                                
                                # Display sentiment indicator
                                col1, col2, col3 = st.columns([1,2,1])
                                with col2:
                                    sentiment_color = 'red' if avg_sentiment < -0.2 else 'green' if avg_sentiment > 0.2 else 'gray'
                                    st.markdown(f"""
                                        ### Overall Market Sentiment
                                        <div style='text-align: center; color: {sentiment_color}; font-size: 24px; padding: 20px;'>
                                            {
                                                '🐻 Bearish' if avg_sentiment < -0.2 
                                                else '🐂 Bullish' if avg_sentiment > 0.2 
                                                else '😐 Neutral'
                                            }
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                # Display recent news with sentiment
                                st.markdown("### Recent News")
                                for article in valid_news[:5]:
                                    try:
                                        sentiment = TextBlob(article.get('title', '')).sentiment.polarity
                                        sentiment_icon = "🟢" if sentiment > 0.2 else "🔴" if sentiment < -0.2 else "⚪"
                                        
                                        # Safely get and format timestamp
                                        try:
                                            published = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                                            published_str = published.strftime('%Y-%m-%d %H:%M:%S')
                                        except:
                                            published_str = "Recent"
                                        
                                        st.markdown(f"""
                                            #### {sentiment_icon} {article.get('title', 'No Title')}
                                            *{published_str}*
                                            
                                            {article.get('summary', 'No summary available.')}
                                            
                                            [Read More]({article.get('link', '#')})
                                            ---
                                        """)
                                    except Exception as e:
                                        continue
                            else:
                                st.info("Unable to analyze sentiment due to insufficient news data.")
                        
                        except Exception as e:
                            st.info("Unable to process news data. Please try again later.")
                        
                except Exception as e:
                    st.info("""
                    ### News Temporarily Unavailable
                    
                    The news feed is currently unavailable. This might be due to:
                    - API rate limits
                    - Connection issues
                    - Service maintenance
                    
                    Please try again in a few minutes.
                    """)

            with tab7:
                st.subheader("Smart Investment Analysis")
                
                try:
                    # Get all necessary data
                    info = stock.info
                    current_price = df['Close'].iloc[-1]
                    returns = df['Close'].pct_change()
                    volatility = returns.std() * np.sqrt(252)
                    rsi = df['RSI'].iloc[-1]
                    
                    # Calculate scores for different factors (0-100)
                    scores = {}
                    
                    # 1. Technical Indicators Score
                    tech_score = 0
                    if df['Close'].iloc[-1] > df['MA50'].iloc[-1]:
                        tech_score += 30  # Above MA50
                    if df['Close'].iloc[-1] > df['MA20'].iloc[-1]:
                        tech_score += 20  # Above MA20
                    if 30 <= rsi <= 70:
                        tech_score += 25  # Healthy RSI
                    elif rsi < 30:
                        tech_score += 15  # Oversold
                    scores['Technical'] = tech_score
                    
                    # 2. Fundamental Score
                    fund_score = 0
                    pe_ratio = info.get('forwardPE', None)
                    if pe_ratio:
                        if 10 <= pe_ratio <= 25:
                            fund_score += 30
                        elif pe_ratio < 10:
                            fund_score += 20
                    if info.get('dividendYield', 0):
                        fund_score += 20
                    if info.get('beta', 0) and info.get('beta') < 1.5:
                        fund_score += 25
                    scores['Fundamental'] = fund_score
                    
                    # 3. Risk Score (higher means lower risk)
                    risk_score = 100
                    if volatility > 0.4:  # High volatility
                        risk_score -= 40
                    if abs(returns.iloc[-1]) > 0.05:  # Recent high movement
                        risk_score -= 20
                    scores['Risk'] = max(0, risk_score)
                    
                    # 4. Sentiment Score
                    sentiment_score = 50  # Neutral base
                    if 'avg_sentiment' in locals():
                        sentiment_score += avg_sentiment * 50
                    scores['Sentiment'] = max(0, min(100, sentiment_score))
                    
                    # Calculate overall score
                    overall_score = sum(scores.values()) / len(scores)
                    
                    # Display overall recommendation
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"""
                            ### Overall Score
                            <div style='text-align: center; font-size: 48px; font-weight: bold; 
                                      color: {'green' if overall_score >= 70 else 'orange' if overall_score >= 50 else 'red'}'>
                                {overall_score:.1f}%
                            </div>
                        """, unsafe_allow_html=True)
                        
                        recommendation = (
                            "Strong Buy" if overall_score >= 80
                            else "Buy" if overall_score >= 70
                            else "Hold" if overall_score >= 50
                            else "Sell" if overall_score >= 30
                            else "Strong Sell"
                        )
                        
                        st.markdown(f"""
                            <div style='text-align: center; font-size: 24px; margin-top: 10px;'>
                                Recommendation: <span style='color: {'green' if overall_score >= 70 else 'orange' if overall_score >= 50 else 'red'}'>
                                    {recommendation}
                                </span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Display individual scores
                        for factor, score in scores.items():
                            st.markdown(f"""
                                <div style='margin-bottom: 10px;'>
                                    <div style='display: flex; justify-content: space-between;'>
                                        <span>{factor} Score:</span>
                                        <span>{score:.1f}%</span>
                                    </div>
                                    <div style='background-color: #ddd; height: 20px; border-radius: 10px;'>
                                        <div style='background-color: {"green" if score >= 70 else "orange" if score >= 50 else "red"}; 
                                                  width: {score}%; height: 100%; border-radius: 10px;'>
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Key Factors Analysis
                    st.markdown("### Key Decision Factors")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Positive Factors ✅")
                        positive_factors = []
                        if df['Close'].iloc[-1] > df['MA50'].iloc[-1]:
                            positive_factors.append("Price above 50-day moving average")
                        if 30 <= rsi <= 70:
                            positive_factors.append("RSI in healthy range")
                        if info.get('dividendYield', 0):
                            positive_factors.append("Pays dividends")
                        if scores['Sentiment'] > 60:
                            positive_factors.append("Positive market sentiment")
                        
                        for factor in positive_factors:
                            st.markdown(f"- {factor}")
                    
                    with col2:
                        st.markdown("#### Risk Factors ⚠️")
                        risk_factors = []
                        if volatility > 0.3:
                            risk_factors.append(f"High volatility ({volatility:.1%})")
                        if rsi > 70:
                            risk_factors.append("Potentially overbought (RSI > 70)")
                        if rsi < 30:
                            risk_factors.append("Potentially oversold (RSI < 30)")
                        if pe_ratio and pe_ratio > 25:
                            risk_factors.append("High P/E ratio")
                        
                        for factor in risk_factors:
                            st.markdown(f"- {factor}")
                    
                    # Investment Considerations
                    st.markdown("""
                    ### Investment Considerations
                    
                    This analysis is based on multiple factors including:
                    - Technical indicators (Moving averages, RSI)
                    - Fundamental metrics (P/E ratio, dividend yield)
                    - Market sentiment and news analysis
                    - Risk metrics (volatility, beta)
                    
                    **Remember:**
                    - Past performance doesn't guarantee future results
                    - Always diversify your portfolio
                    - Consider your investment timeline and risk tolerance
                    - Consult with a financial advisor for personalized advice
                    """)
                    
                except Exception as e:
                    st.error(f"Error in smart analysis: {str(e)}")

            with tab8:
                st.subheader("Strategy Backtesting")
                
                try:
                    # Calculate signals for different strategies
                    def calculate_signals(df):
                        signals_df = pd.DataFrame(index=df.index)
                        
                        # 1. Moving Average Crossover
                        signals_df['MA_Signal'] = np.where(df['MA20'] > df['MA50'], 1, -1)
                        
                        # 2. RSI Strategy
                        signals_df['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
                        
                        # 3. MACD Strategy (simplified)
                        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
                        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
                        signals_df['MACD_Signal'] = np.where(df['MACD'] > df['Signal_Line'], 1, -1)
                        
                        return signals_df
                    
                    # Backtest function
                    def backtest_strategy(df, signals, initial_capital=100000):
                        position = 0
                        balance = initial_capital
                        trades = []
                        
                        for i in range(1, len(df)):
                            if signals[i] == 1 and position <= 0:  # Buy signal
                                shares = balance / df['Close'][i]
                                position = shares
                                trades.append({
                                    'date': df.index[i],
                                    'type': 'BUY',
                                    'price': df['Close'][i],
                                    'shares': shares,
                                    'balance': balance
                                })
                            elif signals[i] == -1 and position > 0:  # Sell signal
                                balance = position * df['Close'][i]
                                position = 0
                                trades.append({
                                    'date': df.index[i],
                                    'type': 'SELL',
                                    'price': df['Close'][i],
                                    'shares': shares,
                                    'balance': balance
                                })
                        
                        # Final value if still holding
                        if position > 0:
                            balance = position * df['Close'][-1]
                        
                        return balance, trades
                    
                    # Calculate signals
                    signals_df = calculate_signals(df)
                    
                    # Run backtests
                    strategies = {
                        'Moving Average Crossover': signals_df['MA_Signal'],
                        'RSI Strategy': signals_df['RSI_Signal'],
                        'MACD Strategy': signals_df['MACD_Signal']
                    }
                    
                    initial_capital = 100000  # $100,000 initial investment
                    results = {}
                    
                    for strategy_name, signals in strategies.items():
                        final_balance, trades = backtest_strategy(df, signals, initial_capital)
                        returns = (final_balance - initial_capital) / initial_capital
                        results[strategy_name] = {
                            'final_balance': final_balance,
                            'returns': returns,
                            'trades': trades
                        }
                    
                    # Display results
                    st.markdown("### Strategy Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    for (strategy_name, result), col in zip(results.items(), [col1, col2, col3]):
                        with col:
                            roi_color = "green" if result['returns'] > 0 else "red"
                            st.markdown(f"""
                                #### {strategy_name}
                                <div style='text-align: center;'>
                                    <div style='font-size: 24px; color: {roi_color};'>
                                        {result['returns']:.1%}
                                    </div>
                                    <div>Return</div>
                                    <div style='font-size: 18px;'>
                                        ${result['final_balance']:,.2f}
                                    </div>
                                    <div>Final Balance</div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Plot strategy comparisons
                    fig = go.Figure()
                    
                    # Buy & Hold strategy for comparison
                    buy_hold_return = (df['Close'][-1] / df['Close'][0] - 1)
                    
                    # Add traces for each strategy
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Close'] / df['Close'][0],
                        name='Buy & Hold',
                        line=dict(color='gray')
                    ))
                    
                    for strategy_name, result in results.items():
                        strategy_returns = []
                        current_value = 1.0
                        
                        for trade in result['trades']:
                            if trade['type'] == 'SELL':
                                current_value *= (1 + (trade['balance'] - initial_capital) / initial_capital)
                            strategy_returns.append(current_value)
                        
                        if strategy_returns:
                            fig.add_trace(go.Scatter(
                                x=[trade['date'] for trade in result['trades']],
                                y=strategy_returns,
                                name=strategy_name
                            ))
                    
                    fig.update_layout(
                        title='Strategy Performance Comparison',
                        yaxis_title='Return Multiple',
                        xaxis_title='Date',
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Strategy explanations
                    st.markdown("""
                    ### Strategy Explanations
                    
                    1. **Moving Average Crossover**
                       - Buys when 20-day MA crosses above 50-day MA
                       - Sells when 20-day MA crosses below 50-day MA
                       - Good for trending markets
                    
                    2. **RSI Strategy**
                       - Buys when RSI goes below 30 (oversold)
                       - Sells when RSI goes above 70 (overbought)
                       - Good for ranging markets
                    
                    3. **MACD Strategy**
                       - Buys when MACD line crosses above Signal line
                       - Sells when MACD line crosses below Signal line
                       - Good for identifying momentum changes
                    
                    **Note:** Past performance does not guarantee future results. These strategies are for educational purposes only.
                    """)
                    
                except Exception as e:
                    st.error(f"Error in backtesting analysis: {str(e)}")  

if __name__ == "__main__":
    main()
