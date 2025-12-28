# Integrated Volatility Forecasting Dashboard
A Streamlit-based postgraduate project that integrates statistical, econometric, and deep learning models to analyze and forecast stock market volatility.

## Features
- Live stock selection (Yahoo Finance)
- Statistical tests: ADF, KPSS, Jarque-Bera, Ljung-Box
- STL decomposition
- ARIMA & SARIMA forecasting
- ARCH & GARCH volatility modeling
- LSTM-based volatility forecasting
- Ensemble forecasting
- Intraday volatility analysis
- Interactive Streamlit dashboard

## Models Used
- ARIMA
- SARIMA
- ARCH
- GARCH
- LSTM
- Ensemble methods

## Data Source
- Yahoo Finance (via yfinance)

## Technologies
- Python
- Streamlit
- TensorFlow / Keras
- Statsmodels
- Scikit-learn

## How to Run
```bash
py -m pip install -r requirements.txt
py -m streamlit run volatility_forecasting_app.py