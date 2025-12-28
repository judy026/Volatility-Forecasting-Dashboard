import streamlit as st
import numpy as np
import pandas as pd
import keras
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.stats import t

st.set_page_config(layout="wide")
st.title("Integrated Volatility Forecasting Dashboard")




# ----- Data Loading -----
st.sidebar.header("Stock Selection")
st.sidebar.subheader("Live Ticker Search")
popular_tickers = [
    "HDFCBANK.NS", "RELIANCE.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS", "AAPL", "MSFT", "GOOGL",
    "AMZN", "TSLA", "META", "NFLX", "NVDA", "JPM", "BAC", "XOM", "TATAMOTORS.NS", "SBIN.NS"
]
selected_ticker = st.sidebar.selectbox(
    "Search or select ticker", options=popular_tickers
)
st.sidebar.success(f"Selected Ticker : {selected_ticker}")
start = st.sidebar.date_input("Start Date",pd.to_datetime("2008-01-01"))
end = st.sidebar.date_input("End Date",datetime.today().date())
data = yf.download(selected_ticker,start=start,end=end)
returns = 100*np.log(data['Close']/data['Close'].shift(1)).dropna()
#st.write(f"Raw Returns : {returns}")
# st.subheader("Raw Returns")
# st.line_chart(returns)







# ----- Statistical Tests -----
st.subheader("Statistical Tests")

def run_tests(series):
    adf = adfuller(series)
    kpss_test = kpss(series,regression='c',nlags="auto")
    jb = jarque_bera(series)
    lb = acorr_ljungbox(series,lags=[20],return_df=True)
    
    return{
        "ADF p-value" : adf[1],
        "KPSS p-value" : kpss_test[1],
        "Jarque-Bera p-value" : jb[1],
        "LJung-Box p-value" : lb['lb_pvalue'].values[0]
    }

test_results = run_tests(returns)
st.json(test_results)







# ----- Rosner's Test (Outliers) -----
st.subheader("Rosner's Test (Outlier Detection)")
z_scores = (returns - returns.mean())/returns.std()
outliers = returns[np.abs(z_scores) > 3]
st.write(f"Detected Outliers : {len(outliers)}")






# ----- Forecast Metrics (RMSE, MAE, MAPE) -----
def forecast_metrics(true,pred):
    rmse = np.sqrt(mean_squared_error(true,pred))
    mae = mean_absolute_error(true,pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return rmse, mae, mape
    
    
    
    

# ----- STL Decomposition -----
st.subheader("STL Decomposition")
stl = STL(returns,period=252)
res = stl.fit()
fig = res.plot()
st.pyplot(fig)
residuals = res.resid.dropna()






# ----- ARIMA Model -----
st.subheader("ARIMA Forecast")
arima = ARIMA(residuals,order=(1,0,1)).fit()
arima_forecast = arima.forecast(30)
st.line_chart(arima_forecast)

true_vals = residuals[-30:]
rmse, mae, mape = forecast_metrics(true_vals,arima_forecast)
st.write("ARIMA Accuracy")
st.metric("RMSE",round(rmse,4))
# st.metric("MAE",round(mae,4))
# st.metric("MAPE (%)",round(mape,2))







# ----- SARIMA Model -----
st.subheader("SARIMA Forecast")
sarima = SARIMAX(residuals,order=(1,0,1),seasonal_order=(1,0,1,12)).fit(disp=False)
sarima_forecast = sarima.forecast(30)
st.line_chart(sarima_forecast)

true_vals = residuals[-30:]
rmse, mae, mape = forecast_metrics(true_vals,sarima_forecast)
st.write("SARIMA Accuracy")
st.metric("RMSE",round(rmse,4))
# st.metric("MAE",round(mae,4))
# st.metric("MAPE (%)",round(mape,2))








# ----- ARCH and GARCH Models -----
st.subheader("ARCH and GARCH Volatility")

arch_mod = arch_model(residuals,vol='ARCH',p=1)
arch_res = arch_mod.fit(disp='off')

garch_mod = arch_model(residuals,vol='GARCH',p=1,q=1)
garch_res = garch_mod.fit(disp='off')

st.write("ARCH Forecast")
st.line_chart(arch_res.conditional_volatility)
st.write("GARCH Forecast")
st.line_chart(garch_res.conditional_volatility)

# true_vals = residuals[-30:]
# rmse, mae, mape = forecast_metrics(true_vals,arch_res)

# st.write("ARCH Accuracy")
# st.metric("RMSE",round(rmse,4))
# st.metric("MAE",round(mae,4))
# st.metric("MAPE (%)",round(mape,2))








# ----- LSTM Model -----
st.subheader("LSTM Volatility Forecast")
scaler = MinMaxScaler()
scaled = scaler.fit_transform(residuals.values.reshape(-1,1))

X,y = [],[]
for i in range(20,len(scaled)):
    X.append(scaled[i-20:i])
    y.append(scaled[i])

X,y = np.array(X), np.array(y)

model = Sequential([
    LSTM(50,return_sequences=False,input_shape=(X.shape[1],1)), Dense(1)
])

model.compile(optimizer='adam',loss='mse')
model.fit(X,y,epochs=20,batch_size=32,verbose=0)

lstm_pred = model.predict(X[-30:])
lstm_pred = scaler.inverse_transform(lstm_pred)
st.line_chart(lstm_pred.flatten())

true_vals = residuals[-30:]
rmse, mae, mape = forecast_metrics(true_vals,lstm_pred.flatten())

st.write("LSTM Accuracy")
st.metric("RMSE",round(rmse,4))
# st.metric("MAE",round(mae,4))
# st.metric("MAPE (%)",round(mape,2))








# ----- Ensemble Forecast -----
ensemble_forecast = (
    arima_forecast.values + sarima_forecast.values + lstm_pred.flatten() + garch_res.forecast(horizon=30).variance.values[-1]**0.5
) / 4
st.subheader("Ensemble Volatility Forecast")
st.line_chart(ensemble_forecast)







# ----- Tabs + Comparison -----
tab1, tab2, tab3, tab4 = st.tabs([
    "Statistical Tests",
    "Individual Models",
    "Ensemble Forecast",
    "Model Comparison"
])

with tab1:
    st.json(test_results)
    
with tab2:
    st.write("ARIMA Forecast")
    st.line_chart(arima_forecast)
    st.write("SARIMA Forecast")
    st.line_chart(sarima_forecast)
    st.write("LSTM Forecast")
    st.line_chart(lstm_pred.flatten())
    
with tab3:
    st.line_chart(ensemble_forecast)
    
with tab4:
    comparison_df = pd.DataFrame({
        "ARIMA" : arima_forecast.values,
        "SARIMA" : sarima_forecast.values,
        "LSTM" : lstm_pred.flatten(),
        "Ensemble" : ensemble_forecast
    })
    st.line_chart(comparison_df)
    





# ----- Sector Volatility Heatmap -----
sector_stocks = {
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "JPM", "BAC"],
    "IT": ["INFY.NS", "TCS.NS", "AAPL", "MSFT", "GOOGL"],
    "Energy": ["RELIANCE.NS", "XOM"],
    "Automobile": ["TATAMOTORS.NS", "TSLA"],
    "Technology": ["META", "NFLX", "NVDA", "AMZN"]
}
st.subheader("Sector-wise Volatility Heatmap")
sector_volatility = {}
for sector, tickers in sector_stocks.items():
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False
        )["Close"]
        returns = 100 * np.log(data / data.shift(1))
        vol = returns.std()
        sector_volatility[sector] = vol.mean()
    except Exception as e:
        st.warning(f"Skipping {sector}: {e}")

sector_vol_df = pd.DataFrame.from_dict(
    sector_volatility,
    orient="index",
    columns=["Average Volatility"]
)

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    sector_vol_df,
    annot=True,
    cmap="RdYlGn_r",
    linewidths=0.5,
    cbar_kws={"label": "Volatility"}
)
ax.set_title("Sector-wise Average Volatility")
st.pyplot(fig)


# sector_heatmap_map = {
#     "Banking" : ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
#     "Technology" : ["INFY.NS", "TCS.NS", "AAPL", "MSFT", "META", "NFLX", "GOOGL"],
#     "Automobile" : ["TSLA", "TATAMOTORS.NS"],
#     "Energy" : ["RELIANCE.NS", "XOM", "AMZN", "NVDA"],
#     "Finance (US)" : ["JPM", "BAC"]
# }
# data = yf.download(selected_ticker,start=start,end=end)["Close"]
# returns = np.log(data / data.shift(1)) * 100
# volatility = returns.std()

# close = data['Close']
# returns = np.log(close / close.shift(1)) * 100
# windows = [5, 10, 20]
# vol_df = pd.DataFrame()
# for w in windows:
#     vol_df[f"{w}-Day"] = returns.rolling(w).std()
# vol_df = vol_df.dropna()

# fig, ax = plt.subplots(figsize=(12, 4))
# sns.heatmap(
#     vol_df.T,
#     cmap="RdYlGn_r",
#     cbar_kws={"label": "Volatility"},
#     ax=ax
# )
# ax.set_xlabel("Date")
# ax.set_ylabel("Rolling Window")
# st.pyplot(fig)
# # sector_volatility = {}
# # for sector, tickers in sector_heatmap_map.items():
# #     vols = []
# #     for tkr in tickers:
# #         df = yf.download(tkr,start=start,end=end,progress=False)
# #         returns = 100*np.log(df['Close'] / df['Close'].shift(1)).dropna()
# #         vols.append(returns.std())
# #     sector_volatility[sector] = np.mean(vols)
# # heatmap_df = pd.DataFrame.from_dict(sector_volatility,orient="index",columns=["Volatility"])

# # st.subheader("Sector-wise Volatility Heatmap")
# # st.dataframe(
# #     heatmap_df.style.background_gradient(cmap="Reds",subset=["Volatility"])
# # )






# ----- Intraday Volatility -----
st.sidebar.subheader("Intraday Settings")
interval = st.sidebar.selectbox(
    "Select Intraday Interval",
    ["5m", "15m", "30m", "60m"]
)
intraday_data = yf.download(selected_ticker,period="30d",interval=interval,progress=False)
intraday_returns = 100*np.log(intraday_data['Close'] / intraday_data['Close'].shift(1)).dropna()
intraday_vol = intraday_returns.rolling(20).std()
st.subheader(f"Intraday Volatility ({interval})")
st.line_chart(intraday_vol)
st.write("Intraday Volatility Statistics")
st.write(intraday_vol.describe())





st.success("Forecasting Completed Successfully")