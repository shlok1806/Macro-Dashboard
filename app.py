
import streamlit as st
import pandas_datareader.data as web
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import plotly.graph_objs as go


def fetch_series(code, source='fred', start='2000-01-01', end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    return web.DataReader(code, source, start, end)

def make_arima_forecast(series, order=(1,1,1), steps=12):
    model = ARIMA(series, order=order)
    fit = model.fit()
    fc = fit.get_forecast(steps=steps)
    return fc.predicted_mean, fc.conf_int()

st.sidebar.header("Choose indicator & forecast horizon")
indicator = st.sidebar.selectbox(
    "Indicator",
    ("GDP", "Unemployment Rate", "CPI")
)
horizon = st.sidebar.slider(
    "Forecast horizon (periods):",
    min_value=1, max_value=36, value=12
)
start_date = st.sidebar.date_input("Start date", datetime(2000,1,1))

fred_codes = {
    "GDP": "GDP",
    "Unemployment Rate": "UNRATE",
    "CPI": "CPIAUCSL"
}
series_code = fred_codes[indicator]

data = fetch_series(series_code, start=start_date.isoformat())
data = data.dropna()

forecast_mean, conf_int = make_arima_forecast(data, steps=horizon)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data.index, y=data.values.flatten(),
    name="Actual", mode="lines"
))
fig.add_trace(go.Scatter(
    x=forecast_mean.index, y=forecast_mean.values,
    name="Forecast", mode="lines"
))
fig.add_trace(go.Scatter(
    x=conf_int.index, y=conf_int.iloc[:,0],
    name="Lower CI", line=dict(dash="dash")
))
fig.add_trace(go.Scatter(
    x=conf_int.index, y=conf_int.iloc[:,1],
    name="Upper CI", line=dict(dash="dash")
))
fig.update_layout(
    title=f"{indicator} with {horizon}-period ARIMA Forecast",
    xaxis_title="Date", yaxis_title=indicator
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("""
**Interpretation:**  
- The blue line shows historical data.  
- The orange line is our {order} ARIMA forecast for the next {horizon} periods.  
- Dashed bands are the 95% confidence interval.  
""".format(order="(1,1,1)", horizon=horizon))
if st.sidebar.button("Download historical + forecast"):
    df_hist = pd.DataFrame(data.values, index=data.index, columns=[indicator])
    df_fc   = pd.DataFrame(forecast_mean.values, index=forecast_mean.index, columns=[f"{indicator} Forecast"])
    df_ci   = conf_int
    download_df = pd.concat([df_hist, df_fc, df_ci], axis=1)
    st.sidebar.download_button(
        "CSV",
        download_df.to_csv().encode('utf-8'),
        file_name="macro_series_with_forecast.csv"
    )