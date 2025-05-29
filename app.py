import streamlit as st
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go

@st.cache_data
def fetch_series(code, source='fred', start='2000-01-01', end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    try:
        df = web.DataReader(code, source, start, end)
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching {code}: {e}")
        return pd.Series(dtype=float)

st.set_page_config(page_title="MacroInsight Dashboard", layout="wide")
st.title("📈 MacroInsight Dashboard")

with st.sidebar:
    st.header("Controls")
    series_options = {"GDP": "GDP", "Unemployment Rate": "UNRATE", "CPI (Inflation)": "CPIAUCSL"}
    selected = st.multiselect("Select indicators:", list(series_options.keys()), default=["GDP"])
    start_date = st.date_input("Start date", datetime(2000, 1, 1))
    end_date = st.date_input("End date", datetime.today())
    horizon = st.slider("Forecast horizon (periods):", 1, 36, 12)
    st.markdown("---")
    p = st.number_input("AR(p):", 0, 5, 1)
    d = st.number_input("I(d):", 0, 2, 1)
    q = st.number_input("MA(q):", 0, 5, 1)
    st.markdown("---")
    upload = st.file_uploader("Or upload your own CSV (date,index and value):", type=["csv"])

if upload:
    data = pd.read_csv(upload, parse_dates=[0], index_col=0)
    st.success("Custom data loaded")
else:
    data = pd.DataFrame()
    for name in selected:
        code = series_options[name]
        df = fetch_series(code, start=start_date.isoformat(), end=end_date.isoformat())
        if not df.empty:
            data[name] = df.iloc[:, 0]

if not data.empty and len(data.columns) > 0:
    st.markdown("## Latest Metrics")
    cols = st.columns(len(data.columns))
    for idx, col in enumerate(data.columns):
        with cols[idx]:
            val = data[col].iloc[-1]
            pct = (val - data[col].iloc[0]) / data[col].iloc[0] * 100
            st.metric(label=col, value=f"{val:.2f}", delta=f"{pct:.2f}%")
else:
    st.write("No data selected to display metrics.")

fig = go.Figure()
for col in data.columns:
    fig.add_trace(go.Scatter(x=data.index, y=data[col], mode='lines', name=col))

if not data.empty:
    primary = data.iloc[:, 0]
    last_date = primary.index[-1]
    freq = primary.index.freq or pd.infer_freq(primary.index) or 'M'
    with st.spinner("Fitting ARIMA model..."):
        model = ARIMA(primary, order=(p, d, q)).fit()
        pred = model.get_forecast(steps=horizon)
        fc_index = pd.date_range(start=last_date, periods=horizon+1, freq=freq)[1:]
        fc_mean = pd.Series(pred.predicted_mean, index=fc_index)
        fc_ci = pd.DataFrame(pred.conf_int(), index=fc_index, columns=['lower', 'upper'])
    fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, mode='lines', name='Forecast', line=dict(color='yellow', width=2)))
    fig.add_trace(go.Scatter(x=fc_ci.index, y=fc_ci['upper'], mode='lines', name='Upper CI', line=dict(dash='dash', width=2, color='firebrick')))
    fig.add_trace(go.Scatter(x=fc_ci.index, y=fc_ci['lower'], mode='lines', name='Lower CI', fill='tonexty', fillcolor='rgba(178,34,34,0.2)', line=dict(dash='dash', width=2, color='firebrick')))

fig.update_layout(title="Macro Series & ARIMA Forecast", xaxis_title="Date", yaxis_title="Value", legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True)

hist = data.copy()
if 'fc_mean' in locals():
    hist = hist.join(fc_mean.rename('Forecast'), how='outer')
    hist = hist.join(fc_ci, how='outer')
csv = hist.to_csv().encode('utf-8')
st.download_button("Download CSV", csv, "macro_data.csv", "text/csv")

st.markdown("---")
st.header("Interpretation & Guidance")
st.markdown(
    """
- **Blue lines** show historical values for each indicator.
- **Yellow line** shows the ARIMA forecast for the next periods.
- **Shaded red band** shows the 95% confidence interval.
- Adjust ARIMA parameters and forecast horizon to explore scenarios.
    """
)
