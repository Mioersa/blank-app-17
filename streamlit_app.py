import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Futures Multi‑Metric Analytics", layout="wide")

# ------------------------------------------------------------
# 1. Upload CSVs
# ------------------------------------------------------------
st.sidebar.title("📂 Upload Intraday Futures CSVs")
uploaded_files = st.sidebar.file_uploader(
    "Drag‑drop CSVs (5‑minute interval)",
    type="csv",
    accept_multiple_files=True,
)
if not uploaded_files:
    st.warning("👋 Upload at least one CSV.")
    st.stop()

# ------------------------------------------------------------
# 2. Read files
# ------------------------------------------------------------
dfs, upload_times, upload_labels = [], [], []
for uploaded in uploaded_files:
    fn = uploaded.name
    m = re.search(r"_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})\.csv$", fn)
    base_time, label = datetime.now(), "unknown"
    if m:
        dd, mm, yyyy, HH, MM, SS = m.groups()
        base_time = datetime(int(yyyy), int(mm), int(dd), int(HH), int(MM), int(SS))
        label = f"{HH}:{MM}"
    upload_times.append(base_time)
    upload_labels.append(label)

    df = pd.read_csv(uploaded)
    df["timestamp"] = base_time + pd.to_timedelta(np.arange(len(df)) * 5, unit="min")
    if "totalTurnover" in df:
        df["totalTurnover"] = df["totalTurnover"].round(2)
    dfs.append(df)

# ------------------------------------------------------------
# 3. Expiry selector
# ------------------------------------------------------------
first_df = dfs[0]
if "expiryDate" not in first_df:
    st.error("❌ Column 'expiryDate' not found.")
    st.stop()

expiry_opts = sorted(first_df["expiryDate"].unique())
expiry = st.selectbox("Select expiry", expiry_opts)

filtered = []
for i, df_file in enumerate(dfs):
    sub = df_file[df_file["expiryDate"] == expiry].copy()
    if sub.empty:
        continue
    sub["label"] = upload_labels[i]
    sub["capture_time"] = upload_times[i]
    filtered.append(sub)
if not filtered:
    st.stop()

final_df = pd.concat(filtered).sort_values(["contract", "timestamp"]).reset_index(drop=True)

# ------------------------------------------------------------
# 4. Compute per‑metric indicators
# ------------------------------------------------------------
def compute_indicators(df: pd.DataFrame, metric: str):
    if metric not in df.columns:
        return pd.DataFrame()

    records = []
    for lbl in upload_labels:
        sub = df[df["label"] == lbl]
        if sub.empty:
            continue
        val = sub[metric].iloc[0]
        price = sub["lastPrice"].iloc[-1] if "lastPrice" in sub else np.nan
        records.append({"time": lbl, metric: val, "last_price": price})

    out = pd.DataFrame(records)
    out[f"Δ {metric}"] = out[metric].diff()
    out["Δ Price"] = out["last_price"].diff()
    out["Ratio"] = out[f"Δ {metric}"] / out[f"Δ {metric}"].rolling(5, min_periods=1).mean()
    out["Osc"] = (
        out[f"Δ {metric}"].ewm(span=3, adjust=False).mean()
        - out[f"Δ {metric}"].ewm(span=10, adjust=False).mean()
    )
    out["RollCorr"] = out["Δ Price"].rolling(5).corr(out[f"Δ {metric}"])
    out["SMA_Δ"] = out[f"Δ {metric}"].rolling(5, min_periods=1).mean().round(2)

    def classify(row):
        slope = np.sign(row[f"Δ {metric}"]) or 0
        corr = np.sign(row["RollCorr"]) if not np.isnan(row["RollCorr"]) else 0
        score = slope * corr
        return 1 if score > 0 else (-1 if score < 0 else 0)

    out["Signal_Val"] = out.apply(classify, axis=1)
    out["Signal_Label"] = out["Signal_Val"].map(
        {1: "🟢 Bullish", 0: "⚪ Neutral", -1: "🔴 Bearish"}
    )

    def describe_ratio(x):
        return "🚀 High" if x > 1.5 else ("🧊 Low" if x < 0.7 else "⚪ Normal")

    def describe_osc(x):
        return "🟢 Up" if x > 0 else ("🔴 Down" if x < 0 else "⚪ Flat")

    out["Ratio_Signal"] = out["Ratio"].apply(describe_ratio)
    out["Osc_Signal"] = out["Osc"].apply(describe_osc)

    return out[
        [
            "time",
            f"Δ {metric}",
            "Δ Price",
            "Ratio",
            "Ratio_Signal",
            "Osc",
            "Osc_Signal",
            "RollCorr",
            "SMA_Δ",
            "Signal_Label",
        ]
    ]

# ------------------------------------------------------------
# 5. Compute for all three metrics
# ------------------------------------------------------------
vol_df = compute_indicators(final_df, "volume")
oi_df = compute_indicators(final_df, "openInterest")
turn_df = compute_indicators(final_df, "totalTurnover")

# Helper for custom chart
def plot_custom_chart(df, column, title, color):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df[column],
            mode="lines+markers",
            line=dict(color=color),
            name=column,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=column,
        height=500,
        margin=dict(l=60, r=40, t=60, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 6. Display tables + custom plotters
# ------------------------------------------------------------
st.subheader("📊 Volume‑based Indicators")
st.dataframe(vol_df)
st.markdown("#### 📊 Volume‑based Column Plotter")
if not vol_df.empty:
    vol_cols = [c for c in vol_df.columns if c != "time"]
    selected_vol_col = st.selectbox("Select column (Volume Indicators)", vol_cols, key="volplot")
    if st.button("Plot Volume Chart"):
        plot_custom_chart(vol_df, selected_vol_col, f"{selected_vol_col} vs Time (Volume Indicators)", "firebrick")
else:
    st.info("No Volume indicator data available.")

# --- Open Interest section ---
st.subheader("📈 Open Interest‑based Indicators")
st.dataframe(oi_df)
st.markdown("#### 📊 OI‑based Column Plotter")
if not oi_df.empty:
    oi_cols = [c for c in oi_df.columns if c != "time"]
    selected_oi_col = st.selectbox("Select column (OI Indicators)", oi_cols, key="oiplot")
    if st.button("Plot OI Chart"):
        plot_custom_chart(oi_df, selected_oi_col, f"{selected_oi_col} vs Time (Open Interest Indicators)", "teal")
else:
    st.info("No OI indicator data available.")

# --- Turnover section ---
st.subheader("💰 Total Turnover‑based Indicators")
st.dataframe(turn_df)
st.markdown("#### 📊 Turnover‑based Column Plotter")
if not turn_df.empty:
    turn_cols = [c for c in turn_df.columns if c != "time"]
    selected_turn_col = st.selectbox("Select column (Turnover Indicators)", turn_cols, key="turnplot")
    if st.button("Plot Turnover Chart"):
        plot_custom_chart(turn_df, selected_turn_col, f"{selected_turn_col} vs Time (Turnover Indicators)", "darkorange")
else:
    st.info("No Turnover indicator data available.")

# ------------------------------------------------------------
# 7. Combined summary + overall check
# ------------------------------------------------------------
combined = pd.DataFrame({"time": vol_df["time"] if not vol_df.empty else None})
if not vol_df.empty:
    combined["Vol_Signal"] = vol_df["Signal_Label"]
    combined["Vol_Ratio"] = vol_df["Ratio_Signal"]
    combined["Vol_Osc"] = vol_df["Osc_Signal"]
if not oi_df.empty:
    combined["OI_Signal"] = oi_df["Signal_Label"]
    combined["OI_Ratio"] = oi_df["Ratio_Signal"]
    combined["OI_Osc"] = oi_df["Osc_Signal"]
if not turn_df.empty:
    combined["Turn_Signal"] = turn_df["Signal_Label"]
    combined["Turn_Ratio"] = turn_df["Ratio_Signal"]
    combined["Turn_Osc"] = turn_df["Osc_Signal"]

def overall_signal(row):
    vals = [row.get("Vol_Signal"), row.get("OI_Signal"), row.get("Turn_Signal")]
    if all(v == "🟢 Bullish" for v in vals): return "🟢 Bullish"
    if all(v == "🔴 Bearish" for v in vals): return "🔴 Bearish"
    return ""

def overall_ratio(row):
    vals = [row.get("Vol_Ratio"), row.get("OI_Ratio"), row.get("Turn_Ratio")]
    if all(v == "🚀 High" for v in vals): return "🚀 High"
    if all(v == "🧊 Low" for v in vals): return "🧊 Low"
    return ""

def overall_osc(row):
    vals = [row.get("Vol_Osc"), row.get("OI_Osc"), row.get("Turn_Osc")]
    if all(v == "🟢 Up" for v in vals): return "🟢 Up"
    if all(v == "🔴 Down" for v in vals): return "🔴 Down"
    return ""

combined["Overall_Signal"] = combined.apply(overall_signal, axis=1)
combined["Overall_Ratio"] = combined.apply(overall_ratio, axis=1)
combined["Overall_Osc"] = combined.apply(overall_osc, axis=1)

st.subheader("🪄 Combined Signal Summary (Vol / OI / Turnover + Overall)")
st.dataframe(combined)

# ------------------------------------------------------------
# 8. Default ΔVolume chart
# ------------------------------------------------------------
if not vol_df.empty:
    st.subheader("📈 Chart – Last Price (top) & Δ Volume (bottom)")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=vol_df["time"],
            y=vol_df["Δ volume"],
            name="Δ Volume",
            marker_color="orange",
            opacity=0.6,
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=vol_df["time"],
            y=vol_df["Δ Price"].cumsum(),
            mode="lines+markers",
            name="Last Price (proxy)",
            line=dict(color="blue"),
            yaxis="y1",
        )
    )
    fig.update_layout(
        height=600,
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(title="Capture Time (HH:MM)", rangeslider=dict(visible=True)),
        yaxis=dict(domain=[0.45, 1.0], title="Last Price"),
        yaxis2=dict(domain=[0.0, 0.35], title="Δ Volume"),
        hovermode="x unified",
        legend=dict(orientation="h"),
        title="Last Price & Δ Volume (precise numbers)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 9. Turnover Bias Summary
# ------------------------------------------------------------
if not turn_df.empty:
    last = turn_df.tail(5)
    slope = np.sign(last["SMA_Δ"].iloc[-1] - last["SMA_Δ"].iloc[0])
    corr_sign = np.sign(turn_df["RollCorr"].iloc[-1])
    score = slope * corr_sign
    if score > 0:
        result = "🟢 **Bullish Turnover Bias**"
    elif score < 0:
        result = "🔴 **Bearish Turnover Bias**"
    else:
        result = "⚪ **Neutral Turnover Bias**"
    st.subheader("📈 Overall Auto‑Signal (Last 5)")
    st.markdown(result)

# ------------------------------------------------------------
# 10. Custom Column Plotter for Combined Summary
# ------------------------------------------------------------
st.subheader("📊 Custom Column Plotter (Combined Summary)")
if not combined.empty:
    columns_to_plot = [c for c in combined.columns if c != "time"]
    selected_col = st.selectbox("Select column to plot", columns_to_plot, key="combinedplot")
    if st.button("Plot Combined Chart"):
        plot_custom_chart(combined, selected_col, f"{selected_col} vs Time (Combined Summary)", "purple")
else:
    st.info("No Combined Summary available.")
