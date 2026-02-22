import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

st.set_page_config(page_title="Futures Multi‑Metric Signal Analytics", layout="wide")

# ------------------------------------------------------------
# 1. File Upload
# ------------------------------------------------------------
st.sidebar.title("📂 Upload Intraday Futures CSVs")
uploaded_files = st.sidebar.file_uploader(
    "Drag‑drop CSVs (5‑minute interval)", type="csv", accept_multiple_files=True
)
if not uploaded_files:
    st.warning("👋 Upload at least one CSV.")
    st.stop()

# ------------------------------------------------------------
# 2. Read & parse
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
if "expiryDate" not in first_df.columns:
    st.error("❌ Column 'expiryDate' not found.")
    st.stop()

expiry_opts = sorted(first_df["expiryDate"].unique())
expiry = st.selectbox("Select expiry", expiry_opts)

# ------------------------------------------------------------
# 4. Merge datasets for that expiry
# ------------------------------------------------------------
filtered = []
for i, df_file in enumerate(dfs):
    sub = df_file[df_file["expiryDate"] == expiry].copy()
    if sub.empty:
        continue
    sub["label"] = upload_labels[i]
    sub["capture_time"] = upload_times[i]
    filtered.append(sub)

if not filtered:
    st.warning("No data for chosen expiry.")
    st.stop()

final_df = pd.concat(filtered).sort_values(["contract", "timestamp"]).reset_index(drop=True)

# ------------------------------------------------------------
# 5. Utility function for metric analytics
# ------------------------------------------------------------
def compute_indicators(df: pd.DataFrame, metric: str):
    if metric not in df.columns:
        st.warning(f"Missing column '{metric}' in data.")
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
    out[f"Cum_{metric}"] = out[f"Δ {metric}"].cumsum()

    # Relative ratio & oscillator
    N = 5
    out["Ratio"] = out[f"Δ {metric}"] / out[f"Δ {metric}"].rolling(N, min_periods=1).mean()
    out["Osc"] = (
        out[f"Δ {metric}"].ewm(span=3, adjust=False).mean()
        - out[f"Δ {metric}"].ewm(span=10, adjust=False).mean()
    )
    out["RollCorr"] = out["Δ Price"].rolling(5).corr(out[f"Δ {metric}"])

    mu, sig = out[f"Δ {metric}"].mean(), out[f"Δ {metric}"].std()
    out["spike_flag"] = out[f"Δ {metric}"] > (mu + 2 * sig)
    out["SMA_Δ"] = out[f"Δ {metric}"].rolling(5, min_periods=1).mean().round(2)

    # classification
    def classify(row):
        slope = np.sign(row[f"Δ {metric}"]) or 0
        corr = np.sign(row["RollCorr"]) if not np.isnan(row["RollCorr"]) else 0
        score = slope * corr
        return 1 if score > 0 else (-1 if score < 0 else 0)
    out["Signal_Val"] = out.apply(classify, axis=1)
    out["Signal_Label"] = out["Signal_Val"].map({1: "🟢 Bullish", 0: "⚪ Neutral", -1: "🔴 Bearish"})

    def describe_ratio(x):
        return "🚀 High" if x > 1.5 else ("🧊 Low" if x < 0.7 else "⚪ Normal")
    def describe_osc(x):
        return "🟢 Up" if x > 0 else ("🔴 Down" if x < 0 else "⚪ Flat")
    out["Ratio_Signal"] = out["Ratio"].apply(describe_ratio)
    out["Osc_Signal"] = out["Osc"].apply(describe_osc)

    return out[
        [
            "time", f"Δ {metric}", "Δ Price",
            "Ratio", "Ratio_Signal",
            "Osc", "Osc_Signal",
            "RollCorr", "spike_flag", "SMA_Δ",
            "Signal_Label",
        ]
    ]

# ------------------------------------------------------------
# 6. Run computations
# ------------------------------------------------------------
vol_df = compute_indicators(final_df, "volume")
oi_df = compute_indicators(final_df, "openInterest")
turn_df = compute_indicators(final_df, "totalTurnover")

# ------------------------------------------------------------
# 7. Show individual tables
# ------------------------------------------------------------
st.subheader("📊 Volume‑based Indicators")
st.dataframe(vol_df)

st.subheader("📈 Open Interest‑based Indicators")
st.dataframe(oi_df)

st.subheader("💰 Total Turnover‑based Indicators")
st.dataframe(turn_df)

# ------------------------------------------------------------
# 8. Combined Summary Table — now includes Ratio + Osc signals
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

st.subheader("🪄 Combined Signal Summary (Vol / OI / Turnover)")
st.dataframe(combined)

# ------------------------------------------------------------
# 9. Overall Turnover bias signal
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
