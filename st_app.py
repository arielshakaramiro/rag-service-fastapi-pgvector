import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Reaction Monitoring Dashboard")

reaction_type = st.selectbox("Pilih tipe reaksi:", ["like", "dislike", "regenerate"], index=1)
start_date = st.date_input("Tanggal mulai:", value=(datetime.now() - timedelta(days=30)).date())
end_date = st.date_input("Tanggal akhir:", value=datetime.now().date())
plot_type = st.selectbox("Jenis grafik:", ["Bar", "Line"])

if start_date > end_date:
    st.error("Tanggal akhir harus setelah tanggal mulai.")
    st.stop()


def fetch_reactions(reaction_type, start, end):
    params = {
        "reaction_type": reaction_type,
        "start_datetime": datetime.combine(start, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S"),
        "end_datetime": datetime.combine(end, datetime.max.time()).strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        resp = requests.get(f"{API_URL}/all-reactions", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()["reactions"]
    except requests.RequestException as e:
        st.error(f"Gagal mengambil data dari API: {e}")
        return None


reactions = fetch_reactions(reaction_type, start_date, end_date)

if reactions:
    df = pd.DataFrame(reactions)
    df["created_at"] = pd.to_datetime(df["created_at"])
    daily = df.groupby(df["created_at"].dt.normalize()).size()
    # Hari tanpa reaksi diisi 0, supaya grafik tidak menarik garis melintasi hari kosong
    full_range = pd.date_range(start_date, end_date, freq="D")
    daily = daily.reindex(full_range, fill_value=0).rename("jumlah")
    st.metric(f"Total {reaction_type}", int(daily.sum()))
    (st.bar_chart if plot_type == "Bar" else st.line_chart)(daily)
elif reactions is not None:
    st.warning("Tidak ada reaksi pada rentang tanggal ini.")
