# app.py
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
import pandas as pd

st.set_page_config(page_title="Strompreisoptimierung", layout="wide")
st.title("Dynamische Strompreise")

# Siedebar Eingaben
st.sidebar.header("Ort")

@st.cache_data(ttl=60 * 60)  # 1 Stunde cachen
def geocode_place(place: str):
    try:
        geolocator = Nominatim(user_agent="awattar_streamlit_app")
        loc = geolocator.geocode(place, addressdetails=True, language="de")
        if not loc:
            return None
        return {"lat": loc.latitude, "lon": loc.longitude, "name": loc.address}
    except (GeocoderServiceError, GeocoderTimedOut):
        return "timeout"
    except Exception:
        return None

with st.sidebar:
    place_input = st.text_input("Ort eingeben", value="Gummersbach")
    col_sb1, col_sb2 = st.columns([1, 1])
    with col_sb1:
        search_clicked = st.button("Suchen")
    with col_sb2:
        clear_clicked = st.button("Zurücksetzen")

    if clear_clicked:
        st.rerun()

    if search_clicked and place_input.strip():
        result = geocode_place(place_input.strip())
        if result == "timeout":
            st.error("Geocoding-Zeitüberschreitung. Bitte erneut versuchen.")
        elif result is None:
            st.warning("Ort nicht gefunden.")
        else:
            st.success(f"Gefunden: {result['name']}")
            st.write(
                f"**Breitengrad:** {result['lat']:.6f}  \n"
                f"**Längengrad:** {result['lon']:.6f}"
            )
            # Kleine Karte
            st.map(pd.DataFrame([{"lat": result["lat"], "lon": result["lon"]}]))

# PV-Anlagendaten in Sidebar
st.sidebar.header("PV-Anlage Parameter")

kapazitaet_ac = st.sidebar.number_input(
    "Kapazität AC (kW)", min_value=0.0, step=0.1, value=0.0
)
kapazitaet_dc = st.sidebar.number_input(
    "Kapazität DC Modul (kW)", min_value=0.0, step=0.1, value=0.0
)
azimuth = st.sidebar.number_input(
    "Ausrichtung (Azimuth)", min_value=0, max_value=360, step=1, value=0
)
winkel = st.sidebar.number_input(
    "Neigungswinkel (°)", min_value=0, max_value=90, step=1, value=0
)

# Dynamische Strompreise
@st.cache_data(ttl=15 * 60)  # 15 Minuten cachen
def fetch_market_data():
    url = "https://api.awattar.de/v1/marketdata"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    timestamps, prices = [], []
    for entry in data.get("data", []):
        timestamps.append(datetime.fromtimestamp(entry["start_timestamp"] / 1000))
        prices.append(entry["marketprice"] * 0.1)  # €/MWh → ct/kWh
    return timestamps, prices

try:
    timestamps, prices = fetch_market_data()
    if not timestamps:
        st.warning("Keine Daten erhalten.")
        st.stop()

    # Farben und Transparenz festlegen
    fill_color = "#8cd7f8"
    line_color = "#12b8ef"
    alpha = 0.25

    # Für die Treppenform: bis zum Ende der letzten Stunde erweitern
    last_end = timestamps[-1] + timedelta(hours=1)
    x_step = timestamps + [last_end]
    y_step = prices + [prices[-1]]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(x_step, y_step, step="post", alpha=alpha, color=fill_color)
    ax.step(x_step, y_step, where="post", linewidth=2.5, color=line_color)

    # Labels über jedem Plateau (Mitte der Stunde)
    for t, p in zip(timestamps, prices):
        ax.text(
            t + timedelta(minutes=30),
            p + 0.2,
            f"{p:.2f}".replace('.', ','),
            ha="center", va="bottom", fontsize=9
        )

    # Achsenformatierung
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=90)
    ax.set_xlim(timestamps[0], last_end)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Uhrzeit")
    ax.set_ylabel("Preis (ct/kWh)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    st.pyplot(fig, clear_figure=True)

    # Kennzahlen unten
    col1, col2, col3 = st.columns(3)
    # col1.metric("Aktuelle Stunde (ct/kWh)", f"{prices[-1]:.2f}".replace('.', ','))
    col2.metric("Minimum (ct/kWh)", f"{min(prices):.2f}".replace('.', ','))
    col3.metric("Maximum (ct/kWh)", f"{max(prices):.2f}".replace('.', ','))

except requests.RequestException as e:
    st.error(f"Fehler beim Abrufen der Daten: {e}")
