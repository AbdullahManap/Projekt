import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import pydeck as pdk

# OpenWeatherMap API-Key
API_KEY = "5605fc8cf0d2b91d22a620881533b0c4"

# Koordinaten für Bergneustadt
lat = "51.0205"
lon = "7.6486"

# API-URLs
weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric&lang=de"
forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric&lang=de"
awattar_url = "https://api.awattar.de/v1/marketdata"

# --- Wetterdaten abrufen ---
def get_weather():
    try:
        response = requests.get(weather_url)
        response.raise_for_status()
        data = response.json()
        return {
            "description": data["weather"][0]["description"].capitalize(),
            "temperature": round(data["main"]["temp"]),
            "min_temperature": round(data["main"]["temp_min"]),
            "max_temperature": round(data["main"]["temp_max"]),
            "icon": data["weather"][0]["icon"],
            "raw": data
        }
    except:
        return None

# --- Wettervorhersage abrufen ---
def get_forecast():
    try:
        response = requests.get(forecast_url)
        response.raise_for_status()
        data = response.json()

        forecast_data = []
        for i in range(12):  # Nächste 4 Zeiträume (~12 Stunden)
            entry = data["list"][i]
            dt = datetime.fromtimestamp(entry["dt"])
            description = entry["weather"][0]["description"].capitalize()
            temp = round(entry["main"]["temp"])
            forecast_data.append({
                "time": dt.strftime("%d.%m. %H:%M"),
                "description": description,
                "temp": temp
            })
        return forecast_data
    except:
        return None

# --- Strompreise abrufen ---
def get_energy_prices():
    try:
        response = requests.get(awattar_url)
        response.raise_for_status()
        data = response.json()

        timestamps = []
        prices = []

        for entry in data["data"]:
            timestamps.append(datetime.fromtimestamp(entry["start_timestamp"]/1000))  # ms → s
            prices.append(entry["marketprice"] / 1000)  # Cent/MWh → €/kWh

        return timestamps, prices
    except:
        return None, None

# --- Streamlit Dashboard ---
st.set_page_config(page_title="Live Dashboard", layout="wide")

st.title("📊 Live Dashboard: Strompreise & Wetter")
st.write("🔄 Daten werden bei jedem Neustart aktualisiert.")

# --- Wetteranzeige ---
weather = get_weather()

if weather:
    st.subheader("🌤️ Aktuelles Wetter in Bergneustadt")

    col1, col2, col3 = st.columns(3)
    col1.metric("🌡️ Temperatur", f"{weather['temperature']}°C")
    col2.metric("📉 Min. Temp.", f"{weather['min_temperature']}°C")
    col3.metric("📈 Max. Temp.", f"{weather['max_temperature']}°C")
    st.write(f"🌍 Wetter: **{weather['description']}**")

   
# --- Wettervorhersage anzeigen ---
forecast = get_forecast()

if forecast:
    st.subheader("Wettervorhersage (nächste Stunden)")
    for entry in forecast:
        st.write(f"🕒 **{entry['time']}** — {entry['description']}, 🌡️ {entry['temp']}°C")
else:
    st.error("❌ Fehler beim Abrufen der Wettervorhersage!")

# --- Strompreise als Diagramm ---
timestamps, prices = get_energy_prices()

if timestamps and prices:
    st.subheader("⚡ Strompreise (€/kWh)")

    df = pd.DataFrame({
        "Zeit": timestamps,
        "Strompreis (€/kWh)": prices
    })
    df.set_index("Zeit", inplace=True)

    # Spalte für das Diagramm
    col1, col2, col3 = st.columns([1, 2, 1])  # Die mittlere Spalte ist breiter
    with col2:
        st.line_chart(df)
else:
    st.error("❌ Fehler beim Abrufen der Strompreisdaten!")

