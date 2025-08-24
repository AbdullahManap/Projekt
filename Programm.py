import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

st.title("Dynamische Strompreise")

# Daten abrufen
url = "https://api.awattar.de/v1/marketdata"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    timestamps = []
    prices = []

    for entry in data["data"]:
        timestamps.append(datetime.fromtimestamp(entry["start_timestamp"] / 1000))
        prices.append(entry["marketprice"] * 0.1)  # €/MWh → ct/kWh

    # Matplotlib-Figur erstellen
    fig, ax = plt.subplots(figsize=(14, 6))

    bar_width = 1/24  # Breite = 1 Stunde

    # Balkenplot
    bars = ax.bar(timestamps, prices, width=bar_width, color="skyblue", edgecolor="black", align='edge')

    # Werte über den Balken
    for bar, price in zip(bars, prices):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{price:.1f}", ha="center", va="center", fontsize=10)

    # Achsenformat
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.set_xlabel("Uhrzeit")
    ax.set_ylabel("Preis (ct/kWh)")
    ax.set_title("Dynamische Strompreise")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Plot in Streamlit anzeigen
    st.pyplot(fig)
else:
    st.error("Fehler beim Abrufen der Daten!")
