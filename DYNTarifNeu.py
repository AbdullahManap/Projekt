import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Daten abrufen
url = "https://api.awattar.de/v1/marketdata"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()['data']  # Nur den relevanten Teil nehmen

    # Zeitstempel und Preise extrahieren
    times = [datetime.fromtimestamp(item['start_timestamp'] / 1000) for item in data]
    prices = [item['marketprice'] / 10 for item in data]  # in Cent/kWh umrechnen

    # Diagramm erstellen
    plt.figure(figsize=(10, 5))
    plt.plot(times, prices, marker='o', linestyle='-', label="Spotpreis (ct/kWh)")
    plt.title("aWATTar Strompreis (Deutschland)")
    plt.xlabel("Zeit")
    plt.ylabel("Preis (ct/kWh)")
    plt.grid(True)

    # Formatierung der Zeitachse
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.gcf().autofmt_xdate()

    plt.legend()
    plt.show()

else:
    print("Fehler beim Abrufen der Daten:", response.status_code)

for item in data:
    start = datetime.fromtimestamp(item['start_timestamp'] / 1000)
    price = item['marketprice'] / 10
    print(start.strftime("%d.%m.%Y %H:%M:%S"), f"{price:.2f} ct/kWh")

