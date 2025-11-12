import requests
from datetime import datetime

API_key = "c154d26cd7087c7ef48de252fe2d6b03"
interval = "1h"
lat = "51.0267"
lon = "7.5693"
date = "2025-10-19"

url = f"https://api.openweathermap.org/energy/2.0/solar/interval_data?lat={lat}&lon={lon}&date={date}&interval={interval}&appid={API_key}"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(data)

    timestamps = []
    dni_values = []
    dhi_values = []

    for i in data["intervals"]:
        starte_time = i["start"]
        timestamps.append(f"{date} {starte_time}")
        dni_values.append(i["avg_irradiance"]["cloudy_sky"]["dni"])
        dhi_values.append(i["avg_irradiance"]["cloudy_sky"]["dhi"]) 

    print("Beispielausgabe:")
    for t, dni, dhi in zip(timestamps[:24], dni_values[:24], dhi_values[:24]):
        print(f"{t} → DNI_cloudy: {dni:.2f}, DHI_cloudy: {dhi:.2f}")
else:
    print("Fehler")

jetzt = datetime.now().date()
dt = datetime.strptime(str(jetzt), "%Y-%m-%d")
day_number = dt.timetuple().tm_yday

Zeit = datetime.now().time()
LokaleZeit = Zeit.strftime("%H:%M")

print("Lokale Zeit:", LokaleZeit)
print("Aktuelles Datum:", str(jetzt))
print("Tag im Jahr:", day_number)

zeitStunden = Zeit.hour
zeitMinuten = Zeit.minute
zeitInDezimal = round(zeitStunden + (zeitMinuten / 60),2)
print("Zeit in Dezimal:", zeitInDezimal)