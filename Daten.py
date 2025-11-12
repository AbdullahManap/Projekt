import requests
import json
from datetime import datetime, timedelta

# API-URL
url = "https://satellite-api.open-meteo.com/v1/archive?latitude=51.0267&longitude=7.5693&hourly=direct_normal_irradiance,diffuse_radiation,direct_radiation&models=satellite_radiation_seamless&timezone=Europe%2FBerlin&utm_source=chatgpt.com"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    hourly = data["hourly"]

    zeiten = hourly["time"]
    direkte_normale_strahlung = [x if x is not None else 0.0 for x in hourly["direct_normal_irradiance"]]
    diffuse_strahlung = [x if x is not None else 0.0 for x in hourly["diffuse_radiation"]]
    direkte_strahlung = [x if x is not None else 0.0 for x in hourly["direct_radiation"]]

    # Aktuelle Zeit und Grenze (24h zurück)
    jetzt = datetime.now()
    grenze = jetzt - timedelta(hours=24)

    # Gefilterte Werte
    werte_24h = []
    for i in range(len(zeiten)):
        zeit = datetime.fromisoformat(zeiten[i])
        if grenze <= zeit <= jetzt:
            werte_24h.append({
                "time": zeiten[i],
                "direct_normal_irradiance": direkte_normale_strahlung[i],
                "diffuse_radiation": diffuse_strahlung[i],
                "direct_radiation": direkte_strahlung[i]
            })

    # Ausgabe
    print(f"=== Werte der letzten 24 Stunden (bis {jetzt.strftime('%Y-%m-%d %H:%M')} Uhr) ===")
    print(json.dumps(werte_24h, indent=2))

else:
    print(f"Fehler {response.status_code}: {response.text}")
