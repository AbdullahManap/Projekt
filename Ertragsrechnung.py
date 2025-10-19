import math
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt

#Datum und Standort
date = datetime.now().date()
dt = datetime.strptime(str(date), "%Y-%m-%d")
day_number = dt.timetuple().tm_yday
lat = 51.0267
lon = 7.5693
#Zeit
Zeitzone = 2.0
Zeit = datetime.now().time()
Zeitvoll = Zeit.replace(minute = 0, second = 0, microsecond = 0)
LokaleZeit = Zeitvoll.strftime("%H:%M")
zeitstunden = Zeit.hour
zeitminuten = Zeit.minute
zeitindezimal = round(zeitstunden + (zeitminuten / 60),2)
#PV-Anlagenparameter
Nennleistung = 5000  # in Watt
Azimut = 180
Neigungswinkel = 30
Albedo = 0.2

#API Abfrage
API_key = "c154d26cd7087c7ef48de252fe2d6b03"
interval = "1h"

values_direkt, values_diff, values_reflekt, Zeit, values_global, values_Ertrag = [], [], [], [], [], []

timestamps, dni_values, dhi_values = [], [], []
now = datetime.now()
now = now.replace(minute=0, second=0, microsecond=0)

for d in range(2):  # heute + morgen
    date = (now + timedelta(days=d)).strftime("%Y-%m-%d")
    url = f"https://api.openweathermap.org/energy/2.0/solar/interval_data?lat={lat}&lon={lon}&date={date}&interval={interval}&appid={API_key}"
    response = requests.get(url)
    if response.status_code != 200:
        continue

    for i in response.json().get("intervals", []):
        start = datetime.strptime(f"{date} {i['start']}", "%Y-%m-%d %H:%M")
        if d > 0 or start >= now:  # ab jetzt oder morgen
            timestamps.append(start.isoformat(timespec="minutes"))
            dni_values.append(i["avg_irradiance"]["cloudy_sky"]["dni"])
            dhi_values.append(i["avg_irradiance"]["cloudy_sky"]["dhi"])
        if len(timestamps) >= 24:
            break
    if len(timestamps) >= 24:
        break

#print(f"{len(timestamps)} Zeitpunkte gespeichert ab aktueller Uhrzeit:")
#print(timestamps)


def Jahreswinkel(Tag,TagJahr):
    return (360*(Tag/TagJahr))

def Sonnendeklination(J_deg: float) -> float:
    J_deg = math.radians(J_deg)
    return (0.3948
            -23.2559 * math.cos(J_deg + math.radians(9.1))
            -0.3915 * math.cos((2 * J_deg) + math.radians(5.4))
            -0.1764 * math.cos((3 * J_deg) + math.radians(26)))

def Zeitgleichung(J_deg):
    J_deg = math.radians(J_deg)
    return (0.0066
            +7.3525 * math.cos(J_deg + math.radians(85.9))
            + 9.9359 * math.cos(2* J_deg + math.radians(108.9))
            + 0.3387 * math.cos(3 * J_deg + math.radians(105.2)))#min

def MittlereOrtszeit(LZ,Zeitzone,Langitude):
    zeit = datetime.strptime(LZ, "%H:%M")
    LZ_dezimal = zeit.hour + (zeit.minute / 60)
    print("LZ", LZ)
    print("LZ in Dezimal:", LZ_dezimal)
    print("Zeitzone:", Zeitzone)
    MOZ = LZ_dezimal - Zeitzone + ((4*Langitude)/60) 
    return (MOZ)

def WahreOrtszeit(MOZ,Zeitgleichung):
    Zeitgleichung = Zeitgleichung/60
    WahreOrtszeit = dezimal_zu_zeit(MOZ+Zeitgleichung)
    print("Wahre Ortszeit:", WahreOrtszeit)
    return (MOZ+Zeitgleichung)

def Stundenwinkel(WOZ):
    return (12 - WOZ)*15

def Sonnenhöhe(Latitude,deklination,Stundenwinkel):
    Latitude = math.radians(Latitude)
    deklination = math.radians(deklination)
    Stundenwinkel = math.radians(Stundenwinkel)

    return math.degrees(math.asin(math.cos(Stundenwinkel)*math.cos(Latitude)*math.cos(deklination)
                    +math.sin(Latitude)*math.sin(deklination)))


def Sonnenazimut(lat, deklination, Sonnenhöhe, WOZ):
    lat = math.radians(lat)
    deklination = math.radians(deklination)
    Sonnenhöhe = math.radians(Sonnenhöhe)

    zähler = math.sin(Sonnenhöhe) * math.sin(lat) - math.sin(deklination)
    nenner = math.cos(Sonnenhöhe) * math.cos(lat)
    term = zähler / nenner

    # numerische Stabilität (Werte leicht außerhalb [-1, 1] korrigieren)
    #term = max(min(term, 1), -1)

    if WOZ <= 12:
        azimut = 180 - math.degrees(math.acos(term))
    else:
        azimut = 180 + math.degrees(math.acos(term))

    return azimut


def Einfallswinkel(Sonnenhöhe,Sonnenazimunt,Azimut,Neigungswinkel):
    Sonnenhöhe = math.radians(Sonnenhöhe)
    Sonnenazimunt = math.radians(Sonnenazimunt)
    Azimut = math.radians(Azimut)
    Neigungswinkel = math.radians(Neigungswinkel)
    return math.degrees((math.acos(-math.cos(Sonnenhöhe)*math.sin(Neigungswinkel)
                      *math.cos(Sonnenazimunt-Azimut)+math.sin(Sonnenhöhe)
                      *math.cos(Neigungswinkel))))

def DirekteEinstrahlung(EdirektHor,Einfallswinkel,Sonnenhöhe):
    Einfallswinkel = math.radians(Einfallswinkel)
    Sonnenhöhe = math.radians(Sonnenhöhe)
    return (EdirektHor*(math.cos(Einfallswinkel)/math.sin(Sonnenhöhe)))

def DiffuseEinstrahlung(EdiffusHor,Neigungswinkel):
    Neigungswinkel = math.radians(Neigungswinkel)
    return (EdiffusHor*1/2*(1+math.cos(Neigungswinkel)))

def ReflektierteEinstrahlung(EdirektHor,EdiffusHor,Neigungswinkel,Albedo):
    Neigungswinkel = math.radians(Neigungswinkel)
    return ((EdirektHor+EdiffusHor)*Albedo*1/2*(1-math.cos(Neigungswinkel)))

def dezimal_zu_zeit(dezimalstunden):
    stunden = int(dezimalstunden)                 # Ganze Stunden
    minuten = int(round((dezimalstunden - stunden) * 60))  # Minutenanteil

    if minuten == 60:
        minuten = 0
        stunden += 1

    return f"{stunden:02d}:{minuten:02d}"

def PV_ErtragIdeal(Globalstrahlunhg, Nennleistung):
    return (Globalstrahlunhg/1000)*Nennleistung


Jahreswinkel_Wert = Jahreswinkel(day_number,365)
print("Jahreswinkel: ",Jahreswinkel_Wert)

Sonnendeklination_Wert = Sonnendeklination(Jahreswinkel_Wert)
print("Sonnendeklination: ",Sonnendeklination_Wert)

Zeitgleichung_Wert = Zeitgleichung(Jahreswinkel_Wert)
print("Zeitgleichung: ",Zeitgleichung_Wert)


start = datetime.now().replace(minute=0, second=0, microsecond=0)
for i in range(24):
    aktuelle_Zeit = start + timedelta(hours=i)
    print("Zeit: ", aktuelle_Zeit.strftime("%H:%M"))
    Zeit.append(aktuelle_Zeit.strftime("%H:%M"))

    MittlereOrtszeit_Wert = MittlereOrtszeit(LokaleZeit,Zeitzone,lon)
    print("MittlereOrtszeit: ",MittlereOrtszeit_Wert)
    MOZecht = dezimal_zu_zeit(MittlereOrtszeit_Wert)
    print("MOZ in Zeit: ",MOZecht)

    WahreOrtszeit_Wert = WahreOrtszeit(MittlereOrtszeit_Wert,Zeitgleichung_Wert)
    print("WahreOrtszeit: ",WahreOrtszeit_Wert)

    Stundenwinkel_Wert = Stundenwinkel(WahreOrtszeit_Wert)
    print("Stundenwinkel: ",Stundenwinkel_Wert)

    Sonnenhöhe_Wert = Sonnenhöhe(lat,Sonnendeklination_Wert,Stundenwinkel_Wert)
    print("Sonnenhöhe: ",Sonnenhöhe_Wert)

    Sonnenazimut_Wert = Sonnenazimut(lat,Sonnendeklination_Wert,Sonnenhöhe_Wert,WahreOrtszeit_Wert)
    print("Sonnenazimut: ",Sonnenazimut_Wert)

    Einfallswinkel_Wert = Einfallswinkel(Sonnenhöhe_Wert,Sonnenazimut_Wert,Azimut,Neigungswinkel)
    print("Einfalsswinkel: ",Einfallswinkel_Wert)

    DirekteEinstrahlung_Wert = DirekteEinstrahlung(dni_values[i],Einfallswinkel_Wert,Sonnenhöhe_Wert)
    print("DirekteEinstrahlung: ",DirekteEinstrahlung_Wert)
    values_direkt.append(DirekteEinstrahlung_Wert)

    DiffuseEinstrahlung_Wert = DiffuseEinstrahlung(dhi_values[i],Neigungswinkel)
    print("DiffuseEinstrahlung: ",DiffuseEinstrahlung_Wert)
    values_diff.append(DiffuseEinstrahlung_Wert)

    ReflektierteEinstrahlung_Wert = ReflektierteEinstrahlung(dni_values[i],dhi_values[i],Neigungswinkel,Albedo)
    print("ReflektierteEinstrahlung: ",ReflektierteEinstrahlung_Wert)
    values_reflekt.append(ReflektierteEinstrahlung_Wert)


    values_global.append(DirekteEinstrahlung_Wert + DiffuseEinstrahlung_Wert + ReflektierteEinstrahlung_Wert)

    LokaleZeitUhrzeit = datetime.strptime(LokaleZeit, "%H:%M")
    LokaleZeitUhrzeit += timedelta(hours=1)
    LokaleZeit = LokaleZeitUhrzeit.strftime("%H:%M")

# Ertragsrechnung PV

for i in range(len(values_global)):
    Ertrag = PV_ErtragIdeal(values_global[i], Nennleistung)
    values_Ertrag.append(Ertrag)
    print(f"PV-Ertrag um {Zeit[i]}: {Ertrag:.2f} W")




# Punkte plotten
plt.figure(figsize=(14,8))
#plt.scatter(Zeit, values_direkt, color="orange", label="DNI (W/m²)")   # Punkte
#plt.plot(Zeit, values_direkt, linestyle='-', color="gray", alpha=0.5) # optionale Verbindungslinie
#plt.plot(Zeit,values_diff, linestyle='--', color="blue", label="DHI (W/m²)")
plt.plot(Zeit,values_global, linestyle='-', color="green", label="GHI (W/m²)")
plt.plot(Zeit,values_Ertrag, linestyle='--', color="blue", label="PV-Ertrag (W)")

plt.title("Direkte Sonneneinstrahlung (DNI) über Zeit")
plt.xlabel("Zeit")
plt.ylabel("DNI [W/m²]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()