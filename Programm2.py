import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import math


st.set_page_config(page_title="Strompreisoptimierung", layout="wide")
st.title("Strompreisoptimierung")

st.sidebar.header("Ort")

@st.cache_data(ttl=60 * 60, show_spinner=False)  # 1 Stunde cachen
def geocode_place(place: str):
    try:
        geolocator = Nominatim(
            user_agent="awattar_streamlit_app/1.0",
            timeout=10  # höherer Timeout
        )
        geocode_rl = RateLimiter(
            geolocator.geocode,
            min_delay_seconds=1.0, 
            max_retries=3,
            error_wait_seconds=2.0,
            swallow_exceptions=False
        )
        loc = geocode_rl(place, addressdetails=True, language="de", exactly_one=True)
        if loc:
            return {"lat": loc.latitude, "lon": loc.longitude, "name": loc.address}
    except (GeocoderTimedOut, GeocoderServiceError, Exception):
        # weiter zu Fallback
        pass


    try:
        params = {
            "name": place,
            "count": 1,
            "language": "de",
            "format": "json"
        }
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=params, timeout=10)
        if r.status_code == 200:
            js = r.json()
            results = js.get("results") or []
            if results:
                res = results[0]
                name_parts = [res.get("name"), res.get("admin1"), res.get("country")]
                name = ", ".join([p for p in name_parts if p])
                return {"lat": float(res["latitude"]), "lon": float(res["longitude"]), "name": name}
    except requests.RequestException:
        pass

    # Nichts gefunden
    return None

with st.sidebar:
    place_input = st.text_input("Ort eingeben", value="Gummersbach")
    col_sb1, col_sb2 = st.columns([1, 1])
    with col_sb1:
        search_clicked = st.button("Suchen")
    with col_sb2:
        clear_clicked = st.button("Zurücksetzen")

    # SessionState für aktuelle Position (Default: Gummersbach)
    if "current_place" not in st.session_state:
        st.session_state.current_place = {
            "name": "Gummersbach, Deutschland",
            "lat": 51.026,
            "lon": 7.565,
        }

    if clear_clicked:
        st.session_state.current_place = {
            "name": "Gummersbach, Deutschland",
            "lat": 51.026,
            "lon": 7.565,
        }
        st.rerun()

    if search_clicked and place_input.strip():
        with st.spinner("Suche Ort..."):
            result = geocode_place(place_input.strip())
        if result is None:
            st.warning("Ort konnte nicht bestimmt werden (Timeout/keine Treffer). Bitte erneut versuchen oder genauer eingeben.")
        else:
            st.session_state.current_place = {
                "name": result["name"],
                "lat": result["lat"],
                "lon": result["lon"],
            }
            st.success(f"Gefunden: {result['name']}")
            st.map(pd.DataFrame([{"lat": result["lat"], "lon": result["lon"]}]))

current_lat = st.session_state.current_place["lat"]
current_lon = st.session_state.current_place["lon"]
current_name = st.session_state.current_place["name"]

# PV-Anlagendaten in Sidebar
st.sidebar.header("PV-Anlage Parameter")
kapazitaet_ac = st.sidebar.number_input(
    "Kapazität AC (kW)", min_value=0.0, step=0.1, value=0.0
)
Nennleistung = st.sidebar.number_input(
    "Kapazität DC Modul (W)", min_value=0.0, step=0.1, value=5000.0
)
Azimuth = st.sidebar.number_input(
    "Azimuth (°)", min_value=0, max_value=360, step=1, value=90
)
Neigungswinkel = st.sidebar.number_input(
    "Neigungswinkel (°)", min_value=0, max_value=90, step=1, value=30
)

print(Azimuth, Neigungswinkel)


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

    fill_color = "#8cd7f8"
    line_color = "#12b8ef"
    alpha = 0.25

    last_end = timestamps[-1] + timedelta(hours=1)
    x_step = timestamps + [last_end]
    y_step = prices + [prices[-1]]

    st.subheader("Dynamische Strompreise")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(x_step, y_step, step="post", alpha=alpha, color=fill_color)
    ax.step(x_step, y_step, where="post", linewidth=2.5, color=line_color)

    for t, p in zip(timestamps, prices):
        ax.text(
            t + timedelta(minutes=30),
            p + 0.2,
            f"{p:.2f}".replace(".", ","),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=90)
    ax.set_xlim(timestamps[0], last_end)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Uhrzeit")
    ax.set_ylabel("Preis (ct/kWh)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    st.pyplot(fig, clear_figure=True)

    col1, col2, col3 = st.columns(3)
    col2.metric("Minimum (ct/kWh)", f"{min(prices):.2f}".replace(".", ","))
    col3.metric("Maximum (ct/kWh)", f"{max(prices):.2f}".replace(".", ","))

except requests.RequestException as e:
    st.error(f"Fehler beim Abrufen der Daten (awattar): {e}")

#================ Ertragsrechnung PV-Vorhersage ================#

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
#Nennleistung = 5000  # in Watt
#Azimut = 180
#Neigungswinkel = 30
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
    url = f"https://api.openweathermap.org/energy/2.0/solar/interval_data?lat={current_lat}&lon={current_lon}&date={date}&interval={interval}&appid={API_key}"
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
    Zeit.append(aktuelle_Zeit.strftime("%H:%M"))

    MittlereOrtszeit_Wert = MittlereOrtszeit(LokaleZeit,Zeitzone,current_lon)
    MOZecht = dezimal_zu_zeit(MittlereOrtszeit_Wert)

    WahreOrtszeit_Wert = WahreOrtszeit(MittlereOrtszeit_Wert,Zeitgleichung_Wert)

    Stundenwinkel_Wert = Stundenwinkel(WahreOrtszeit_Wert)

    Sonnenhöhe_Wert = Sonnenhöhe(current_lat,Sonnendeklination_Wert,Stundenwinkel_Wert)

    Sonnenazimut_Wert = Sonnenazimut(current_lat,Sonnendeklination_Wert,Sonnenhöhe_Wert,WahreOrtszeit_Wert)

    Einfallswinkel_Wert = Einfallswinkel(Sonnenhöhe_Wert,Sonnenazimut_Wert,Azimuth,Neigungswinkel)

    DirekteEinstrahlung_Wert = DirekteEinstrahlung(dni_values[i],Einfallswinkel_Wert,Sonnenhöhe_Wert)
    values_direkt.append(DirekteEinstrahlung_Wert)

    DiffuseEinstrahlung_Wert = DiffuseEinstrahlung(dhi_values[i],Neigungswinkel)
    values_diff.append(DiffuseEinstrahlung_Wert)

    ReflektierteEinstrahlung_Wert = ReflektierteEinstrahlung(dni_values[i],dhi_values[i],Neigungswinkel,Albedo)
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




#======================== Lastprofil========================#
#===========================================================

RAW = """00:00-00:15	22,152
00:15-00:30	20,809
00:30-00:45	19,757
00:45-01:00	18,889
01:00-01:15	18,217
01:15-01:30	17,530
01:30-01:45	16,988
01:45-02:00	16,399
02:00-02:15	16,199
02:15-02:30	15,893
02:30-02:45	15,660
02:45-03:00	15,442
03:00-03:15	15,398
03:15-03:30	15,219
03:30-03:45	15,102
03:45-04:00	15,029
04:00-04:15	15,190
04:15-04:30	15,066
04:30-04:45	15,097
04:45-05:00	15,312
05:00-05:15	15,366
05:15-05:30	15,192
05:30-05:45	15,338
05:45-06:00	15,819
06:00-06:15	16,806
06:15-06:30	17,632
06:30-06:45	18,582
06:45-07:00	19,477
07:00-07:15	21,015
07:15-07:30	22,340
07:30-07:45	23,947
07:45-08:00	25,491
08:00-08:15	26,837
08:15-08:30	28,317
08:30-08:45	29,179
08:45-09:00	30,038
09:00-09:15	30,834
09:15-09:30	31,737
09:30-09:45	32,301
09:45-10:00	32,771
10:00-10:15	33,451
10:15-10:30	34,172
10:30-10:45	34,721
10:45-11:00	35,472
11:00-11:15	36,638
11:15-11:30	37,602
11:30-11:45	38,463
11:45-12:00	38,883
12:00-12:15	38,394
12:15-12:30	37,981
12:30-12:45	37,549
12:45-13:00	37,194
13:00-13:15	36,759
13:15-13:30	36,299
13:30-13:45	36,095
13:45-14:00	35,686
14:00-14:15	35,325
14:15-14:30	35,331
14:30-14:45	35,305
14:45-15:00	35,132
15:00-15:15	35,158
15:15-15:30	35,179
15:30-15:45	35,203
15:45-16:00	35,450
16:00-16:15	36,116
16:15-16:30	36,902
16:30-16:45	37,761
16:45-17:00	38,927
17:00-17:15	40,646
17:15-17:30	42,110
17:30-17:45	43,119
17:45-18:00	43,955
18:00-18:15	44,419
18:15-18:30	44,443
18:30-18:45	44,085
18:45-19:00	43,524
19:00-19:15	42,928
19:15-19:30	42,060
19:30-19:45	41,173
19:45-20:00	40,164
20:00-20:15	39,182
20:15-20:30	37,781
20:30-20:45	36,282
20:45-21:00	35,381
21:00-21:15	34,692
21:15-21:30	33,815
21:30-21:45	32,837
21:45-22:00	32,768
22:00-22:15	32,678
22:15-22:30	31,368
22:30-22:45	30,280
22:45-23:00	28,952
23:00-23:15	27,554
23:15-23:30	26,256
23:30-23:45	25,052
23:45-00:00	23,942
"""

# =========================
# 2) BDEW-Dynamisierung
# =========================
def bdew_factor(t: int) -> float:
    """Berechnet den Dynamisierungsfaktor P(t) für Kalendertag t (1..365/366)."""
    return (
        -3.92e-10 * t**4
        + 3.20e-7  * t**3
        - 7.02e-5  * t**2
        + 2.10e-3  * t
        + 1.24
    )

def dynamize_profile(series: pd.Series, t: int) -> pd.Series:
    """Erst skalieren (/1e6 * 3500), dann dynamisieren."""
    scaled = series / 1_000_000 * 3500
    return scaled * bdew_factor(t)

# =========================
# 3) Parsing Rohdaten
# =========================
rows = []
for line in RAW.strip().splitlines():
    times, v = line.split('\t')
    start, end = times.split('-')
    value = float(v.replace('.', '').replace(',', '.'))  # Dezimalkomma → Punkt
    rows.append((start, end, value))

df = pd.DataFrame(rows, columns=["start", "end", "value"])

# Zeitindex für Plot
base_date = datetime(2025, 1, 1)
df["timestamp"] = pd.to_datetime(base_date.strftime("%Y-%m-%d") + " " + df["start"])
df = df.set_index("timestamp")

# =========================
# 4) Dynamisierung
# =========================
t = 1  # Kalendertag festlegen (1..365/366)
df["final"] = dynamize_profile(df["value"], t)

#========================
#======= Plotten =
#=======================

fig, ax = plt.subplots(figsize=(14, 8))

#ax.plot(Zeit, values_global, linestyle='-', color="green", label="GHI (W/m²)")
ax.plot(Zeit, values_Ertrag, linestyle='-', color="blue", label="PV-Ertrag (W)")

ax.set_title("PV-Ertrag")
ax.set_xlabel("Zeit")
ax.set_ylabel("Ertrag [W]")
ax.grid(True)
ax.legend()
plt.tight_layout()

# Plot in Streamlit anzeigen
st.pyplot(fig)



fig2, ax = plt.subplots(figsize=(12, 5))

ax.plot(df.index, df["final"], label=f"Skaliert + dynamisiert (t={t})", linestyle="-")
ax.set_title("Lastprofil – skaliert und dynamisiert")
ax.set_xlabel("Uhrzeit")
ax.set_ylabel("kWh")

ticks = pd.date_range(df.index.min(), df.index.max(), freq="2H")
ax.set_xticks(ticks)
ax.set_xticklabels([ts.strftime("%H:%M") for ts in ticks], rotation=45)
ax.legend()
plt.tight_layout()

# ===================================
# 5) Plot anzeigen
# ===================================
st.pyplot(fig2)