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
import plotly.express as px

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
Nennleistung = st.sidebar.number_input(
    "Kapazität DC Modul (W)", min_value=0.0, step=0.1, value=5000.0
)
Azimuth = st.sidebar.number_input(
    "Azimuth (°)", min_value=0, max_value=360, step=1, value=90
)
Neigungswinkel = st.sidebar.number_input(
    "Neigungswinkel (°)", min_value=0, max_value=90, step=1, value=30
)
use_storage = st.sidebar.checkbox("Speicher verwenden", value=True)

if use_storage:
    Speicherkapazität = st.sidebar.number_input(
        "Speicherkapazität (kWh)", min_value=1, max_value=90, step=1, value=10
    )
else:
    Speicherkapazität = 0

use_pv = st.sidebar.checkbox("PV verwenden", value=True)
use_Echarging = st.sidebar.checkbox("E-Ladestation", value=False)



print(Azimuth, Neigungswinkel)

#Dynamische Strompreise

@st.cache_data(ttl=15 * 60)  # 15 Minuten cachen
def fetch_market_data():
    url = "https://api.awattar.de/v1/marketdata"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    timestamps, prices = [], []
    
    for entry in data.get("data", []):
        timestamps.append(datetime.fromtimestamp(entry["start_timestamp"] / 1000))
        original_price = entry["marketprice"] * 0.1
        end_price = (original_price * 1.2) / 0.4
        prices.append(end_price)

    target_len = 24
    if len(prices) > target_len:
        prices = prices[:target_len]
        timestamps = timestamps[:target_len]

    elif len(prices) < target_len:
        missing = target_len - len(prices)
        prices += [0] * missing
        
        if timestamps:
            last_ts = timestamps[-1]
        else:
            last_ts = datetime.now().replace(minute=0, second=0, microsecond=0)
        for i in range(missing):
            timestamps.append(last_ts + timedelta(hours=i+1))

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


    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"<h4 style='text-align: center;'>Minimum (ct/kWh)</h4>"
            f"<h2 style='text-align: center;'>{min(prices):.2f} €</h2>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"<h4 style='text-align: center;'>Maximum (ct/kWh)</h4>"
            f"<h2 style='text-align: center;'>{max(prices):.2f} €</h2>",
            unsafe_allow_html=True,
        )

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
Zeitzone = 1.0
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

values_direkt, values_diff, values_reflekt, ZeitT, values_global, values_Ertrag = [], [], [], [], [], []

#Daten einlesen
df = pd.read_csv("dataBestrahlung.csv", header=None, names=["timestamp", "dni", "dhi"])
df = df.replace("NaN", pd.NA)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M")
df["dni"] = pd.to_numeric(df["dni"], errors="coerce").fillna(0.0)
df["dhi"] = pd.to_numeric(df["dhi"], errors="coerce").fillna(0.0)
df = df.sort_values("timestamp").reset_index(drop=True)
df["time"] = df["timestamp"].dt.time  

now = datetime.now().replace(second=0, microsecond=0)
now_time = now.time().replace(minute=0)  

print(f"Aktuelle Uhrzeit (nur Time-of-Day): {now_time}")
mask = df["time"] >= now_time
if mask.any():
    start_index = mask.idxmax() 
else:
    start_index = 0

n = len(df)
indices = [(start_index + i) % n for i in range(24)]
subset = df.iloc[indices].copy()

timestamps = subset["time"].astype(str).str.slice(0,5).tolist()  # "HH:MM"
dni_values = subset["dni"].tolist()
dhi_values = subset["dhi"].tolist()


#for t, dni, dhi in zip(timestamps, dni_values, dhi_values):
#    print(f"{t}  DNI={dni:.1f}  DHI={dhi:.1f}")


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
    #print("LokaltZeit: ", LZ_dezimal)
    MOZ = LZ_dezimal - Zeitzone + ((4*Langitude)/60) 
    print(MOZ)
    return (MOZ)

def WahreOrtszeit(MOZ,Zeitgleichung):
    Zeitgleichung = Zeitgleichung/60
    WahreOrtszeit = dezimal_zu_zeit(MOZ+Zeitgleichung)
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
print(day_number)

Sonnendeklination_Wert = Sonnendeklination(Jahreswinkel_Wert)
Zeitgleichung_Wert = Zeitgleichung(Jahreswinkel_Wert)



start = datetime.now().replace(minute=0, second=0, microsecond=0)
for i in range(24):
    aktuelle_Zeit = start + timedelta(hours=i)
    ZeitT.append(aktuelle_Zeit.strftime("%H:%M"))

    MittlereOrtszeit_Wert = MittlereOrtszeit(LokaleZeit,Zeitzone,current_lon)
    MOZecht = dezimal_zu_zeit(MittlereOrtszeit_Wert)


    WahreOrtszeit_Wert = WahreOrtszeit(MittlereOrtszeit_Wert,Zeitgleichung_Wert)


    Stundenwinkel_Wert = Stundenwinkel(WahreOrtszeit_Wert)


    Sonnenhöhe_Wert = Sonnenhöhe(current_lat,Sonnendeklination_Wert,Stundenwinkel_Wert)


    Sonnenazimut_Wert = Sonnenazimut(current_lat,Sonnendeklination_Wert,Sonnenhöhe_Wert,WahreOrtszeit_Wert)


    Einfallswinkel_Wert = Einfallswinkel(Sonnenhöhe_Wert,Sonnenazimut_Wert,Azimuth,Neigungswinkel)



    DirekteEinstrahlung_Wert = DirekteEinstrahlung(dni_values[i],Einfallswinkel_Wert,Sonnenhöhe_Wert)
    
    if(DirekteEinstrahlung_Wert < 0):
        DirekteEinstrahlung_Wert = 0
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
    values_Ertrag.append(Ertrag/1000)  # in kWh
    #print(f"PV-Ertrag um {ZeitT[i]}: {Ertrag:.2f} Wh")



#======================== Lastprofil========================#
#===========================================================

RAW = """
00:00-01:00	81,607
01:00-02:00	69,134
02:00-03:00	63,194
03:00-04:00	60,748
04:00-05:00	60,665
05:00-06:00	61,715
06:00-07:00	72,497
07:00-08:00	92,793
08:00-09:00	114,371
09:00-10:00	127,643
10:00-11:00	137,816
11:00-12:00	151,586
12:00-13:00	151,118
13:00-14:00	144,839
14:00-15:00	141,093
15:00-16:00	140,990
16:00-17:00	149,706
17:00-18:00	169,830
18:00-19:00	176,471
19:00-20:00	166,325
20:00-21:00	148,626
21:00-22:00	134,112
22:00-23:00	123,278
23:00-00:00	102,804
"""

# BDEW-Dynamisierung
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
    scaled = series / 1_000_000 * 4000
    return scaled * bdew_factor(t)


rows = []
for line in RAW.strip().splitlines():
    times, v = line.split('\t')
    start, end = times.split('-')
    value = float(v.replace('.', '').replace(',', '.'))  # Dezimalkomma → Punkt
    rows.append((start, end, value))

df = pd.DataFrame(rows, columns=["start", "end", "value"])

base_date = datetime.now().date()
df["timestamp"] = pd.to_datetime(df["start"].apply(lambda t: f"{base_date} {t}"))
df = df.set_index("timestamp")


# Dynamisierung
t = day_number  # Kalendertag festlegen (1..365/366)
df["final"] = dynamize_profile(df["value"], t)



now = datetime.now()
now_rounded = now.replace(minute=0, second=0, microsecond=0)
df_future = df[df.index >= now_rounded]
df_past = df[df.index < now_rounded].copy()

# Den früheren Teil zeitlich *nach* den zukünftigen Teil verschieben:
df_past.index = df_past.index + timedelta(days=1)

# Kombinieren
df_combined = pd.concat([df_future, df_past])

if use_Echarging:
    mask = (df_combined.index.hour >= 18) & (df_combined.index.hour < 21)
    df_combined.loc[mask, "final"] += 11



######Last verschieben#######
günstigste_zeiten = sorted(range(len(prices)), key=lambda i: prices[i])[:3]
if use_Echarging:
    mask = (df_combined.index.hour >= 18) & (df_combined.index.hour < 21)
    df_combined.loc[mask, "final"] -= 11
    df_combined.iloc[günstigste_zeiten, df_combined.columns.get_loc("final")] += 11

############################



df_energy = pd.DataFrame({
    "Zeit": ZeitT,
    "PV": values_Ertrag[:len(ZeitT)],
    "Verbrauch": df_combined["final"].iloc[:len(ZeitT)].values,
    "Preis_ct": prices[:len(ZeitT)]
})

if use_pv:
    df_energy["Überschuss"] = df_energy["PV"] - df_energy["Verbrauch"]
else:
    df_energy["Überschuss"] = df_energy["Verbrauch"]


speicher_kapazitaet = Speicherkapazität 
speicher_ladung = 0      
speicher = []
entscheidungen = []
einspeiseVergütung = 0.08  # €/kWh
einspeisung = 0
einspeiseEinkommen = 0
gesamtKosten = 0


for _, row in df_energy.iterrows():
    ueberschuss = row["Überschuss"]
    preis = row["Preis_ct"]
    Zeit = row["Zeit"]
    
    stunde = int(row["Zeit"].split(":")[0])
    entscheidung = ""

    if use_storage and use_pv:
        if ueberschuss > 0:
            if speicher_ladung < speicher_kapazitaet:
                # Strom speichern
                speicherbare_menge = min(ueberschuss, speicher_kapazitaet - speicher_ladung)
                speicher_ladung += speicherbare_menge
                entscheidung = f"Speichern ({speicherbare_menge:.2f} kWh)"
                if ueberschuss > speicherbare_menge:
                    rest_ueberschuss = ueberschuss - speicherbare_menge
                    einspeiseEinkommen += rest_ueberschuss * einspeiseVergütung
                    einspeisung += rest_ueberschuss
                    entscheidung += f"Einpeisen ins Netz ({einspeisung:.2f} kWh)"
            else:
                entscheidung = "Einspeisen ins Netz"
                einspeiseEinkommen += ueberschuss * einspeiseVergütung
                einspeisung += ueberschuss


        elif ueberschuss < 0:  
            if speicher_ladung > 0 and (stunde >= 20 or stunde < 6):
                # Speicher entladen erst zwischen 20 und 6 Uhr
                entnehmbare_menge = min(-ueberschuss, speicher_ladung)
                speicher_ladung -= entnehmbare_menge
                # wenn nicht genug aus der Speicher entnommen werden kann, rest kaufen
                rest_bedarf = (-ueberschuss) -  entnehmbare_menge
                if(rest_bedarf > 0):
                    entscheidung = f"Entladen ({entnehmbare_menge:.2f} kWh), Netzbezug({rest_bedarf:.2f} kWh) "
                else:
                    entscheidung = f"Entladen ({entnehmbare_menge:.2f} kWh)"
                if rest_bedarf > 0:
                    gesamtKosten += rest_bedarf * preis
            else:
                # Speicher leer Strom kaufen
                entscheidung = "Netzbezug"
                gesamtKosten += -ueberschuss * preis
    elif use_pv and not use_storage:
        if ueberschuss > 0:
            entscheidung = "Einspeisen ins Netz"
            einspeiseEinkommen += ueberschuss * einspeiseVergütung
            einspeisung += ueberschuss
        else:
            entscheidung = "Netzbezug"
            gesamtKosten += -ueberschuss * preis
    elif not use_pv and not use_storage:
        entscheidung = "Netzbezug"
        gesamtKosten += ueberschuss * preis
    speicher.append(speicher_ladung)
    entscheidungen.append(entscheidung)


fig3, ax = plt.subplots(figsize=(14, 8))

# Hauptachsen: PV-Ertrag und Verbrauch
ax.plot(ZeitT, df_energy["Verbrauch"], label="Lastprofil", linestyle="-", drawstyle="steps-post")
if use_pv:
    ax.plot(ZeitT, values_Ertrag, linestyle='-', color="blue", label="PV-Ertrag", drawstyle="steps-post")

ax.set_xlabel("Zeit")
ax.set_ylabel("Leistung [kWh]")

# Speicher nur anzeigen, wenn aktiviert
if use_storage:
    ax2 = ax.twinx()
    ax2.plot(ZeitT, speicher, color="green", linestyle="--", linewidth=2, label="Speicherinhalt")
    ax2.set_ylabel("Speicherinhalt [kWh]")
    title = "PV-Ertrag, Verbrauch und Speicherstand"

    # Legenden kombinieren
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")

else:
    title = "PV-Ertrag und Verbrauch"
    ax.legend(loc="upper left")

if not use_pv and not use_storage:
        title= "Verbrauch"

ax.set_title(title)
ax.grid(True)
plt.tight_layout()
st.pyplot(fig3)


df_energy["Speicherinhalt_kWh"] = speicher
df_energy["Entscheidung"] = entscheidungen

col1, col2, col3 = st.columns([1, 20, 1])  # Verhältnis für zentrierte Darstellung
with col2:
    st.dataframe(
        df_energy[["Zeit", "PV", "Verbrauch", "Preis_ct", "Überschuss", "Speicherinhalt_kWh", "Entscheidung"]],
        use_container_width=True
    )

Kosten = round(gesamtKosten/100,2) - round(einspeiseEinkommen,2)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"<h4 style='text-align: center;'>Gesamtkosten</h4>"
        f"<h2 style='text-align: center;'>{round(Kosten,2)} €</h2>",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"<h4 style='text-align: center;'>Einspeiseinahmen</h4>"
        f"<h2 style='text-align: center;'>{round(einspeiseEinkommen,2)} €</h2>",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"<h4 style='text-align: center;'>Einspeisung</h4>"
        f"<h2 style='text-align: center;'>{round(einspeisung,2)} kWh</h2>",
        unsafe_allow_html=True,
    )






