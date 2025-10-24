import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =========================
# 1) Rohdaten hier einfügen
# =========================
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

# =========================
# 2) BDEW-Dynamisierung
# =========================
def bdew_factor(t: int) -> float:
    return (
        -3.92e-10 * t**4
        + 3.20e-7  * t**3
        - 7.02e-5  * t**2
        + 2.10e-3  * t
        + 1.24
    )

def dynamize_profile(series: pd.Series, t: int) -> pd.Series:
    scaled = series / 1_000_000 * 3500
    return scaled * bdew_factor(t)

# =========================
# 3) Parsing Rohdaten
# =========================
rows = []
for line in RAW.strip().splitlines():
    times, v = line.split('\t')
    start, end = times.split('-')
    value = float(v.replace('.', '').replace(',', '.'))
    rows.append((start, end, value))

df = pd.DataFrame(rows, columns=["start", "end", "value"])

# Zeitindex
base_date = datetime.now().date()
df["timestamp"] = pd.to_datetime(df["start"].apply(lambda t: f"{base_date} {t}"))
df = df.set_index("timestamp")

# =========================
# 4) Dynamisierung
# =========================
t = 1
df["final"] = dynamize_profile(df["value"], t)

# =========================
# 5) Ab jetzt + Rest anhängen
# =========================
now = datetime.now()
df_future = df[df.index >= now]
df_past = df[df.index < now].copy()

# Den früheren Teil zeitlich *nach* den zukünftigen Teil verschieben:
df_past.index = df_past.index + timedelta(days=1)

# Kombinieren
df_combined = pd.concat([df_future, df_past])

# =========================
# 6) Plot
# =========================
plt.figure(figsize=(12, 5))
plt.plot(df_combined.index, df_combined["final"], label=f"Start ab {now.strftime('%H:%M')}", linestyle="-")
plt.title("Lastprofil – ab aktueller Uhrzeit mit Rest angehängt")
plt.xlabel("Zeit (rollierend über 24h)")
plt.ylabel("kWh")
ticks = pd.date_range(df_combined.index.min(), df_combined.index.max(), freq="2H")
plt.xticks(ticks, [ts.strftime("%H:%M") for ts in ticks])
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 7) Summen
# =========================
energy_combined = df_combined["final"].sum()
print(f"Gesamtenergie (24h, neu sortiert ab {now.strftime('%H:%M')}): {energy_combined:,.3f} kWh")
