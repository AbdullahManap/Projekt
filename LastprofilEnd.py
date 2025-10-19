import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# 1) Rohdaten hier einfügen
# =========================
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

def dynamize_profile(df: pd.DataFrame, t: int) -> pd.Series:
    """Wendet Dynamisierungsfaktor an und rechnet um."""
    factor = bdew_factor(t)
    return df["value"] * factor

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
# 4) Dynamisierung & Skalierung
# =========================
t = 15  # Kalendertag festlegen (1..365/366)
df["dyn_kwh"] = dynamize_profile(df, t)

# Endumrechnung: durch 1e6 teilen und mit 3500 multiplizieren
df["final"] = df["dyn_kwh"] / 1_000_000 * 3500

# =========================
# 5) Plot: Original vs. Final
# =========================
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["final"], label=f"Dynamisiert+skaliert (t={t})", linestyle="--")
plt.title("Lastprofil – dynamisiert und skaliert")
plt.xlabel("Uhrzeit")
plt.ylabel("Wert (Einheit nach Skalierung)")
ticks = pd.date_range(df.index.min(), df.index.max(), freq="2H")
plt.xticks(ticks, [ts.strftime("%H:%M") for ts in ticks])
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 6) Summen (Tageswerte)
# =========================
energy_orig = df["value"].sum()
energy_dyn  = df["dyn_kwh"].sum()
energy_final = df["final"].sum()

print(f"Tagesenergie original:     {energy_orig:,.3f} kWh")
print(f"Tagesenergie dynamisiert:  {energy_dyn:,.3f} kWh")
print(f"Tagessumme skaliert:       {energy_final:,.3f} (nach /1e6 * 3500)")
