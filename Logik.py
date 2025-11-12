if use_storage:
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
        elif speicher_ladung > 0 and (stunde >= 20 or stunde < 6) and ueberschuss < 0:
                # Speicher entladen
                entnehmbare_menge = min(-ueberschuss, speicher_ladung)
                speicher_ladung -= entnehmbare_menge
                entscheidung = f"Entladen ({entnehmbare_menge:.2f} kWh)"
                if(-ueberschuss) -  entnehmbare_menge > 0:
                    rest_kauf = (-ueberschuss) - entnehmbare_menge
                    gesamtKosten += rest_kauf * preis
        elif ueberschuss < 0 and preis < 0:
            entscheidung = "Netzbezug"
            gesamtKosten += -ueberschuss * preis 