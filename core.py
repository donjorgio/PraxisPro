import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime
import os
import re

# Daten laden
def load_symptom_db():
    with open("symptoms.json", "r", encoding="utf-8") as f:
        symptom_data = json.load(f)["symptome"]
        return {s["id"]: s for s in symptom_data}

# Trainingsdaten erweitern
def load_training_data():
    # Versuche, CSV zu laden, wenn vorhanden
    if os.path.exists("patient_cases.csv"):
        df = pd.read_csv("patient_cases.csv")
        # Überprüfe, ob 'anamnese' statt 'symptome' vorhanden ist
        if 'anamnese' in df.columns and 'symptome' not in df.columns:
            # Umbenennen der Spalte oder eine neue Spalte erstellen
            df['symptome'] = df['anamnese']
        return df
    else:
        # Fallback zu eingebetteten Daten
        return pd.DataFrame([
            {"symptome": "Brustschmerzen,Kurzatmigkeit", "diagnose": "Akuter Herzinfarkt"},
            {"symptome": "Fieber >38°C,Husten", "diagnose": "Lungenentzündung"},
            {"symptome": "Muedigkeit,Uebelkeit", "diagnose": "Virusgrippe"}
        ])

# Pädiatrische Trainingsdaten hinzufügen
def add_pediatric_cases():
    pediatric_cases = [
        {"symptome": "Säuglingshusten,Fieber >38°C,Atemnot", "vitals": "HR:130,T:38.5,RR:45", "diagnose": "RSV-Bronchiolitis"},
        {"symptome": "Bellender Husten,Heiserkeit,Fieber >38°C", "vitals": "HR:120,T:38.7,SpO2:94", "diagnose": "Pseudokrupp"},
        {"symptome": "Säuglingshusten,Fieber >38°C,Schnupfen", "vitals": "HR:125,T:38.3,RR:40", "diagnose": "Virale Atemwegsinfektion"},
        {"symptome": "Kindliches Fieber,Husten,Pfeifen beim Atmen", "vitals": "HR:130,T:38.6,SpO2:92", "diagnose": "Bronchiolitis"},
        {"symptome": "Hohes Fieber,Schluckbeschwerden,Speichelfluss", "vitals": "HR:140,T:39.5,SpO2:91", "diagnose": "Epiglottitis"},
        {"symptome": "Fieber >38°C,Ohrenschmerzen,Weinen", "vitals": "HR:125,T:38.4,BP:100/65", "diagnose": "Akute Otitis media"},
        {"symptome": "Durchfall,Erbrechen,Fieber >38°C", "vitals": "HR:120,T:38.3,BP:95/60", "diagnose": "Gastroenteritis bei Kindern"},
        {"symptome": "Husten,Schnupfen,Halsschmerzen bei Kindern", "vitals": "HR:110,T:37.8,BP:100/70", "diagnose": "Virale Atemwegsinfektion"},
        {"symptome": "Fieber >39°C,Hautausschlag,Unwohlsein", "vitals": "HR:130,T:39.2,BP:100/65", "diagnose": "Kinderkrankheit"},
        {"symptome": "Kindliches Fieber,Kopfschmerzen,Lichtempfindlichkeit", "vitals": "HR:125,T:38.9,BP:110/70", "diagnose": "Meningitis"},
        {"symptome": "Fieber >38°C,Ohrenschmerzen,Erbrechen", "vitals": "HR:120,T:38.5,BP:95/60", "diagnose": "Akute Otitis media"},
        {"symptome": "Säuglingshusten,Atemnot,Fieber >38°C", "vitals": "HR:135,T:38.4,SpO2:91", "diagnose": "Bronchopneumonie bei Säuglingen"},
        {"symptome": "Fieber >38°C,Hautausschlag (masernähnlich),Husten", "vitals": "HR:120,T:38.6,BP:100/65", "diagnose": "Masern"},
        {"symptome": "Fieber >38°C,Halsschmerzen,Hautausschlag", "vitals": "HR:115,T:38.4,BP:105/70", "diagnose": "Scharlach"},
        {"symptome": "Fieber >39°C,Erbrechen,Nackenstarre", "vitals": "HR:130,T:39.3,BP:100/60", "diagnose": "Meningitis"}
    ]
    
    # Erstellen oder aktualisieren der CSV-Datei
    existing_data = None
    max_id = 0
    
    if os.path.exists("patient_cases.csv"):
        existing_data = pd.read_csv("patient_cases.csv")
        if 'id' in existing_data.columns:
            max_id = existing_data['id'].max()
    
    pediatric_df = pd.DataFrame(pediatric_cases)
    
    # IDs zuweisen
    id_start = max_id + 1 if max_id > 0 else 1
    pediatric_df['id'] = range(id_start, id_start + len(pediatric_cases))
    
    # Zu bestehenden Daten hinzufügen oder neue Datei erstellen
    if existing_data is not None:
        # Prüfen, ob die pädiatrischen Fälle bereits enthalten sind
        existing_pediatric = existing_data[existing_data['diagnose'].str.contains('RSV|Bronchiolitis|Pseudokrupp|Epiglottitis', na=False)]
        if len(existing_pediatric) < 5:  # Wenn weniger als 5 pädiatrische Fälle vorhanden sind
            combined_df = pd.concat([existing_data, pediatric_df], ignore_index=True)
            combined_df.to_csv("patient_cases.csv", index=False)
    else:
        pediatric_df.to_csv("patient_cases.csv", index=False)

# ML Setup
def train_model(symptom_db, training_data):
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(training_data["symptome"].str.split(","))
    y = training_data["diagnose"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, mlb

# Vitalparameter-Verarbeitung
def parse_vitals(vitals_string):
    """Extrahiert Vitalparameter aus einem String wie 'HR:80,BP:120/80,T:36.8,SpO2:98'"""
    if not vitals_string or vitals_string.strip() == "":
        return {}
    
    vitals = {}
    pairs = vitals_string.split(',')
    
    for pair in pairs:
        pair = pair.strip()
        if ':' in pair:
            key, value = pair.split(':', 1)
            vitals[key.strip()] = value.strip()
    
    return vitals

def check_vitals(vitals, alter="Erwachsener"):
    """Prüft Vitalparameter auf kritische Werte und gibt Warnungen zurück"""
    warnings = []
    
    # Altersbasierte Referenzwerte
    ist_kind = alter in ["Säugling", "Kleinkind", "Kind"]
    ist_saugling = alter == "Säugling"
    
    if 'HR' in vitals:
        try:
            hr = int(vitals['HR'])
            if ist_saugling:
                if hr > 160:
                    warnings.append("Tachykardie (HR > 160 bei Säugling)")
                elif hr < 100:
                    warnings.append("Bradykardie (HR < 100 bei Säugling)")
            elif ist_kind:
                if hr > 140:
                    warnings.append("Tachykardie (HR > 140 bei Kind)")
                elif hr < 80:
                    warnings.append("Bradykardie (HR < 80 bei Kind)")
            else:
                if hr > 120:
                    warnings.append("Tachykardie (HR > 120)")
                elif hr < 50:
                    warnings.append("Bradykardie (HR < 50)")
        except ValueError:
            pass
    
    if 'BP' in vitals and '/' in vitals['BP']:
        try:
            systolic, diastolic = map(int, vitals['BP'].split('/'))
            if ist_kind:
                if systolic > 130:
                    warnings.append("Erhöhter Blutdruck (Systolisch > 130 bei Kind)")
                elif systolic < 80:
                    warnings.append("Hypotonie (Systolisch < 80 bei Kind)")
            else:
                if systolic > 180:
                    warnings.append("Schwere Hypertonie (Systolisch > 180)")
                elif systolic < 90:
                    warnings.append("Hypotonie (Systolisch < 90)")
        except ValueError:
            pass
    
    if 'T' in vitals:
        try:
            temp = float(vitals['T'])
            if temp >= 39.5:
                warnings.append("Hohes Fieber (T ≥ 39.5°C)")
            elif temp <= 35.5:
                warnings.append("Hypothermie (T ≤ 35.5°C)")
        except ValueError:
            pass
    
    if 'SpO2' in vitals:
        try:
            spo2 = int(vitals['SpO2'])
            if ist_kind:
                if spo2 < 94:
                    warnings.append("Sauerstoffsättigung kritisch (SpO2 < 94% bei Kind)")
            else:
                if spo2 < 92:
                    warnings.append("Sauerstoffsättigung kritisch (SpO2 < 92%)")
        except ValueError:
            pass
    
    return warnings

def adjust_diagnosis_with_vitals(diagnosen, vitals, alter="Erwachsener"):
    """Passt Diagnosewahrscheinlichkeiten basierend auf Vitalparametern an"""
    if not vitals:
        return diagnosen
    
    # Kopie der Diagnosen, um die Originale nicht zu verändern
    adjusted_diagnosen = diagnosen.copy()
    
    # Ist es ein pädiatrischer Fall?
    ist_kind = alter in ["Säugling", "Kleinkind", "Kind"]
    ist_saugling = alter == "Säugling"
    
    # Anpassung für Fieber
    if 'T' in vitals:
        try:
            temp = float(vitals['T'])
            # Allgemeine Anpassungen
            if temp >= 38.5 and "Lungenentzündung" in adjusted_diagnosen:
                adjusted_diagnosen["Lungenentzündung"] = min(100, adjusted_diagnosen["Lungenentzündung"] * 1.2)
            if temp >= 39.0 and "Virusgrippe" in adjusted_diagnosen:
                adjusted_diagnosen["Virusgrippe"] = min(100, adjusted_diagnosen["Virusgrippe"] * 1.2)
                
            # Pädiatrische Anpassungen
            if ist_kind:
                if temp >= 38.5 and "Bronchiolitis" in adjusted_diagnosen:
                    adjusted_diagnosen["Bronchiolitis"] = min(100, adjusted_diagnosen["Bronchiolitis"] * 1.3)
                if temp >= 39.0 and "Pseudokrupp" in adjusted_diagnosen:
                    adjusted_diagnosen["Pseudokrupp"] = min(100, adjusted_diagnosen["Pseudokrupp"] * 1.2)
                if temp >= 39.0 and "Otitis media" in adjusted_diagnosen:
                    adjusted_diagnosen["Otitis media"] = min(100, adjusted_diagnosen["Otitis media"] * 1.3)
                if temp >= 39.5 and "Meningitis" in adjusted_diagnosen:
                    adjusted_diagnosen["Meningitis"] = min(100, adjusted_diagnosen["Meningitis"] * 1.4)
        except ValueError:
            pass
    
    # Anpassung für Sauerstoffsättigung
    if 'SpO2' in vitals:
        try:
            spo2 = int(vitals['SpO2'])
            # Allgemeine Anpassungen
            if spo2 < 92 and "Pulmonalembolie" in adjusted_diagnosen:
                adjusted_diagnosen["Pulmonalembolie"] = min(100, adjusted_diagnosen["Pulmonalembolie"] * 1.3)
            if spo2 < 90 and "Lungenentzündung" in adjusted_diagnosen:
                adjusted_diagnosen["Lungenentzündung"] = min(100, adjusted_diagnosen["Lungenentzündung"] * 1.2)
                
            # Pädiatrische Anpassungen
            if ist_kind:
                if spo2 < 94 and "Bronchiolitis" in adjusted_diagnosen:
                    adjusted_diagnosen["Bronchiolitis"] = min(100, adjusted_diagnosen["Bronchiolitis"] * 1.4)
                if spo2 < 93 and "RSV-Bronchiolitis" in adjusted_diagnosen:
                    adjusted_diagnosen["RSV-Bronchiolitis"] = min(100, adjusted_diagnosen["RSV-Bronchiolitis"] * 1.4)
                if spo2 < 92 and "Pneumonie bei Kindern" in adjusted_diagnosen:
                    adjusted_diagnosen["Pneumonie bei Kindern"] = min(100, adjusted_diagnosen["Pneumonie bei Kindern"] * 1.5)
        except ValueError:
            pass
    
    # Anpassung für Herzfrequenz
    if 'HR' in vitals:
        try:
            hr = int(vitals['HR'])
            # Allgemeine Anpassungen
            if hr > 110 and "Akuter Herzinfarkt" in adjusted_diagnosen:
                adjusted_diagnosen["Akuter Herzinfarkt"] = min(100, adjusted_diagnosen["Akuter Herzinfarkt"] * 1.3)
            if hr > 120 and "Lungenembolie" in adjusted_diagnosen:
                adjusted_diagnosen["Lungenembolie"] = min(100, adjusted_diagnosen["Lungenembolie"] * 1.4)
                
            # Pädiatrische Anpassungen
            if ist_kind:
                if hr > 120 and "Pseudokrupp" in adjusted_diagnosen:
                    adjusted_diagnosen["Pseudokrupp"] = min(100, adjusted_diagnosen["Pseudokrupp"] * 1.2)
                if hr > 130 and "RSV-Bronchiolitis" in adjusted_diagnosen:
                    adjusted_diagnosen["RSV-Bronchiolitis"] = min(100, adjusted_diagnosen["RSV-Bronchiolitis"] * 1.3)
                if hr > 140 and "Meningitis" in adjusted_diagnosen:
                    adjusted_diagnosen["Meningitis"] = min(100, adjusted_diagnosen["Meningitis"] * 1.5)
        except ValueError:
            pass
    
    # Normalisiere die Werte wieder zu Prozentwerten
    total = sum(adjusted_diagnosen.values())
    if total > 0:
        for key in adjusted_diagnosen:
            adjusted_diagnosen[key] = round((adjusted_diagnosen[key] / total) * 100, 1)
    
    return adjusted_diagnosen

# Symptome aus Text mit DB und Synonymen abgleichen
def symptome_abgleichen(eingabe, symptom_db):
    eingabe_liste = [s.strip().lower() for s in eingabe.split(",")]
    symptom_ids = []
    unmatched = []
    
    for symptom in eingabe_liste:
        if not symptom:  # Überspringe leere Strings
            continue
            
        found = False
        for sid, data in symptom_db.items():
            if symptom in data["name"].lower() or any(syn.lower() in symptom for syn in data.get("synonyme", [])):
                symptom_ids.append(sid)
                found = True
                break
        if not found:
            unmatched.append(symptom)
    
    return symptom_ids, unmatched

# Wahrscheinlichkeiten für Diagnosen berechnen
def diagnostizieren(symptom_ids, symptom_db, model, mlb):
    if not symptom_ids:
        return {"Fehler": "Keine gültigen Symptome erkannt"}
    
    symptom_namen = [symptom_db[sid]["name"] for sid in symptom_ids]
    eingabe_vektor = mlb.transform([symptom_namen])
    wahrscheinlichkeiten = model.predict_proba(eingabe_vektor)[0]
    diagnosen = model.classes_
    
    return {diagnose: round(prob * 100, 1) for diagnose, prob in zip(diagnosen, wahrscheinlichkeiten) if prob > 0}

# Medizinisches Wissen anwenden
def apply_medical_rules(symptome, vitals, alter=None, zusatzinfos=""):
    zusatz_diagnosen = {}
    symptom_list = [s.strip().lower() for s in symptome.split(",")]
    
    # Leitsymptome identifizieren und höher gewichten
    leitsymptome = {
        "brustschmerzen": ["Akuter Herzinfarkt", "Angina pectoris", "Lungenembolie"],
        "atemnot": ["Akuter Herzinfarkt", "Lungenembolie", "Asthma-Anfall", "Pneumonie"],
        "starke kopfschmerzen": ["Meningitis", "Schlaganfall", "Subarachnoidalblutung"],
        "lähmungserscheinungen": ["Schlaganfall", "Multiple Sklerose"],
        "sprachstörungen": ["Schlaganfall", "Transitorische ischämische Attacke"],
        "bewusstlosigkeit": ["Synkope", "Epilepsie", "Hypoglykämie"]
    }
    
    # Prüfen auf Leitsymptome
    for symptom in symptom_list:
        for leitsymptom, diagnosen in leitsymptome.items():
            if leitsymptom in symptom:
                for diagnose in diagnosen:
                    zusatz_diagnosen[diagnose] = zusatz_diagnosen.get(diagnose, 0) + 35.0  # Höhere Basiswahrscheinlichkeit

    
    # Zusatzinfos parsen (für Vorerkrankungen etc.)
    vorerkrankungen = []
    if "vorerkrankungen:" in zusatzinfos.lower():
        vorerkrankungs_text = re.search(r"vorerkrankungen:\s*(.*?)(?:\n|$)", zusatzinfos, re.IGNORECASE)
        if vorerkrankungs_text:
            vorerkrankungen = [v.strip().lower() for v in vorerkrankungs_text.group(1).split(",")]
    
    # Untersuchungsergebnisse parsen
    untersuchungsergebnisse = {}
    if "untersuchungsergebnisse:" in zusatzinfos.lower():
        untersuchungs_text = re.search(r"untersuchungsergebnisse:\s*(.*?)(?:\n|$)", zusatzinfos, re.IGNORECASE)
        if untersuchungs_text:
            untersuchungs_items = untersuchungs_text.group(1).split(",")
            for item in untersuchungs_items:
                if ":" in item:
                    key, value = item.split(":", 1)
                    untersuchungsergebnisse[key.strip().lower()] = value.strip().lower()
    
    # Laborwerte parsen
    laborwerte = {}
    if "laborwerte:" in zusatzinfos.lower():
        labor_text = re.search(r"laborwerte:\s*(.*?)(?:\n|$)", zusatzinfos, re.IGNORECASE)
        if labor_text:
            labor_items = labor_text.group(1).split(",")
            for item in labor_items:
                if ":" in item:
                    key, value = item.split(":", 1)
                    laborwerte[key.strip().lower()] = value.strip().lower()
    
    # Pädiatrische Regeln
    if alter in ["Säugling", "Kleinkind", "Kind"] or any(term in symptome.lower() for term in ["säugling", "kind", "kindlich", "bellend"]):
        if "husten" in " ".join(symptom_list) and any(term in symptome.lower() for term in ["bellend", "heiser", "heiserkeit"]):
            zusatz_diagnosen["Pseudokrupp"] = 80.0
        
        if "husten" in " ".join(symptom_list) and "fieber" in " ".join(symptom_list) and alter == "Säugling":
            zusatz_diagnosen["RSV-Bronchiolitis"] = 75.0
            zusatz_diagnosen["Virale Atemwegsinfektion"] = 65.0
            zusatz_diagnosen["Bronchiolitis"] = 70.0
        
        if "fieber" in " ".join(symptom_list) and any(term in symptome.lower() for term in ["atemnot", "kurzatmig", "atembeschwerden"]):
            if alter == "Säugling":
                zusatz_diagnosen["Bronchiolitis"] = 80.0
                zusatz_diagnosen["RSV-Bronchiolitis"] = 75.0
            else:
                zusatz_diagnosen["Bronchitis"] = 65.0
                zusatz_diagnosen["Pneumonie bei Kindern"] = 60.0
        
        if "fieber" in " ".join(symptom_list) and "ohrenschmerzen" in " ".join(symptom_list):
            zusatz_diagnosen["Akute Otitis media"] = 80.0
            
        # HNO-Untersuchungsergebnisse berücksichtigen
        if "hno" in untersuchungsergebnisse:
            hno_befund = untersuchungsergebnisse["hno"]
            if "trommelfell gerötet" in hno_befund or "trommelfell vorgewölbt" in hno_befund:
                zusatz_diagnosen["Akute Otitis media"] = 85.0
    
    # Erwachsene Regeln
    else:
        if "brustschmerzen" in " ".join(symptom_list) and "kurzatmigkeit" in " ".join(symptom_list):
            if vitals and 'HR' in vitals and int(vitals['HR']) > 100:
                zusatz_diagnosen["Akuter Herzinfarkt"] = 70.0
                zusatz_diagnosen["Akutes Koronarsyndrom"] = 65.0
            else:
                zusatz_diagnosen["Angina pectoris"] = 60.0
                
            # Cardiale Untersuchungsergebnisse berücksichtigen
            if "cor" in untersuchungsergebnisse:
                cor_befund = untersuchungsergebnisse["cor"]
                if "herzgeräusch" in cor_befund or "systolikum" in cor_befund:
                    zusatz_diagnosen["Herzklappenfehler"] = 55.0
                if "tachykardie" in cor_befund:
                    zusatz_diagnosen["Akuter Herzinfarkt"] = 75.0
        
        if "kopfschmerzen" in " ".join(symptom_list) and "lichtempfindlichkeit" in " ".join(symptom_list):
            zusatz_diagnosen["Migräne"] = 75.0
        
        if "husten" in " ".join(symptom_list) and "fieber" in " ".join(symptom_list) and "auswurf" in " ".join(symptom_list):
            zusatz_diagnosen["Bronchitis"] = 70.0
            zusatz_diagnosen["Pneumonie"] = 65.0
            
            # Pulmonale Untersuchungsergebnisse berücksichtigen
            if "pulmo" in untersuchungsergebnisse:
                pulmo_befund = untersuchungsergebnisse["pulmo"]
                if "rasselgeräusche" in pulmo_befund or "bronchial" in pulmo_befund:
                    zusatz_diagnosen["Pneumonie"] = 80.0
    
    # Vorerkrankungen berücksichtigen
    if vorerkrankungen:
        if any(v in ["asthma", "asthma bronchiale"] for v in vorerkrankungen):
            if "husten" in " ".join(symptom_list) or "atemnot" in " ".join(symptom_list):
                zusatz_diagnosen["Asthma-Exazerbation"] = 70.0
                
        if any(v in ["copd", "chronisch obstruktive lungenerkrankung"] for v in vorerkrankungen):
            if "husten" in " ".join(symptom_list) or "atemnot" in " ".join(symptom_list):
                zusatz_diagnosen["COPD-Exazerbation"] = 75.0
                
        if any(v in ["diabetes", "diabetes mellitus"] for v in vorerkrankungen):
            if "durst" in " ".join(symptom_list) or "polyurie" in " ".join(symptom_list):
                zusatz_diagnosen["Entgleister Diabetes"] = 65.0
    
    # Laborwerte berücksichtigen
    if laborwerte:
        if "crp" in laborwerte:
            try:
                crp_wert = float(laborwerte["crp"].split()[0])
                if crp_wert > 100:
                    if "fieber" in " ".join(symptom_list):
                        zusatz_diagnosen["Bakterielle Infektion"] = 80.0
                        if "husten" in " ".join(symptom_list):
                            zusatz_diagnosen["Pneumonie"] = 85.0
                        if "dysurie" in " ".join(symptom_list) or "pollakisurie" in " ".join(symptom_list):
                            zusatz_diagnosen["Pyelonephritis"] = 85.0
            except ValueError:
                pass
                
        if "leukozyten" in laborwerte:
            try:
                leuko_wert = float(laborwerte["leukozyten"].split("/")[0])
                if leuko_wert > 12:
                    zusatz_diagnosen["Bakterielle Infektion"] = max(65.0, zusatz_diagnosen.get("Bakterielle Infektion", 0))
                elif leuko_wert < 4:
                    zusatz_diagnosen["Virale Infektion"] = 60.0
            except ValueError:
                pass
    
    # Symptomkonstellationen berücksichtigen - klassische Symptomkombinationen stark gewichten
    # Herzinfarkt-Konstellation
    if "brustschmerzen" in " ".join(symptom_list) and "ausstrahlung" in " ".join(symptom_list) and "arm" in " ".join(symptom_list):
        zusatz_diagnosen["Akuter Herzinfarkt"] = max(90.0, zusatz_diagnosen.get("Akuter Herzinfarkt", 0))
    
    # Meningitis-Konstellation    
    if "fieber" in " ".join(symptom_list) and "nackensteifigkeit" in " ".join(symptom_list) and "kopfschmerzen" in " ".join(symptom_list):
        zusatz_diagnosen["Meningitis"] = max(90.0, zusatz_diagnosen.get("Meningitis", 0))
    
    # Schlaganfall-Konstellation
    if ("lähmung" in " ".join(symptom_list) or "schwäche einseitig" in " ".join(symptom_list)) and "sprachstörung" in " ".join(symptom_list):
        zusatz_diagnosen["Schlaganfall"] = max(85.0, zusatz_diagnosen.get("Schlaganfall", 0))
    
    # Appendizitis-Konstellation
    if "bauchschmerzen" in " ".join(symptom_list) and "erbrechen" in " ".join(symptom_list) and "rechtsseitig" in " ".join(symptom_list):
        zusatz_diagnosen["Appendizitis"] = max(80.0, zusatz_diagnosen.get("Appendizitis", 0))
    
    return zusatz_diagnosen

# Diagnosen basierend auf Alter filtern
# Funktion in core.py verbessern
def filter_unlikely_diagnoses(diagnosen, symptome, alter=None, vitals=None):
    """
    Erweiterte Funktion zum Filtern unwahrscheinlicher Diagnosen basierend auf 
    klinischen Faktoren und epidemiologischen Wahrscheinlichkeiten
    """
    symptom_list = symptome.lower().split(",")
    vitals_dict = {}
    
    # Vitalparameter parsen, falls vorhanden
    if vitals:
        pairs = vitals.split(',')
        for pair in pairs:
            if ':' in pair:
                key, value = pair.split(':', 1)
                vitals_dict[key.strip()] = value.strip()
    
    # Altersbasierte Ausschlussregeln mit klinischer Evidenz
    if alter in ["Säugling", "Kleinkind", "Kind"] or any(term in symptome.lower() for term in ["säugling", "kind", "kindlich"]):
        # Diagnosen, die bei Kindern extrem selten sind
        adult_only_diagnoses = [
            "Angina pectoris", "Myokardinfarkt", "Akuter Herzinfarkt", "Kardiomyopathie", 
            "Klimakterisches Syndrom", "Psoriasis", "Herzinsuffizienz", "Fazialisparese",
            "COPD", "Tiefe Beinvenenthrombose", "Lungenembolie", "Koronare Herzkrankheit"
        ]
        
        # Stärkere Reduktion (nicht nur Entfernung) von unplausiblen Diagnosen
        for diagnose in diagnosen.keys():
            if diagnose in adult_only_diagnoses:
                diagnosen[diagnose] = diagnosen[diagnose] * 0.01  # 99% reduzieren, nicht komplett entfernen
    else:
        # Erwachsene sollten keine pädiatrischen Diagnosen bekommen
        child_only_diagnoses = [
            "RSV-Bronchiolitis", "Pseudokrupp", "Bronchiolitis", "Epiglottitis",
            "Säuglingshusten", "Dreimonatskoliken", "Windeldermatitis"
        ]
        for diagnose in diagnosen.keys():
            if diagnose in child_only_diagnoses:
                diagnosen[diagnose] = diagnosen[diagnose] * 0.01
    
    # Symptomkonstellation für typische Krankheitsbilder erkennen
    # Myokardinfarkt-Konstellation
    if ("brustschmerzen" in " ".join(symptom_list) and 
        ("atemnot" in " ".join(symptom_list) or "kurzatmigkeit" in " ".join(symptom_list))):
        for diagnose in ["Akuter Herzinfarkt", "Myokardinfarkt"]:
            if diagnose in diagnosen:
                diagnosen[diagnose] *= 2.5  # Deutliche Verstärkung 

    # Schlaganfall-Konstellation
    if (("lähmung" in " ".join(symptom_list) or "schwäche" in " ".join(symptom_list)) and 
        ("sprachstörung" in " ".join(symptom_list) or "sprachverlust" in " ".join(symptom_list))):
        if "Schlaganfall" in diagnosen:
            diagnosen["Schlaganfall"] *= 3.0
    
    # Red Flags identifizieren und stark gewichten
    red_flags = {
        "plötzliche stärkste kopfschmerzen": ["Subarachnoidalblutung", "Meningitis"],
        "akuter vernichtungsschmerz": ["Aortendissektion", "Akuter Herzinfarkt"],
        "atemstillstand": ["Respiratorische Insuffizienz", "Kardiopulmonale Reanimation"],
        "bewusstlosigkeit plötzlich": ["Synkope", "Status epilepticus", "Hirnblutung"]
    }

    # Red Flags prüfen (vor der Normalisierung)
    for symptom in symptom_list:
        for flag, flag_diagnoses in red_flags.items():
            if flag in symptom:
                for diagnose in diagnosen.keys():
                    if diagnose in flag_diagnoses:
                        diagnosen[diagnose] *= 5.0  # Sehr starke Verstärkung für kritische Diagnosen
    
    # Vitalparameter-basierte Ausschlüsse
    if vitals_dict:
        # Hohes Fieber + normale Herzfrequenz macht Infarkt sehr unwahrscheinlich
        if 'T' in vitals_dict and 'HR' in vitals_dict:
            try:
                temp = float(vitals_dict['T'])
                hr = int(vitals_dict['HR'])
                
                if temp > 38.5 and hr < 100:
                    for diagnose in ["Akuter Herzinfarkt", "Myokardinfarkt"]:
                        if diagnose in diagnosen:
                            diagnosen[diagnose] = diagnosen[diagnose] * 0.1
                            
                # Hohes Fieber + erhöhte Herzfrequenz bei jungen Patienten = infektiöse Ursache wahrscheinlicher
                if temp > 38.5 and hr > 100 and alter in ["Kind", "Kleinkind", "Säugling"]:
                    infectious_diagnoses = ["Pneumonie", "Bronchiolitis", "Virale Infektion", "Bronchitis"]
                    for diagnose in diagnosen.keys():
                        if any(inf_dx in diagnose for inf_dx in infectious_diagnoses):
                            diagnosen[diagnose] = diagnosen[diagnose] * 2.0
            except (ValueError, TypeError):
                pass
    
    # Symptomkombinations-basierte Ausschlüsse
    if "dysurie" in " ".join(symptom_list) or "brennen beim wasserlassen" in " ".join(symptom_list):
        urological_diagnoses = ["Harnwegsinfektion", "Zystitis", "Pyelonephritis"]
        for diagnose in diagnosen.keys():
            if diagnose in urological_diagnoses:
                diagnosen[diagnose] = diagnosen[diagnose] * 3.0
            elif "kardial" in diagnose.lower() or "herz" in diagnose.lower() or "infarkt" in diagnose.lower():
                diagnosen[diagnose] = diagnosen[diagnose] * 0.02
    
    # Minimale Verbleib-Wahrscheinlichkeit
    min_threshold = 1.0  # 1% als minimale Grenze
    diagnosen = {k: max(v, min_threshold) for k, v in diagnosen.items() if v > min_threshold}
    
    # Normalisieren nach Anpassungen
    total = sum(diagnosen.values())
    if total > 0:
        diagnosen = {k: round(v / total * 100, 1) for k, v in diagnosen.items()}
    
    # Top-Diagnosen behalten (>2%)
    return {k: v for k, v in diagnosen.items() if v > 2.0}

# Behandlung für Top-Diagnose vorschlagen
def behandlung_vorschlagen(diagnosen):
    top_diagnose = max(diagnosen, key=diagnosen.get) if diagnosen else None
    
    behandlungen = {
        "Akuter Herzinfarkt": "Sofortige Notfallbehandlung: Aspirin, Nitroglycerin, ggf. Katheterlabor",
        "Lungenentzündung": "Antibiotika, Sauerstoff bei Bedarf, ausreichend Flüssigkeit",
        "Virusgrippe": "Ruhe, ausreichend Flüssigkeit, Fiebersenkung, symptomatische Therapie",
        "Bronchitis": "Bronchodilatatoren, Hustenlöser, evtl. Antibiotika bei bakterieller Infektion",
        "Pulmonalembolie": "Sofortige Hospitalisierung, Antikoagulantien, Sauerstofftherapie",
        "Perikarditis": "Entzündungshemmende Medikamente, Ruhe, kardiologische Kontrolle",
        "Appendizitis": "Chirurgische Konsiliaruntersuchung, ggf. Appendektomie",
        "Migräne": "Triptane, ruhige Umgebung, ausreichend Flüssigkeit, Ruhe",
        "RSV-Bronchiolitis": "Symptomatische Therapie, Atemunterstützung, Flüssigkeitszufuhr, Überwachung",
        "Pseudokrupp": "Corticosteroide, kühle feuchte Luft, Rachensprays, ggf. Adrenalin-Inhalation",
        "Bronchiolitis": "Symptomatische Therapie, Atemunterstützung, Flüssigkeitszufuhr",
        "Epiglottitis": "NOTFALL - Sofortige Krankenhauseinweisung, Intubationsbereitschaft, Antibiotika",
        "Akute Otitis media": "Analgetika, ggf. Antibiotika, Abschwellende Nasentropfen",
        "Gastroenteritis bei Kindern": "Flüssigkeitsersatz, Elektrolytlösung, leichte Kost",
        "Virale Atemwegsinfektion": "Symptomatische Therapie, Flüssigkeitszufuhr, Fiebersenkung",
        "Meningitis": "NOTFALL - Sofortige Krankenhauseinweisung, Antibiotika, Überwachung",
        "Masern": "Symptomatische Therapie, Isolation, Vitamin A, Flüssigkeitszufuhr",
        "Asthma-Exazerbation": "Inhalative Beta-2-Mimetika, systemische Steroide, Sauerstoff bei Bedarf",
        "COPD-Exazerbation": "Bronchodilatatoren, systemische Steroide, Antibiotika bei Infektion",
        "Entgleister Diabetes": "Flüssigkeits- und Elektrolyt-Ausgleich, Insulin, Ursachensuche",
        "Bakterielle Infektion": "Breitspektrum-Antibiotika nach Abnahme von Kulturen",
        "Pyelonephritis": "Antibiotika, ausreichend Flüssigkeit, Fiebersenkung",
        "Harnwegsinfektion": "Antibiotika, reichlich Flüssigkeit, evtl. Schmerzmittel",
        "Hyperthyreose": "Thyreostatika, Beta-Blocker bei Tachykardie, endokrinologische Kontrolle",
        "Depression": "Psychotherapie, ggf. Antidepressiva nach fachärztlicher Beurteilung"
    }
    
    return behandlungen.get(top_diagnose, "Facharzt konsultieren"), top_diagnose

# Abrechnungscode generieren
def abrechnungscode_erzeugen(diagnose):
    codes = {
        "Akuter Herzinfarkt": "I21.9",
        "Lungenentzündung": "J18.9",
        "Virusgrippe": "J10.8",
        "Bronchitis": "J40",
        "Pulmonalembolie": "I26.9",
        "Perikarditis": "I30.9",
        "Appendizitis": "K35.80",
        "Migräne": "G43.9",
        "RSV-Bronchiolitis": "J21.0",
        "Pseudokrupp": "J05.0",
        "Bronchiolitis": "J21.9",
        "Epiglottitis": "J05.1",
        "Akute Otitis media": "H66.9",
        "Gastroenteritis bei Kindern": "A09",
        "Virale Atemwegsinfektion": "J06.9",
        "Meningitis": "G03.9",
        "Masern": "B05.9",
        "Asthma-Exazerbation": "J45.901",
        "COPD-Exazerbation": "J44.1",
        "Entgleister Diabetes": "E10.9",
        "Bakterielle Infektion": "A49.9",
        "Pyelonephritis": "N10",
        "Harnwegsinfektion": "N39.0",
        "Hyperthyreose": "E05.9",
        "Depression": "F32.9"
    }
    return codes.get(diagnose, "Unbekannt")

# Speichern der Daten für DeSci
def an_desci_speichern(symptome, diagnosen, behandlung):
    fall = {
        "zeitstempel": datetime.now().isoformat(),
        "symptome": symptome,
        "diagnosen": diagnosen,
        "behandlung": behandlung
    }
    with open("desci_faelle.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(fall) + "\n")
    return "Fall für DeSci-Netzwerk gespeichert"

# Globale Initialisierung
SYMPTOM_DB = load_symptom_db()
add_pediatric_cases()  # Füge automatisch pädiatrische Fälle hinzu, falls notwendig
TRAINING_DATA = load_training_data()
MODEL, MLB = train_model(SYMPTOM_DB, TRAINING_DATA)

# Hauptfunktion für Patientenverarbeitung
def patient_verarbeiten(eingabe, vitals_string="", alter=None, zusatz_info=""):
    symptom_ids, unmatched = symptome_abgleichen(eingabe, SYMPTOM_DB)
    diagnosen = diagnostizieren(symptom_ids, SYMPTOM_DB, MODEL, MLB)
    
    if "Fehler" in diagnosen:
        return {"fehler": diagnosen["Fehler"], "nicht_erkannt": unmatched}
    
    # Vitaldaten parsen und prüfen
    vitals = parse_vitals(vitals_string)
    warnings = check_vitals(vitals, alter)
    
    # Medizinisches Wissen anwenden
    zusatz_diagnosen = apply_medical_rules(eingabe, vitals, alter, zusatz_info)
    
    # Diagnosen kombinieren
    for diagnose, wahrscheinlichkeit in zusatz_diagnosen.items():
        if diagnose in diagnosen:
            diagnosen[diagnose] = max(diagnosen[diagnose], wahrscheinlichkeit)
        else:
            diagnosen[diagnose] = wahrscheinlichkeit
    
    # Diagnosen anpassen und filtern
    diagnosen = adjust_diagnosis_with_vitals(diagnosen, vitals, alter)
    diagnosen = filter_unlikely_diagnoses(diagnosen, eingabe, alter)
    
    # Nur die Top-N Diagnosen behalten
    diagnosen = dict(sorted(diagnosen.items(), key=lambda x: x[1], reverse=True)[:8])
    
    behandlung, top_diagnose = behandlung_vorschlagen(diagnosen)
    abrechnungscode = abrechnungscode_erzeugen(top_diagnose)
    desci_ergebnis = an_desci_speichern(eingabe.split(","), diagnosen, behandlung)
    
    return {
        "diagnosen": diagnosen,
        "behandlung": behandlung,
        "abrechnungscode": abrechnungscode,
        "desci": desci_ergebnis,
        "nicht_erkannt": unmatched if unmatched else None,
        "vitals": vitals,
        "vital_warnings": warnings,
        "top_diagnose": top_diagnose,
        "alter": alter
    }

