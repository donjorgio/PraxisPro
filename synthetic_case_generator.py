"""
Synthetischer Fallgenerator für medizinische Diagnosesysteme
Generiert realistische medizinische Fälle basierend auf häufigen Diagnosegruppen
"""

import json
import time
import os
import pandas as pd
import random
from datetime import datetime
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("synthetic_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("synthetic_generator")

# Umgebungsvariablen laden
load_dotenv()

# OpenAI-Konfiguration
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Prüfen, ob API-Schlüssel geladen wurde
if not api_key:
    logger.error("Kein OpenAI API-Schlüssel gefunden. Bitte stelle sicher, dass der Schlüssel in der .env Datei korrekt gesetzt ist.")
    raise ValueError("API-Schlüssel fehlt. Bitte OPENAI_API_KEY in .env Datei setzen.")
else:
    logger.info("OpenAI API-Schlüssel erfolgreich geladen.")

# Kategorien für medizinische Fälle
CASE_CATEGORIES = {
    "kardiovaskulaer": {
        "title": "Kardiovaskuläre Fälle",
        "percentage": 15,
        "examples": ["Myokardinfarkt", "Angina pectoris", "Herzinsuffizienz", "Hypertensive Krise", "Vorhofflimmern"]
    },
    "respiratorisch": {
        "title": "Respiratorische Fälle",
        "percentage": 15,
        "examples": ["Pneumonie", "COPD-Exazerbation", "Asthma-Anfall", "Bronchitis", "Lungenembolie", "COVID-19"]
    },
    "gastrointestinal": {
        "title": "Gastrointestinale Fälle",
        "percentage": 10,
        "examples": ["Gastroenteritis", "Appendizitis", "Divertikulitis", "Cholezystitis", "Darmobstruktion"]
    },
    "neurologisch": {
        "title": "Neurologische Fälle", 
        "percentage": 8,
        "examples": ["Schlaganfall", "Migräne", "Krampfanfall", "Meningitis", "Benigner paroxysmaler Lagerungsschwindel"]
    },
    "infektionskrankheiten": {
        "title": "Infektionskrankheiten",
        "percentage": 12,
        "examples": ["Grippe", "Harnwegsinfekt", "Erysipel", "Gastroenteritis", "Tonsillitis", "Sinusitis"]
    },
    "muskuloskeletal": {
        "title": "Muskuloskelettale Fälle",
        "percentage": 10,
        "examples": ["Lumbago", "Ischialgie", "Fraktur", "Tendinitis", "Arthritis", "Bandscheibenleiden"]
    },
    "paediatrisch": {
        "title": "Pädiatrische Fälle",
        "percentage": 10,
        "examples": ["Otitis media", "Bronchiolitis", "Krupp", "Gastroenteritis bei Kindern", "Fieberkrampf", "Pädiatrische Pneumonie"]
    },
    "urologie_nephrologie": {
        "title": "Urologische/Nephrologische Fälle",
        "percentage": 8,
        "examples": ["Harnwegsinfekt", "Urolithiasis", "Harnretention", "Akutes Nierenversagen", "Prostatitis"]
    },
    "gynaekologie": {
        "title": "Gynäkologische Fälle",
        "percentage": 6,
        "examples": ["Dysmenorrhoe", "Vaginitis", "Extrauteringravidität", "Ovarialzyste", "Menorrhagie"]
    },
    "haut": {
        "title": "Dermatologische Fälle", 
        "percentage": 5,
        "examples": ["Urticaria", "Zoster", "Allergische Reaktion", "Erysipel", "Dermatitis", "Psoriasisschub"]
    },
    "endokrinologie": {
        "title": "Endokrinologische Fälle",
        "percentage": 4,
        "examples": ["Diabetische Ketoazidose", "Hypoglykämie", "Thyreotoxische Krise", "Hypothyreose", "Nebenniereninsuffizienz"]
    },
    "psychiatrie": {
        "title": "Psychiatrische Fälle",
        "percentage": 4,
        "examples": ["Akute Angststörung", "Depression", "Alkoholentzug", "Drogenintoxikation", "Akute Psychose"]
    },
    "trauma": {
        "title": "Traumatologische Fälle",
        "percentage": 5,
        "examples": ["Fraktur", "Luxation", "Schädel-Hirn-Trauma", "Wirbelsäulenverletzung", "Weichteilverletzung"]
    }
}

def create_case_prompt(category_key):
    """Erstellt einen detaillierten Prompt für die Generierung eines medizinischen Falls"""
    category = CASE_CATEGORIES[category_key]
    examples = ", ".join(category["examples"])
    
    prompt = f"""Als erfahrener Notfall- und Allgemeinmediziner generiere einen realistischen medizinischen Fall für die Kategorie "{category["title"]}".
    
    Mögliche Diagnosen in dieser Kategorie sind: {examples}. Du darfst auch andere passende Diagnosen aus der Kategorie {category["title"]} wählen.
    
    Der Fall sollte folgende Informationen im JSON-Format enthalten:
    {{
        "id": [eindeutige Fall-ID],
        "alter": [Alter des Patienten (realistisch für die Erkrankung)],
        "geschlecht": ["männlich" oder "weiblich"],
        "symptome": [
            [Liste von 3-7 Symptomen als Strings, die typisch für die Erkrankung sind]
        ],
        "vitalparameter": [String mit Vitalzeichen im Format "HR:80,BP:120/80,T:36.8,SpO2:98"],
        "vorerkrankungen": [
            [Liste relevanter Vorerkrankungen (0-3)]
        ],
        "vorherige_operationen": [
            [Liste relevanter vorheriger Operationen (0-2) oder "keine"]
        ],
        "befunde": [Beschreibung von Untersuchungsergebnissen],
        "enddiagnose": [Die korrekte Diagnose]
    }}
    
    WICHTIGE REGELN:
    1. Stelle sicher, dass die Symptome, Vitalparameter und Befunde eine realistische Variation aufweisen und zur Enddiagnose passen.
    2. Die Vitalparameter müssen im vorgegebenen Format sein und realistische Werte enthalten.
    3. Bei pädiatrischen Fällen passe Alter und Vitalparameter entsprechend an.
    4. Berücksichtige bekannte Komorbiditäten und typische Begleiterkrankungen.
    5. Gib nur das JSON-Objekt zurück, keine Einleitung oder weiteren Text.
    6. Das JSON muss syntaktisch korrekt und gültig sein.
    7. Verwende nur übliche und DRINGLICH WICHTIG wirklich existierende, präzise medizinische Diagnosen, keine erfundenen.
    
    Liefere nur das reine JSON ohne weitere Erklärungen.
    """
    return prompt

def validate_case_prompt(case_json):
    """Erstellt einen Prompt zur Validierung eines generierten Falls"""
    prompt = f"""Prüfe als erfahrener Notfall- und Allgemeinmediziner den folgenden medizinischen Fall auf klinische Plausibilität und Realismus:

    {json.dumps(case_json, indent=2, ensure_ascii=False)}
    
    Bewerte folgende Aspekte:
    1. Sind die Symptome typisch und realistisch für die angegebene Diagnose?
    2. Passen die Vitalparameter zu den Symptomen und der Diagnose?
    3. Sind Alter, Geschlecht und Vorerkrankungen stimmig?
    4. Ist die Befundbeschreibung medizinisch korrekt?
    5. Gibt es Inkonsistenzen oder unplausible Angaben?
    
    Antworte im folgenden JSON-Format:
    {{
        "valid": [true/false],
        "score": [Bewertung von 1-10, wobei 10 höchst realistisch ist],
        "issues": [Liste von Problemen, falls vorhanden, sonst leere Liste],
        "suggestions": [Vorschläge zur Verbesserung, falls nötig]
    }}
    
    Liefere nur das reine JSON ohne weitere Erklärungen.
    """
    return prompt

def generate_case(category_key):
    """Generiert einen einzelnen medizinischen Fall für die angegebene Kategorie"""
    prompt = create_case_prompt(category_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        case_text = response.choices[0].message.content.strip()
        
        # Entferne eventuelle Markdown-Formatierung
        if case_text.startswith("```json"):
            case_text = case_text.replace("```json", "", 1)
        if case_text.endswith("```"):
            case_text = case_text.rsplit("```", 1)[0]
        
        case_text = case_text.strip()
        
        try:
            case = json.loads(case_text)
            logger.info(f"Fall generiert: {category_key} - {case.get('enddiagnose', 'Unbekannt')}")
            return case
        except json.JSONDecodeError as e:
            logger.error(f"JSON-Parsing-Fehler: {e}")
            logger.error(f"Problematischer Text: {case_text}")
            return None
        
    except Exception as e:
        logger.error(f"Fehler bei der Fallgenerierung für {category_key}: {str(e)}")
        return None

def validate_case(case):
    """Validiert einen generierten Fall auf medizinische Plausibilität"""
    if not case:
        return {"valid": False, "score": 0, "issues": ["Kein Fall zum Validieren"], "suggestions": []}
    
    prompt = validate_case_prompt(case)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        validation_text = response.choices[0].message.content.strip()
        
        # Entferne eventuelle Markdown-Formatierung
        if validation_text.startswith("```json"):
            validation_text = validation_text.replace("```json", "", 1)
        if validation_text.endswith("```"):
            validation_text = validation_text.rsplit("```", 1)[0]
            
        validation_text = validation_text.strip()
        
        try:
            validation = json.loads(validation_text)
            logger.info(f"Fall validiert: Score {validation.get('score', 0)}")
            return validation
        except json.JSONDecodeError as e:
            logger.error(f"Validierungs-JSON-Parsing-Fehler: {e}")
            logger.error(f"Problematischer Validierungstext: {validation_text}")
            return {"valid": False, "score": 0, "issues": [f"JSON-Parsing-Fehler: {str(e)}"], "suggestions": []}
        
    except Exception as e:
        logger.error(f"Fehler bei der Fallvalidierung: {str(e)}")
        return {"valid": False, "score": 0, "issues": [f"Fehler bei API-Aufruf: {str(e)}"], "suggestions": []}

def generate_batch(count=100, min_score=7):
    """Generiert einen Batch von medizinischen Fällen mit der angegebenen Verteilung"""
    cases = []
    validations = []
    stats = {cat: 0 for cat in CASE_CATEGORIES}
    
    # Stelle sicher, dass jede Kategorie mindestens einen Fall bekommt
    min_cases_per_category = 1
    
    # Berechne die Anzahl der Fälle pro Kategorie
    category_counts = {}
    
    # Wenn die Batch-Größe kleiner ist als die Anzahl der Kategorien, stellen wir sicher,
    # dass jede Kategorie zumindest eine faire Chance hat
    if count < len(CASE_CATEGORIES):
        # Wähle zufällig count Kategorien aus
        selected_cats = random.sample(list(CASE_CATEGORIES.keys()), count)
        for cat in CASE_CATEGORIES:
            category_counts[cat] = 1 if cat in selected_cats else 0
    else:
        # Erst mindestens einen Fall pro Kategorie zuweisen
        for cat in CASE_CATEGORIES:
            category_counts[cat] = min_cases_per_category
        
        # Verbleibende Fälle nach Prozentsatz verteilen
        remaining = count - (min_cases_per_category * len(CASE_CATEGORIES))
        
        if remaining > 0:
            for cat, info in CASE_CATEGORIES.items():
                # Zusätzliche Fälle basierend auf Prozentsatz zuweisen
                additional_count = int(remaining * (info["percentage"] / 100))
                category_counts[cat] += additional_count
                remaining -= additional_count
            
            # Verteile verbleibende Fälle auf Kategorien mit dem höchsten Prozentsatz
            if remaining > 0:
                sorted_cats = sorted(CASE_CATEGORIES.items(), key=lambda x: x[1]["percentage"], reverse=True)
                for i in range(remaining):
                    category_counts[sorted_cats[i % len(sorted_cats)][0]] += 1
    
    logger.info(f"Zielverteilung der Fälle: {category_counts}")
    
    # Generiere Fälle für jede Kategorie
    for category, target_count in category_counts.items():
        cat_cases = 0
        attempts = 0
        max_attempts = target_count * 2  # Maximal doppelt so viele Versuche wie Zielfälle
        
        logger.info(f"Generiere {target_count} Fälle für Kategorie '{CASE_CATEGORIES[category]['title']}'")
        
        while cat_cases < target_count and attempts < max_attempts:
            attempts += 1
            
            # Generiere und validiere einen Fall
            case = generate_case(category)
            if case:
                validation = validate_case(case)
                
                # Prüfe, ob der Fall den Qualitätsanforderungen entspricht
                if validation.get("valid", False) and validation.get("score", 0) >= min_score:
                    # Füge den Fall zur Liste hinzu
                    case["kategorie"] = category
                    case["validation_score"] = validation.get("score", 0)
                    cases.append(case)
                    validations.append(validation)
                    cat_cases += 1
                    stats[category] += 1
                    
                    logger.info(f"Fall {cat_cases}/{target_count} für '{category}' akzeptiert (Score: {validation.get('score', 0)})")
                else:
                    issues = ", ".join(validation.get("issues", ["Unbekannter Fehler"]))
                    logger.warning(f"Fall für '{category}' abgelehnt (Score: {validation.get('score', 0)}): {issues}")
            
            # Pause zwischen API-Aufrufen
            time.sleep(1)
        
        logger.info(f"Abgeschlossen: {cat_cases}/{target_count} Fälle für '{category}' generiert")
    
    logger.info(f"Generierung abgeschlossen: {len(cases)}/{count} Fälle insgesamt")
    logger.info(f"Tatsächliche Verteilung: {stats}")
    
    return cases, validations

def save_to_csv(cases, output_dir):
    """Speichert die generierten Fälle in CSV-Dateien im MIMIC-Format"""
    os.makedirs(output_dir, exist_ok=True)
    
    # patients.csv
    patients = []
    for case in cases:
        patient = {
            'patient_id': case.get('id', ''),
            'age': case.get('alter', 0),
            'gender': 'M' if case.get('geschlecht', '').lower() == 'männlich' else 'F',
            'category': case.get('kategorie', ''),
            'validation_score': case.get('validation_score', 0)
        }
        patients.append(patient)
    
    patients_df = pd.DataFrame(patients)
    patients_df.to_csv(os.path.join(output_dir, 'patients.csv'), index=False)
    
    # diagnoses.csv
    diagnoses = []
    for case in cases:
        diagnose = {
            'patient_id': case.get('id', ''),
            'diagnosis': case.get('enddiagnose', ''),
            'icd_code': ''  # ICD-Codes könnten separat generiert werden
        }
        diagnoses.append(diagnose)
    
    diagnoses_df = pd.DataFrame(diagnoses)
    diagnoses_df.to_csv(os.path.join(output_dir, 'diagnoses.csv'), index=False)
    
    # vitals.csv
    vitals = []
    for case in cases:
        vitals_str = case.get('vitalparameter', '')
        vitals_dict = {'patient_id': case.get('id', '')}
        
        if vitals_str:
            for pair in vitals_str.split(','):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    if key.strip() == 'HR':
                        vitals_dict['heart_rate'] = value.strip()
                    elif key.strip() == 'BP' and '/' in value:
                        sys, dia = value.strip().split('/')
                        vitals_dict['blood_pressure_systolic'] = sys
                        vitals_dict['blood_pressure_diastolic'] = dia
                    elif key.strip() == 'T':
                        vitals_dict['temperature'] = value.strip()
                    elif key.strip() == 'SpO2':
                        vitals_dict['oxygen_saturation'] = value.strip()
                    elif key.strip() == 'RR':
                        vitals_dict['respiratory_rate'] = value.strip()
        
        vitals.append(vitals_dict)
    
    vitals_df = pd.DataFrame(vitals)
    vitals_df.to_csv(os.path.join(output_dir, 'vitals.csv'), index=False)
    
    # Originaldaten speichern
    with open(os.path.join(output_dir, 'cases.json'), 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Daten gespeichert in {output_dir}")
    return {
        'patients_count': len(patients),
        'diagnoses_count': len(diagnoses),
        'vitals_count': len(vitals)
    }

def generate_patient_cases(total_count=100, batch_size=10, min_score=7, output_dir='mimic_data'):
    """Hauptfunktion zur Generierung der synthetischen Patientenfälle"""
    start_time = datetime.now()
    
    # Prüfen, ob bereits Fälle vorhanden sind und diese laden
    cases_file = os.path.join(output_dir, 'cases.json')
    all_cases = []
    all_validations = []
    
    if os.path.exists(cases_file):
        try:
            with open(cases_file, 'r', encoding='utf-8') as f:
                all_cases = json.load(f)
            logger.info(f"Bestehende Daten geladen: {len(all_cases)} Fälle gefunden")
            print(f"Bestehende Daten geladen: {len(all_cases)} Fälle gefunden")
        except Exception as e:
            logger.error(f"Fehler beim Laden bestehender Daten: {str(e)}")
            all_cases = []
    
    existing_count = len(all_cases)
    target_count = existing_count + total_count
    
    logger.info(f"Ziel: {target_count} Fälle (bereits {existing_count} vorhanden, {total_count} zu generieren)")
    print(f"Ziel: {target_count} Fälle (bereits {existing_count} vorhanden, {total_count} zu generieren)")
    
    remaining = total_count
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        current_progress = existing_count + (total_count - remaining)
        logger.info(f"Generiere Batch mit {current_batch} Fällen ({remaining} verbleibend, Fortschritt: {current_progress}/{target_count})")
        print(f"Generiere Batch mit {current_batch} Fällen ({remaining} verbleibend, Fortschritt: {current_progress}/{target_count})")
        
        batch_cases, batch_validations = generate_batch(current_batch, min_score)
        
        all_cases.extend(batch_cases)
        all_validations.extend(batch_validations)
        
        remaining -= len(batch_cases)
        
        logger.info(f"Batch abgeschlossen: {len(batch_cases)} Fälle generiert")
        
        # Zwischenspeichern nach jedem Batch
        save_to_csv(all_cases, output_dir)
        logger.info(f"Zwischenspeicherung: {len(all_cases)} Fälle gespeichert")
        print(f"Zwischenspeicherung: {len(all_cases)}/{target_count} Fälle gespeichert ({(len(all_cases)/target_count)*100:.1f}%)")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    logger.info(f"Generierung abgeschlossen: {len(all_cases)} Fälle in {duration:.1f} Minuten")
    
    # Finale Speicherung
    stats = save_to_csv(all_cases, output_dir)
    
    # Generiere Zusammenfassung
    category_counts = {}
    for case in all_cases:
        category = case.get('kategorie', 'unbekannt')
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1
    
    logger.info("Zusammenfassung der generierten Fälle:")
    logger.info(f"Gesamtzahl: {len(all_cases)}")
    logger.info(f"Verteilung nach Kategorien: {category_counts}")
    logger.info(f"Durchschnittlicher Validierungsscore: {sum(v.get('score', 0) for v in all_validations) / len(all_validations) if all_validations else 0}")
    
    return {
        "cases_count": len(all_cases),
        "duration_minutes": duration,
        "categories": category_counts,
        "stats": stats
    }

if __name__ == "__main__":
    # Standardparameter
    total_cases = 1000  # Anzahl der neu zu generierenden Fälle
    batch_size = 20     # Batch-Größe für die Generierung
    max_total = 5000    # Maximale Gesamtzahl der Fälle
    min_quality_score = 7  # Mindestqualität (1-10)
    output_directory = "mimic_data"
    
    # Prüfen, wie viele Fälle bereits existieren
    cases_file = os.path.join(output_directory, 'cases.json')
    existing_count = 0
    
    if os.path.exists(cases_file):
        try:
            with open(cases_file, 'r', encoding='utf-8') as f:
                existing_cases = json.load(f)
                existing_count = len(existing_cases)
        except Exception as e:
            print(f"Fehler beim Laden bestehender Daten: {str(e)}")
    
    # Berechne, wie viele weitere Fälle generiert werden sollen
    cases_to_generate = min(total_cases, max_total - existing_count)
    
    if cases_to_generate <= 0:
        print(f"Bereits {existing_count} Fälle generiert (Ziel: {max_total}). Keine weiteren Fälle werden generiert.")
    else:
        print(f"Generiere {cases_to_generate} weitere Fälle (Aktuell: {existing_count}, Ziel: {max_total})")
        
        result = generate_patient_cases(
            total_count=cases_to_generate, 
            batch_size=batch_size, 
            min_score=min_quality_score, 
            output_dir=output_directory
        )
    if cases_to_generate > 0:
        print(f"\nGENERIERUNG ABGESCHLOSSEN\n{'='*30}")
        print(f"Neu generierte Fälle: {result['cases_count']}")
        print(f"Dauer: {result['duration_minutes']:.1f} Minuten")
        print(f"Kategorieverteilung der neuen Fälle: {result['categories']}")
        print(f"Gesamtzahl der Fälle: {existing_count + result['cases_count']}")
        print(f"Dateien gespeichert in: {output_directory}")
    else:
        print(f"\nKEINE NEUEN FÄLLE GENERIERT\n{'='*30}")
        print(f"Gesamtzahl vorhandener Fälle: {existing_count}")
        print(f"Dateien befinden sich in: {output_directory}")