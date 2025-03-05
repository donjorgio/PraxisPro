# improved_hybrid_model.py - Verbesserte Integration von ML und LLM mit Fallback-Mechanismen

import json
import os
import requests
import logging
import time
import re
from datetime import datetime
from dotenv import load_dotenv
from core import patient_verarbeiten
from mimic_integration import MIMICIntegration

# Umgebungsvariablen laden (.env Datei)
load_dotenv()

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hybrid_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hybrid_model")

# Konfiguration der LLM-Provider
# Unterstützt multiple Anbieter für Robustheit/Fallback
LLM_CONFIGS = {
    "openai": {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "models": {
            "default": "gpt-3.5-turbo",
            "advanced": "gpt-4"
        },
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "max_tokens": 800,
        "temperature": 0.3
    },
    "anthropic": {
        "api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "models": {
            "default": "claude-instant-1.2",
            "advanced": "claude-2"
        },
        "endpoint": "https://api.anthropic.com/v1/complete",
        "max_tokens": 800
    },
    "cohere": {
        "api_key": os.environ.get("COHERE_API_KEY"),
        "models": {
            "default": "command",
            "advanced": "command"
        },
        "endpoint": "https://api.cohere.ai/v1/generate",
        "max_tokens": 800
    }
}

# MIMIC-Integration initialisieren
mimic = MIMICIntegration()

def get_available_llm_provider():
    """
    Ermittelt den ersten verfügbaren LLM-Provider basierend auf API-Schlüsseln.
    
    Returns:
        tuple: (provider_name, config) oder (None, None) wenn kein Provider verfügbar
    """
    for provider, config in LLM_CONFIGS.items():
        if config["api_key"]:
            return provider, config
    
    return None, None

def call_llm_service(prompt, provider=None, model_type="default"):
    """
    Ruft einen LLM-Service auf, um eine KI-basierte Antwort zu erhalten.
    Unterstützt automatischen Fallback auf verfügbare Provider.
    
    Args:
        prompt (str): Der Prompt/die Frage an das LLM
        provider (str, optional): Spezifischer LLM-Provider 
        model_type (str): "default" oder "advanced" für komplexe medizinische Fälle
        
    Returns:
        str or dict: LLM-Antwort oder Fehlerobjekt
    """
    # Wenn kein spezifischer Provider angegeben wurde, ersten verfügbaren nehmen
    if not provider:
        provider, config = get_available_llm_provider()
        if not provider:
            logger.warning("Kein LLM-Provider mit API-Schlüssel konfiguriert.")
            return {"error": "Kein LLM-Provider verfügbar", "details": "API-Schlüssel fehlen"}
    else:
        config = LLM_CONFIGS.get(provider)
        if not config or not config["api_key"]:
            logger.warning(f"Der gewählte Provider '{provider}' ist nicht verfügbar.")
            return {"error": f"Provider '{provider}' nicht verfügbar", "details": "API-Schlüssel fehlt"}
    
    # Provider-spezifische Anfrage vorbereiten
    try:
        if provider == "openai":
            return _call_openai_api(prompt, config, model_type)
        elif provider == "anthropic":
            return _call_anthropic_api(prompt, config, model_type)
        elif provider == "cohere":
            return _call_cohere_api(prompt, config, model_type)
        else:
            logger.error(f"Unbekannter Provider: {provider}")
            return {"error": f"Unbekannter Provider", "details": provider}
    
    except Exception as e:
        logger.error(f"Fehler beim Aufruf des LLM-Service ({provider}): {str(e)}")
        
        # Fallback auf anderen Provider, falls verfügbar
        for fallback_provider, fallback_config in LLM_CONFIGS.items():
            if fallback_provider != provider and fallback_config["api_key"]:
                logger.info(f"Versuche Fallback auf Provider: {fallback_provider}")
                try:
                    return call_llm_service(prompt, fallback_provider, model_type)
                except Exception as fallback_error:
                    logger.error(f"Auch Fallback auf {fallback_provider} fehlgeschlagen: {str(fallback_error)}")
        
        return {"error": f"Fehler bei LLM-Anfrage", "details": str(e)}

def _call_openai_api(prompt, config, model_type):
    """Ruft die OpenAI API auf"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}"
    }
    
    model = config["models"][model_type]
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"]
    }
    
    logger.info(f"Sende Anfrage an OpenAI API mit Modell {model}")
    response = requests.post(
        config["endpoint"],
        headers=headers,
        json=data,
        timeout=30
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        logger.error(f"OpenAI API-Fehler: {response.status_code}, {response.text}")
        raise Exception(f"OpenAI API-Fehler: {response.status_code}, {response.text}")

def _call_anthropic_api(prompt, config, model_type):
    """Ruft die Anthropic Claude API auf"""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": config["api_key"]
    }
    
    model = config["models"][model_type]
    
    data = {
        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        "model": model,
        "max_tokens_to_sample": config["max_tokens"],
        "temperature": 0.3,
        "stop_sequences": ["\n\nHuman:"]
    }
    
    logger.info(f"Sende Anfrage an Anthropic API mit Modell {model}")
    response = requests.post(
        config["endpoint"],
        headers=headers,
        json=data,
        timeout=30
    )
    
    if response.status_code == 200:
        return response.json()["completion"]
    else:
        logger.error(f"Anthropic API-Fehler: {response.status_code}, {response.text}")
        raise Exception(f"Anthropic API-Fehler: {response.status_code}, {response.text}")

def _call_cohere_api(prompt, config, model_type):
    """Ruft die Cohere API auf"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}"
    }
    
    model = config["models"][model_type]
    
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": config["max_tokens"],
        "temperature": 0.3,
        "k": 0,
        "p": 0.75
    }
    
    logger.info(f"Sende Anfrage an Cohere API mit Modell {model}")
    response = requests.post(
        config["endpoint"],
        headers=headers,
        json=data,
        timeout=30
    )
    
    if response.status_code == 200:
        return response.json()["generations"][0]["text"]
    else:
        logger.error(f"Cohere API-Fehler: {response.status_code}, {response.text}")
        raise Exception(f"Cohere API-Fehler: {response.status_code}, {response.text}")

def get_medical_prompt(symptome, vitals_string, alter, zusatz_info=""):
    """
    Erstellt einen medizinischen Prompt für das LLM mit verbessertem Kontext und 
    strukturiertem Format für bessere Parsing-Ergebnisse.
    
    Returns:
        str: Formatierter Prompt für medizinische LLM-Anfrage
    """
    # Basisinformationen zum Patienten
    prompt = f"""
    Du bist ein erfahrener Arzt und Diagnosespezialist. Führe eine präzise Diagnose durch,
    basierend auf den folgenden klinischen Informationen:

    PATIENT:
    Alter: {alter}
    Hauptsymptome: {symptome}
    """
    
    # Vitalparameter hinzufügen, falls vorhanden
    if vitals_string:
        # Vitalparameter in ein besser lesbares Format konvertieren
        vitals = {}
        for pair in vitals_string.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                vitals[key.strip()] = value.strip()
        
        prompt += "\nVITALPARAMETER:\n"
        if 'HR' in vitals: prompt += f"- Herzfrequenz: {vitals['HR']} /min\n"
        if 'BP' in vitals: prompt += f"- Blutdruck: {vitals['BP']} mmHg\n"
        if 'T' in vitals: prompt += f"- Temperatur: {vitals['T']} °C\n"
        if 'SpO2' in vitals: prompt += f"- Sauerstoffsättigung: {vitals['SpO2']} %\n"
    
    # Zusatzinformationen hinzufügen, falls vorhanden
    if zusatz_info:
        # Vorerkrankungen extrahieren und strukturieren
        vorerkrankungen = []
        if "vorerkrankungen:" in zusatz_info.lower():
            match = re.search(r"vorerkrankungen:\s*(.*?)(?:\n|$)", zusatz_info, re.IGNORECASE)
            if match:
                vorerkrankungen = [v.strip() for v in match.group(1).split(",")]
                prompt += "\nVORERKRANKUNGEN:\n"
                for ve in vorerkrankungen:
                    prompt += f"- {ve}\n"
        
        # Untersuchungsergebnisse extrahieren und strukturieren
        if "untersuchungsergebnisse:" in zusatz_info.lower():
            match = re.search(r"untersuchungsergebnisse:\s*(.*?)(?:\n|$)", zusatz_info, re.IGNORECASE)
            if match:
                prompt += "\nUNTERSUCHUNGSERGEBNISSE:\n"
                ergebnisse = match.group(1).split(",")
                for ergebnis in ergebnisse:
                    if ":" in ergebnis:
                        bereich, befund = ergebnis.split(":", 1)
                        prompt += f"- {bereich.strip()}: {befund.strip()}\n"
                    else:
                        prompt += f"- {ergebnis.strip()}\n"
        
        # Laborwerte extrahieren und strukturieren
        if "laborwerte:" in zusatz_info.lower():
            match = re.search(r"laborwerte:\s*(.*?)(?:\n|$)", zusatz_info, re.IGNORECASE)
            if match:
                prompt += "\nLABORWERTE:\n"
                laborwerte = match.group(1).split(",")
                for laborwert in laborwerte:
                    if ":" in laborwert:
                        test, wert = laborwert.split(":", 1)
                        prompt += f"- {test.strip()}: {wert.strip()}\n"
                    else:
                        prompt += f"- {laborwert.strip()}\n"
        
        # Andere wichtige Zusatzinformationen
        if not any(keyword in zusatz_info.lower() for keyword in ["vorerkrankungen:", "untersuchungsergebnisse:", "laborwerte:"]):
            prompt += f"\nZUSÄTZLICHE INFORMATIONEN:\n{zusatz_info}\n"
    
    # Klinische Überlegungen basierend auf Symptommustern
    symptom_text = symptome.lower()
    
    # Kinderspezifische Symptome gesondert hervorheben
    if alter in ["Säugling", "Kleinkind", "Kind"] or any(term in symptom_text for term in ["säugling", "kind", "kindlich"]):
        prompt += "\nWICHTIG: Es handelt sich um einen pädiatrischen Fall. Berücksichtige altersspezifische Diagnosen und Normwerte.\n"
    
    # Prompt optimieren für spezifische Symptommuster mit verstärkter Gewichtung
    if "brustschmerzen" in symptom_text and "ausstrahlung" in symptom_text:
        prompt += "\nBEACHTE: Die Brustschmerzen mit Ausstrahlung sind ein hochspezifisches Leitsymptom für ein akutes Koronarsyndrom. Gewichte dies besonders stark in deiner Diagnose.\n"
    
    if "brustschmerzen" in symptom_text and ("atemnot" in symptom_text or "kurzatmigkeit" in symptom_text):
        prompt += "\nBEACHTE: Die Kombination aus Brustschmerzen und Atemnot ist hochverdächtig für einen akuten Herzinfarkt und sollte als Leitsymptomkonstellation priorisiert werden.\n"
    
    if ("dysurie" in symptom_text or "brennen" in symptom_text) and "fieber" in symptom_text:
        prompt += "\nBEACHTE: Die Kombination aus Miktionsbeschwerden und Fieber ist typisch für eine Harnwegsinfektion mit möglichem Aufstieg (Pyelonephritis).\n"
    
    if "ausschlag" in symptom_text and "juckreiz" in symptom_text:
        prompt += "\nBEACHTE: Der juckende Hautausschlag deutet auf eine allergische Reaktion hin. Prüfe auf weitere Symptome einer systemischen allergischen Reaktion.\n"
    
    if "fieber" in symptom_text and "nackensteifigkeit" in symptom_text:
        prompt += "\nWARNUNG: Die Kombination aus Fieber und Nackensteifigkeit ist hochverdächtig für eine Meningitis und sollte mit höchster Priorität betrachtet werden!\n"
    
    if ("lähmung" in symptom_text or "schwäche" in symptom_text) and ("sprachstörungen" in symptom_text or "aphasie" in symptom_text):
        prompt += "\nWARNUNG: Die Kombination aus Lähmungserscheinungen und Sprachstörungen ist ein klassisches Zeichen für einen akuten Schlaganfall. Betrachte dies als hochprioritär!\n"
    
    if "bauchschmerzen" in symptom_text and "rechtsseitig" in symptom_text and "übelkeit" in symptom_text:
        prompt += "\nBEACHTE: Rechtsseitige Unterbauchschmerzen mit Übelkeit sind typische Symptome einer Appendizitis.\n"
    
    # Hinweis zur Symptomgewichtung für das LLM
    prompt += """
    WICHTIG: Berücksichtige bei deiner Diagnose die Gewichtung der Symptome. Leitsymptome wie Brustschmerzen 
    bei Verdacht auf Herzinfarkt oder Nackensteifigkeit bei Meningitis sollten stärker gewichtet werden 
    als unspezifische Symptome wie allgemeine Schwäche oder leichtes Unwohlsein.
    
    Typische Symptomkonstellationen wie:
    - Brustschmerzen + Ausstrahlung in den linken Arm + Atemnot → Myokardinfarkt (hohe Konfidenz)
    - Fieber + Nackensteifigkeit + Kopfschmerzen → Meningitis (hohe Konfidenz)
    - Einseitige Lähmung + Sprachstörungen → Schlaganfall (hohe Konfidenz)
    - Rechtsseitige Unterbauchschmerzen + Übelkeit + Fieber → Appendizitis (hohe Konfidenz)
    sollten mit höherer Konfidenz bewertet werden als untypische Symptomkonstellationen.
    
    Sei bei klassischen Krankheitsbildern mit typischer Symptomatik zuversichtlich in deiner Diagnose und weise eindeutig auf die wahrscheinlichste Diagnose hin.
    """
    
    # Finale Anweisungen für das strukturierte Antwortformat
    prompt += """
    Basierend auf diesen klinischen Informationen, gib bitte deine Diagnose im folgenden JSON-Format zurück:
    
    ```json
    {
      "diagnosen": {
        "Hauptdiagnose": 65,
        "Differentialdiagnose 1": 20,
        "Differentialdiagnose 2": 10,
        "Differentialdiagnose 3": 5
      },
      "konfidenz": "hoch/mittel/niedrig",
      "begründung": "Kurze klinische Begründung für die Hauptdiagnose.",
      "behandlung": "Konkrete Behandlungsempfehlung für die Hauptdiagnose.",
      "abrechnungscode": "ICD-10 Code für die Hauptdiagnose (z.B. J18.9)"
    }
    ```
    
    Bei der Konfidenz wähle "hoch", wenn die Symptomkonstellation typisch ist und wenig Zweifel an der Diagnose besteht,
    "mittel", wenn die Diagnose wahrscheinlich, aber nicht eindeutig ist, und
    "niedrig", wenn mehrere Differentialdiagnosen mit ähnlicher Wahrscheinlichkeit vorliegen.
    
    Achte darauf, dass die Wahrscheinlichkeiten in Prozent (ohne %-Zeichen) angegeben werden und in der Summe 100 ergeben.
    Die Diagnosen sollten spezifisch und präzise sein, basierend auf den vorliegenden Symptomen und klinischen Daten.
    """
    
    return prompt

def parse_llm_response(response_text):
    """
    Verbesserte Parsing-Funktion für die LLM-Antwort, die verschiedene Formate erkennt
    und strukturierte Informationen extrahiert.
    
    Args:
        response_text (str): LLM-Antworttext
        
    Returns:
        dict: Strukturiertes Dictionary mit diagnostischen Informationen
    """
    # Standardergebnis
    result = {
        "diagnosen": {},
        "behandlung": "",
        "abrechnungscode": "",
        "begründung": ""
    }
    
    try:
        # 1. Versuche, JSON-Format zu erkennen und zu parsen
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, response_text, re.DOTALL)
        
        if json_match:
            json_data = json_match.group(1).strip()
            try:
                parsed_data = json.loads(json_data)
                return parsed_data
            except json.JSONDecodeError:
                logger.warning(f"Fehler beim Parsen des JSON-Formats: {json_data}")
        
        # 2. Wenn keine JSON-Codeblöcke gefunden wurden, versuche direktes JSON-Format
        if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
            try:
                parsed_data = json.loads(response_text)
                return parsed_data
            except json.JSONDecodeError:
                logger.warning("Fehler beim Parsen des direkten JSON-Formats")
        
        # 3. Wenn kein JSON-Format gefunden wird, versuche reguläre Ausdrücke für Textformat
        # Diagnosen extrahieren
        diagnoses_pattern = r"(?:DIAGNOSEN:|Diagnosen:|Diagnose[n]?:|[1-5]\.|Mögliche Diagnosen:)(.*?)(?:(?:BEGRÜNDUNG|Begründung|BEHANDLUNG|Behandlung|ABRECHNUNGSCODE|Abrechnungscode):|\Z)"
        diagnoses_match = re.search(diagnoses_pattern, response_text, re.DOTALL)
        
        if diagnoses_match:
            diagnoses_text = diagnoses_match.group(1).strip()
            diagnosis_pattern = r"([\w\s\-äöüÄÖÜß\(\)]+)(?:[:-]\s*|:\s*|,\s*|–\s*|:\s*-\s*|\s+–\s+|\s+-\s+)(?:(\d+(?:\.\d+)?)%|(\d+(?:\.\d+)?))"
            
            for match in re.finditer(diagnosis_pattern, diagnoses_text):
                name = match.group(1).strip().rstrip(':').rstrip('-').rstrip()
                percentage = float(match.group(2) if match.group(2) else match.group(3))
                result["diagnosen"][name] = percentage
        
        # Behandlung extrahieren
        treatment_pattern = r"(?:BEHANDLUNG|Behandlung|Therapie|Treatment):\s*(.*?)(?:(?:ABRECHNUNGSCODE|Abrechnungscode|BEGRÜNDUNG|Begründung):|\Z)"
        treatment_match = re.search(treatment_pattern, response_text, re.DOTALL)
        
        if treatment_match:
            result["behandlung"] = treatment_match.group(1).strip()
        
        # Abrechnungscode extrahieren
        code_pattern = r"(?:ABRECHNUNGSCODE|Abrechnungscode|ICD-10|ICD|Code):\s*([A-Z][0-9]+\.?[0-9]*)"
        code_match = re.search(code_pattern, response_text)
        
        if code_match:
            result["abrechnungscode"] = code_match.group(1).strip()
        
        # Begründung extrahieren
        reason_pattern = r"(?:BEGRÜNDUNG|Begründung|Klinische Begründung|Rationale):\s*(.*?)(?:(?:BEHANDLUNG|Behandlung|ABRECHNUNGSCODE|Abrechnungscode):|\Z)"
        reason_match = re.search(reason_pattern, response_text, re.DOTALL)
        
        if reason_match:
            result["begründung"] = reason_match.group(1).strip()
        
        return result
    
    except Exception as e:
        logger.error(f"Fehler beim Parsen der LLM-Antwort: {str(e)}")
        return {"error": f"Fehler beim Parsen der LLM-Antwort: {str(e)}"}
    
def find_matching_diagnosis(target_diagnosis, diagnoses_dict):
    """
    Findet die beste Übereinstimmung für eine Diagnose in einem Dictionary von Diagnosen.
    Berücksichtigt partielle Übereinstimmungen und Synonym-Erkennung.
    
    Args:
        target_diagnosis (str): Zieldiagnose, die gesucht wird
        diagnoses_dict (dict): Dictionary mit Diagnosen und Wahrscheinlichkeiten
    
    Returns:
        str: Name der besten Übereinstimmung oder None, wenn keine gefunden
    """
    target = target_diagnosis.lower()
    
    # Direkte Übereinstimmung
    for diagnosis in diagnoses_dict.keys():
        if diagnosis.lower() == target:
            return diagnosis
    
    # Partielle Übereinstimmung
    best_match = None
    best_score = 0
    
    for diagnosis in diagnoses_dict.keys():
        diagnosis_lower = diagnosis.lower()
        
        # Enthält die Zeichenkette
        if target in diagnosis_lower or diagnosis_lower in target:
            score = len(set(target.split()) & set(diagnosis_lower.split())) / max(len(target.split()), len(diagnosis_lower.split()))
            score += 0.2  # Bonus für Teilstring-Übereinstimmung
            
            if score > best_score:
                best_score = score
                best_match = diagnosis
                
        # Wort-Übereinstimmung
        else:
            score = len(set(target.split()) & set(diagnosis_lower.split())) / max(len(target.split()), len(diagnosis_lower.split()))
            if score > best_score:
                best_score = score
                best_match = diagnosis
    
    # Nur zurückgeben, wenn ausreichende Übereinstimmung
    return best_match if best_score > 0.25 else None

def enhance_ml_results(ml_results, llm_results, mimic_results, symptome, vitals_string, alter, zusatz_info=""):
    """
    Verbesserte Funktion zur Kombination von ML-Ergebnissen mit LLM-Insights und 
    MIMIC-Daten für eine optimierte Diagnose.
    
    Args:
        ml_results (dict): ML-Modell Ergebnisse
        llm_results (dict): LLM-Antworten
        mimic_results (list): Ähnliche Fälle aus MIMIC-Datenbank
        symptome (str): Symptombeschreibung
        vitals_string (str): Vitalparameter
        alter (str): Alter des Patienten
        zusatz_info (str): Zusätzliche klinische Informationen
        
    Returns:
        dict: Erweiterte und optimierte Diagnoseergebnisse
    """
    # Kopie der ML-Ergebnisse erstellen
    enhanced_results = ml_results.copy()
    
    # Symptom- und Zusatzinformationen kombinieren für Kontextanalyse
    symptom_text = symptome.lower() + " " + zusatz_info.lower()
    
    # 1. SPEZIFISCHE ÜBERSCHREIBUNGEN FÜR EINDEUTIGE KLINISCHE FÄLLE
    
    # Hochspezifische Symptomkonstellationen erkennen
    if "brustschmerzen" in symptom_text and "ausstrahlung" in symptom_text and "arm" in symptom_text:
        if "Akuter Herzinfarkt" in enhanced_results["diagnosen"]:
            enhanced_results["diagnosen"]["Akuter Herzinfarkt"] *= 2.0
            enhanced_results["top_diagnose"] = "Akuter Herzinfarkt"
            
    if "fieber" in symptom_text and "nackensteifigkeit" in symptom_text and "kopfschmerzen" in symptom_text:
        if "Meningitis" in enhanced_results["diagnosen"]:
            enhanced_results["diagnosen"]["Meningitis"] *= 2.0
            enhanced_results["top_diagnose"] = "Meningitis"
    
    # Allergiefall-Erkennung und Überschreibung
    is_allergy_case = (
        ("ausschlag" in symptom_text or "quaddel" in symptom_text or "urtik" in symptom_text or "juck" in symptom_text) and
        (
            any(food in symptom_text for food in ["garnele", "nuss", "fisch", "milch", "ei", "verzehr", "lebensmittel"]) or
            "dyspnoe" in symptom_text  # Dyspnoe bei Hauterscheinungen deutet auf allergische Reaktion hin
        )
    )
    
    if is_allergy_case:
        logger.info("Allergiefall erkannt: Anwendung spezifischer klinischer Regeln")
        # Schweregrad einschätzen
        severity = "mittel"
        if "dyspnoe" in symptom_text or "atem" in symptom_text:
            severity = "schwer"  # Dyspnoe deutet auf schwerere Reaktion hin
        
        # Diagnosen nach Schweregrad festlegen
        if severity == "schwer":
            enhanced_results["diagnosen"] = {
                "Anaphylaktische Reaktion": 45.0,
                "Nahrungsmittelallergie": 35.0,
                "Urtikaria": 15.0,
                "Angioödem": 5.0
            }
            enhanced_results["top_diagnose"] = "Anaphylaktische Reaktion"
            enhanced_results["behandlung"] = "NOTFALL: Sofortige Gabe von Adrenalin, Antihistaminika, Glucocorticoide. Überwachung der Vitalparameter. Ggf. Einweisung in Notaufnahme. Nach Stabilisierung Allergenvermeidung und Patientenschulung."
            enhanced_results["abrechnungscode"] = "T78.2"
        else:
            enhanced_results["diagnosen"] = {
                "Urtikaria": 40.0,
                "Nahrungsmittelallergie": 35.0,
                "Allergische Reaktion": 20.0,
                "Angioödem": 5.0
            }
            enhanced_results["top_diagnose"] = "Urtikaria"
            enhanced_results["behandlung"] = "Antihistaminika (z.B. Cetirizin), ggf. kurzfristig Glucocorticoide. Identifikation und Vermeidung des auslösenden Allergens. Bei wiederholten Reaktionen allergologische Abklärung."
            enhanced_results["abrechnungscode"] = "L50.0"
        
        # Vorzeitige Rückgabe bei diesem spezifischen Fall
        return enhanced_results
    
    # Weitere spezifische klinische Szenarien
    # Harnwegsinfektionsfall
    is_uti_case = (
        any(term in symptom_text for term in ["dysurie", "brennen beim wasserlassen", "pollakisurie"]) and
        not "brustschmerz" in symptom_text and not "atemnot" in symptom_text
    )
    
    if is_uti_case:
        logger.info("Harnwegsinfektionsfall erkannt: Spezifische Diagnoseregeln werden angewendet")
        # Prüfen auf Komplikationsfaktoren
        has_fever = "fieber" in symptom_text or (vitals_string and any(x in vitals_string for x in ["T:38", "T:39", "T:40"]))
        has_flank_pain = any(term in symptom_text for term in ["flanke", "rücken", "niere", "seitenschmerz"])
        
        # Diagnosen nach Faktoren festlegen
        if has_fever and has_flank_pain:
            enhanced_results["diagnosen"] = {
                "Pyelonephritis": 60.0,
                "Harnwegsinfektion": 25.0,
                "Zystitis": 15.0
            }
            enhanced_results["top_diagnose"] = "Pyelonephritis"
            enhanced_results["behandlung"] = "Antibiotika (z.B. Ciprofloxacin, Cefuroxim) für 7-14 Tage, reichlich Flüssigkeitszufuhr, Analgetika bei Bedarf. Bei schweren Fällen stationäre Aufnahme erwägen."
            enhanced_results["abrechnungscode"] = "N10"
        else:
            enhanced_results["diagnosen"] = {
                "Harnwegsinfektion": 60.0,
                "Zystitis": 35.0,
                "Urethritis": 5.0
            }
            enhanced_results["top_diagnose"] = "Harnwegsinfektion"
            enhanced_results["behandlung"] = "Antibiotika (z.B. Nitrofurantoin, Fosfomycin), reichlich Flüssigkeitszufuhr, ggf. Schmerzmittel. Bei häufiger Wiederkehr erweiterte Diagnostik."
            enhanced_results["abrechnungscode"] = "N30.0"
        
        return enhanced_results
        
    # Schlaganfall-Erkennung mit höherer Konfidenz
    is_stroke_case = (
        ("lähmung" in symptom_text or "halbseitige schwäche" in symptom_text) and 
        ("sprachstörung" in symptom_text or "aphasie" in symptom_text)
    )
    
    if is_stroke_case:
        logger.info("Schlaganfallsymptomatik erkannt: Höhere Konfidenz")
        if "Schlaganfall" in enhanced_results["diagnosen"]:
            # Deutlich höhere Gewichtung bei klassischer Symptomatik
            enhanced_results["diagnosen"]["Schlaganfall"] = max(85.0, enhanced_results["diagnosen"]["Schlaganfall"] * 3.0)
            enhanced_results["top_diagnose"] = "Schlaganfall"
            enhanced_results["behandlung"] = "NOTFALL: Sofortiger Transport ins Krankenhaus mit Stroke Unit. Zeit ist Hirn! Bildgebung (CT/MRT), evtl. Thrombolyse oder mechanische Thrombektomie."
            enhanced_results["abrechnungscode"] = "I63.9"
            # Reduzierung anderer Diagnosen
            for diagnose in enhanced_results["diagnosen"]:
                if diagnose != "Schlaganfall":
                    enhanced_results["diagnosen"][diagnose] *= 0.3
            
            return enhanced_results
    
    # 2. ALLGEMEINE INTEGRATION VON LLM UND ML ERGEBNISSEN
    
    # LLM-Diagnosen integrieren
    if "diagnosen" in llm_results and isinstance(llm_results["diagnosen"], dict):
        llm_diagnoses = llm_results["diagnosen"]
        
        # Für jede LLM-Diagnose
        for diagnosis, probability in llm_diagnoses.items():
            if isinstance(probability, (int, float)) and probability > 0:
                # Finde übereinstimmende oder ähnliche Diagnose in ML-Ergebnissen
                ml_match = find_matching_diagnosis(diagnosis, enhanced_results["diagnosen"])
                
                if ml_match:
                    # Diagnose existiert bereits - gewichtete Kombination
                    orig_prob = enhanced_results["diagnosen"][ml_match]
                    # Stärkerer Einfluss der LLM-Diagnose bei hohen Wahrscheinlichkeiten
                    llm_weight = 0.4 + (0.3 * (probability / 100))  # 0.4 bis 0.7 je nach LLM-Konfidenz
                    enhanced_results["diagnosen"][ml_match] = (orig_prob * (1 - llm_weight)) + (probability * llm_weight)
                else:
                    # Neue Diagnose - mit moderater Wahrscheinlichkeit hinzufügen
                    adjusted_prob = min(probability * 0.7, 40.0)  # Max 40% für neue Diagnosen vom LLM
                    enhanced_results["diagnosen"][diagnosis] = adjusted_prob

    # 3. MIMIC-BASIERTE ANPASSUNGEN
    
    if mimic_results and len(mimic_results) > 0:
        logger.info(f"Integriere Informationen aus {len(mimic_results)} ähnlichen MIMIC-Fällen")
        
        # Diagnosestatistik aus MIMIC-Fällen
        mimic_diagnoses = {}
        for case in mimic_results:
            for diagnosis in case.get("diagnoses", []):
                if diagnosis not in mimic_diagnoses:
                    mimic_diagnoses[diagnosis] = 0
                # Gewichtung nach Ähnlichkeitsscore
                mimic_diagnoses[diagnosis] += case.get("similarity_score", 1.0)
        
        # Top-3 MIMIC-Diagnosen
        top_mimic_diagnoses = sorted(mimic_diagnoses.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Integration der MIMIC-Diagnosen mit höherer Gewichtung für übereinstimmende Diagnosen
        for diagnosis, score in top_mimic_diagnoses:
            ml_match = find_matching_diagnosis(diagnosis, enhanced_results["diagnosen"])
            
            if ml_match:
                # Verstärke existierende Diagnose - höhere Verstärkung als zuvor
                if score > 0.5:  # Wenn mehrere MIMIC-Fälle dieselbe Diagnose haben
                    boost_factor = 1.0 + (score * 2.0)  # Stärkere Verstärkung
                else:
                    boost_factor = 1.0 + (score * 1.5)
                
                enhanced_results["diagnosen"][ml_match] *= boost_factor
            elif score > 1.5:  # Nur relevante MIMIC-Diagnosen hinzufügen
                # Neue Diagnose mit konservativer Wahrscheinlichkeit
                enhanced_results["diagnosen"][diagnosis] = min(score * 10, 25.0)  # Max 25% für neue MIMIC-Diagnosen
    
    # 4. KLINISCHE REGELN UND KONTEXTUELLE ANPASSUNGEN
    
    # Altersbasierte Anpassungen
    if alter in ["Säugling", "Kleinkind", "Kind"]:
        # Deutliche Reduzierung unwahrscheinlicher Diagnosen bei Kindern
        adult_diagnoses = ["Myokardinfarkt", "Akuter Herzinfarkt", "Angina pectoris"]
        for diag in adult_diagnoses:
            if diag in enhanced_results["diagnosen"]:
                enhanced_results["diagnosen"][diag] *= 0.05  # 95% Reduktion
    
    # Konfidenzgewichtung für bestimmte kritische Symptomkombinationen
    critical_patterns = {
        ("brustschmerzen", "schwitzen", "übelkeit"): ["Akuter Herzinfarkt", 3.0],
        ("fieber", "husten", "atemnot"): ["Pneumonie", 2.5],
        ("kopfschmerzen", "fieber", "erbrechen"): ["Meningitis", 2.7],
        ("bauchschmerzen", "erbrechen", "appetitlosigkeit"): ["Appendizitis", 2.2],
        ("dyspnoe", "husten", "orthopnoe"): ["Herzinsuffizienz", 2.3]
    }
    
    for pattern, (target_diagnosis, factor) in critical_patterns.items():
        # Prüfe, ob alle Symptome im pattern vorhanden sind
        if all(symptom in symptom_text for symptom in pattern):
            if target_diagnosis in enhanced_results["diagnosen"]:
                enhanced_results["diagnosen"][target_diagnosis] *= factor
                # Bei sehr hoher Konfidenz, setze auch top_diagnose
                if enhanced_results["diagnosen"][target_diagnosis] > 50:
                    enhanced_results["top_diagnose"] = target_diagnosis
    
    # 5. WAHRSCHEINLICHKEITEN NORMALISIEREN
    
    # Normalisierung auf 100%
    total_probability = sum(enhanced_results["diagnosen"].values())
    if total_probability > 0:
        for diagnosis in enhanced_results["diagnosen"]:
            enhanced_results["diagnosen"][diagnosis] = round((enhanced_results["diagnosen"][diagnosis] / total_probability) * 100, 1)
    
    # Entferne unwahrscheinliche Diagnosen
    enhanced_results["diagnosen"] = {k: v for k, v in enhanced_results["diagnosen"].items() if v >= 2.0}
    
    # Sortiere Diagnosen nach Wahrscheinlichkeit
    enhanced_results["diagnosen"] = dict(sorted(enhanced_results["diagnosen"].items(), key=lambda x: x[1], reverse=True))
    
    # 6. BESTE DIAGNOSE UND BEHANDLUNG FESTLEGEN
    
    # Top-Diagnose setzen mit höherer Konfidenz wenn eine Diagnose deutlich führt
    if enhanced_results["diagnosen"]:
        sorted_diagnoses = sorted(enhanced_results["diagnosen"].items(), key=lambda x: x[1], reverse=True)
        top_diagnosis = sorted_diagnoses[0][0]
        top_probability = sorted_diagnoses[0][1]
        
        # Wenn eine Diagnose mit über 40% Wahrscheinlichkeit führt und mehr als 
        # 15% vor der nächsten liegt, setze sie mit höherer Konfidenz
        if top_probability > 40 and (len(sorted_diagnoses) < 2 or top_probability - sorted_diagnoses[1][1] > 15):
            enhanced_results["top_diagnose"] = top_diagnosis
            enhanced_results["diagnose_konfidenz"] = "hoch"  # Neue Eigenschaft für Konfidenzwert
        else:
            enhanced_results["top_diagnose"] = top_diagnosis
            enhanced_results["diagnose_konfidenz"] = "mittel"
        
        # Behandlung aus LLM übernehmen, falls vorhanden und relevant
        if "behandlung" in llm_results and llm_results["behandlung"]:
            enhanced_results["behandlung"] = llm_results["behandlung"]
            
        # Abrechnungscode aus LLM übernehmen, falls vorhanden
        if "abrechnungscode" in llm_results and llm_results["abrechnungscode"]:
            enhanced_results["abrechnungscode"] = llm_results["abrechnungscode"]
    
    return enhanced_results

def diagnose(symptome, vitals_string="", alter="Erwachsener", zusatz_info=""):
    """
    Hauptfunktion für die verbesserte hybride Diagnose (ML + LLM + MIMIC)
    
    Args:
        symptome (str): Kommaseparierte Liste von Symptomen
        vitals_string (str): String mit Vitalparametern im Format "HR:80,BP:120/80,T:36.8,SpO2:98"
        alter (str): Altersgruppe oder spezifisches Alter
        zusatz_info (str): Zusätzliche klinische Informationen
        
    Returns:
        dict: Diagnoseergebnisse und weitere Informationen
    """
    start_time = time.time()
    logger.info(f"Starte Diagnose für Symptome: {symptome}")
    
    # 1. ML-Modell aufrufen
    ml_results = patient_verarbeiten(symptome, vitals_string, alter, zusatz_info)
    
    # 2. Fehlerprüfung des ML-Modells
    if "fehler" in ml_results:
        logger.warning(f"ML-Modell Fehler: {ml_results['fehler']}")
        return {"result": ml_results, "source": "ml_model"}
    
    # 3. Ähnliche Fälle aus MIMIC-Datenbank abrufen, wenn möglich
    mimic_results = []
    try:
        # Laborwerte aus zusatz_info extrahieren, wenn vorhanden
        laborwerte = {}
        if "laborwerte:" in zusatz_info.lower():
            match = re.search(r"laborwerte:\s*(.*?)(?:\n|$)", zusatz_info, re.IGNORECASE)
            if match:
                for lab_item in match.group(1).split(","):
                    if ":" in lab_item:
                        key, value = lab_item.split(":", 1)
                        laborwerte[key.strip().lower()] = value.strip()
        
        # Geschlecht extrahieren
        geschlecht = None
        if "geschlecht:" in zusatz_info.lower() or "weiblich" in zusatz_info.lower() or "männlich" in zusatz_info.lower():
            if "weiblich" in zusatz_info.lower():
                geschlecht = "weiblich"
            elif "männlich" in zusatz_info.lower():
                geschlecht = "männlich"
            else:
                match = re.search(r"geschlecht:\s*(.*?)(?:\n|$)", zusatz_info, re.IGNORECASE)
                if match:
                    geschlecht = match.group(1).strip()
        
        # MIMIC-Daten abrufen
        mimic_results = mimic.get_similar_cases(symptome, vitals_string, alter, geschlecht, laborwerte, max_cases=5)
        logger.info(f"MIMIC: {len(mimic_results)} ähnliche Fälle gefunden")
    except Exception as e:
        logger.error(f"Fehler bei MIMIC-Integration: {str(e)}")
        mimic_results = []
    
    # 4. LLM-Unterstützung
    llm_results = {}
    provider, _ = get_available_llm_provider()
    
    if provider:
        try:
            # Prompt für LLM erstellen
            prompt = get_medical_prompt(symptome, vitals_string, alter, zusatz_info)
            
            # Komplexität des Falls bestimmen (für Modellauswahl)
            is_complex_case = len(symptome.split(",")) > 4 or "laborwerte:" in zusatz_info.lower()
            model_type = "advanced" if is_complex_case else "default"
            
            # LLM-Anfrage
            llm_response = call_llm_service(prompt, provider, model_type)
            
            # Fehlerprüfung
            if isinstance(llm_response, dict) and "error" in llm_response:
                logger.warning(f"LLM-Fehler: {llm_response['error']}")
                llm_results = {}
            else:
                # LLM-Antwort parsen
                llm_results = parse_llm_response(llm_response)
                
                # Fehlerprüfung beim Parsing
                if "error" in llm_results:
                    logger.warning(f"LLM-Parsing-Fehler: {llm_results['error']}")
                    llm_results = {}
                else:
                    logger.info("LLM-Integration erfolgreich")
        except Exception as e:
            logger.error(f"Fehler bei LLM-Integration: {str(e)}")
            llm_results = {}
    else:
        logger.info("Diagnose nur mit ML-Modell (kein LLM-Provider verfügbar)")
    
    # 5. Ergebnisse kombinieren für eine optimierte Diagnose
    try:
        enhanced_results = enhance_ml_results(
            ml_results, 
            llm_results, 
            mimic_results,
            symptome, 
            vitals_string, 
            alter, 
            zusatz_info
        )
        
        # Bearbeitungszeit protokollieren
        processing_time = time.time() - start_time
        logger.info(f"Hybride Diagnose abgeschlossen in {processing_time:.2f} Sekunden")
        
        # Quelle der Diagnose angeben
        source = "hybrid"
        if not provider and not mimic_results:
            source = "ml_model"
        elif not provider:
            source = "ml_mimic"
        elif not mimic_results:
            source = "ml_llm"
        
        return {"result": enhanced_results, "source": source}
        
    except Exception as e:
        logger.error(f"Fehler bei der Ergebnisoptimierung: {str(e)}")
        # Fallback auf reine ML-Ergebnisse bei Problemen
        return {"result": ml_results, "source": "ml_model"}

if __name__ == "__main__":
    # Einfacher Test mit verschiedenen Symptomkombinationen
    test_fälle = [
        {
            "symptome": "Fieber >38°C, Husten, Kurzatmigkeit",
            "vitals": "HR:95,SpO2:92,T:38.5",
            "alter": "Erwachsener"
        },
        {
            "symptome": "Brustschmerzen, Ausstrahlung in linken Arm, Schweißausbrüche",
            "vitals": "HR:110,BP:145/95,T:36.7,SpO2:94",
            "alter": "Erwachsener"
        },
        {
            "symptome": "Bellender Husten, Heiserkeit, Fieber >38°C",
            "vitals": "HR:120,T:38.7,SpO2:94",
            "alter": "Kind"
        }
    ]
    
    for i, test_fall in enumerate(test_fälle):
        print(f"\n--- Testfall {i+1}: {test_fall['symptome']} ---")
        result = diagnose(
            test_fall["symptome"], 
            test_fall.get("vitals", ""), 
            test_fall.get("alter", "Erwachsener")
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

