from flask import Flask, request, render_template_string, jsonify, session
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from core import SYMPTOM_DB  # Symptom-Datenbank importieren
from hybrid_model import diagnose, call_llm_service
from mimic_integration import MIMICIntegration

# Umgebungsvariablen laden
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))  # Für Session-Management

# Konfiguration
CHAT_HISTORY_LENGTH = 10  # Maximale Anzahl der Chat-Nachrichten im Speicher

# MIMIC-Integration initialisieren
mimic = MIMICIntegration()

# Symptome für Autovervollständigung vorbereiten
def get_symptom_suggestions():
    symptome = []
    for s in SYMPTOM_DB.values():
        symptome.append(s["name"])
        symptome.extend(s.get("synonyme", []))
    return list(set(symptome))  # Duplikate entfernen

SYMPTOME = get_symptom_suggestions()

@app.route("/", methods=["GET", "POST"])
def startseite():
    return index()

@app.route("/index", methods=["GET"])
def index():
    # Chat-Historie aus der Session laden oder initialisieren
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    return render_template_string(HTML_TEMPLATE, 
                                 symptome=SYMPTOME, 
                                 request=request,
                                 ergebnis=None,
                                 patient=None,
                                 mimic_cases=None,
                                 chat_messages=session['chat_history'])
@app.route("/diagnose", methods=["POST"])
def diagnose_route():
    try:
        # Daten aus dem Formular extrahieren
        symptome_eingabe = request.form.get("symptome", "")
        alter_dropdown = request.form.get("alter", "Erwachsener")
        alter_jahre = request.form.get("alter_jahre", "")
        geschlecht = request.form.get("geschlecht", "")
        zusatz_info = request.form.get("zusatz_info", "")
        vorerkrankungen = request.form.get("vorerkrankungen", "")
        vorherige_operationen = request.form.get("vorherige_operationen", "")
        
        # Überprüfen, ob die Symptomeingabe nicht leer ist
        if not symptome_eingabe.strip():
            return render_template_string(HTML_TEMPLATE, 
                                        symptome=SYMPTOME, 
                                        request=request,
                                        ergebnis={"result": {"fehler": "Bitte geben Sie mindestens ein Symptom ein."}, "source": "validation"},
                                        patient=None,
                                        mimic_cases=None,
                                        chat_messages=session.get('chat_history', []))
        
        # Klinische Untersuchung
        cor = request.form.get("cor", "")
        pulmo = request.form.get("pulmo", "")
        abdomen = request.form.get("abdomen", "")
        hno = request.form.get("hno", "")
        
        # Laborwerte
        leukozyten = request.form.get("leukozyten", "")
        erythrozyten = request.form.get("erythrozyten", "")
        haemoglobin = request.form.get("haemoglobin", "")
        crp = request.form.get("crp", "")
        weitere_laborwerte = request.form.get("weitere_laborwerte", "")
        
        # Vitalparameter sammeln
        vitals_dict = {}
        if request.form.get("hr"):
            vitals_dict["HR"] = request.form.get("hr")
        if request.form.get("bp"):
            vitals_dict["BP"] = request.form.get("bp")
        if request.form.get("temp"):
            vitals_dict["T"] = request.form.get("temp")
        if request.form.get("spo2"):
            vitals_dict["SpO2"] = request.form.get("spo2")
        
        # Vitalparameter in String umwandeln (HR:80,BP:120/80,T:36.8,SpO2:98)
        vitals_string = ','.join([f"{k}:{v}" for k, v in vitals_dict.items()])
        
        # Erstelle ein erweitertes Zusatzinfo-Feld, das alle relevanten Informationen enthält
        erweitertes_zusatz_info = zusatz_info
        
        # Füge Patienteninformationen hinzu
        if vorerkrankungen:
            erweitertes_zusatz_info += f"\nVorerkrankungen: {vorerkrankungen}"
        if vorherige_operationen:
            erweitertes_zusatz_info += f"\nVorherige Operationen: {vorherige_operationen}"
        
        # Füge Untersuchungsinformationen hinzu
        untersuchungsmerkmale = []
        if cor:
            untersuchungsmerkmale.append(f"Cor: {cor}")
        if pulmo:
            untersuchungsmerkmale.append(f"Pulmo: {pulmo}")
        if abdomen:
            untersuchungsmerkmale.append(f"Abdomen: {abdomen}")
        if hno:
            untersuchungsmerkmale.append(f"HNO: {hno}")
        
        if untersuchungsmerkmale:
            erweitertes_zusatz_info += f"\nUntersuchungsergebnisse: {', '.join(untersuchungsmerkmale)}"
        
        # Füge Laborwerte hinzu
        laborwerte = []
        if leukozyten:
            laborwerte.append(f"Leukozyten: {leukozyten}/nl")
        if erythrozyten:
            laborwerte.append(f"Erythrozyten: {erythrozyten}/pl")
        if haemoglobin:
            laborwerte.append(f"Hämoglobin: {haemoglobin} g/dl")
        if crp:
            laborwerte.append(f"CRP: {crp} mg/l")
        if weitere_laborwerte:
            laborwerte.append(weitere_laborwerte)
        
        if laborwerte:
            erweitertes_zusatz_info += f"\nLaborwerte: {', '.join(laborwerte)}"
        
        # Diagnosetimer starten
        start_time = time.time()
        app.logger.info(f"Starte Diagnose für Symptome: {symptome_eingabe}")
        
        # Diagnose durchführen mit verbesserter Implementierung
        ergebnis = diagnose(symptome_eingabe, vitals_string, alter_dropdown, erweitertes_zusatz_info)
        
        # Diagnosezeit protokollieren
        process_time = time.time() - start_time
        app.logger.info(f"Diagnose abgeschlossen in {process_time:.2f} Sekunden")
        
        # MIMIC-ähnliche Fälle für zusätzliches Kontext abrufen
        mimic_cases = []
        try:
            labor_dict = {}
            if leukozyten: labor_dict["leukozyten"] = leukozyten
            if erythrozyten: labor_dict["erythrozyten"] = erythrozyten
            if haemoglobin: labor_dict["haemoglobin"] = haemoglobin
            if crp: labor_dict["crp"] = crp
            
            mimic_cases = mimic.get_similar_cases(
                symptome_eingabe, 
                vitals_string, 
                alter_dropdown, 
                geschlecht, 
                labor_dict, 
                max_cases=3
            )
        except Exception as e:
            # Fehler bei MIMIC-Integration loggen, aber nicht die App stören
            app.logger.error(f"Fehler beim Abruf von MIMIC-Fällen: {str(e)}")
            mimic_cases = []  # Leere Liste als Fallback
        
        # Kennzeichnen wichtiger Diagnosen mit hoher Konfidenz
        if "result" in ergebnis and "diagnosen" in ergebnis["result"]:
            # Prüfen, ob eine diagnose_konfidenz bereits existiert
            konfidenz = ergebnis["result"].get("diagnose_konfidenz", "")
            
            # Prüfen auf kritische Diagnosen mit hoher Wahrscheinlichkeit
            critical_diagnoses = ["Akuter Herzinfarkt", "Meningitis", "Schlaganfall", 
                                "Lungenembolie", "Anaphylaktische Reaktion", "Subarachnoidalblutung"]
            
            top_diagnose = ergebnis["result"].get("top_diagnose", "")
            if top_diagnose in critical_diagnoses and ergebnis["result"]["diagnosen"].get(top_diagnose, 0) > 50:
                # Markieren als kritisch mit höchster Priorität
                ergebnis["result"]["kritisch"] = True
                ergebnis["result"]["prioritaet"] = "sofort"
            elif konfidenz == "hoch" and ergebnis["result"]["diagnosen"].get(top_diagnose, 0) > 65:
                # Markieren als hohe Konfidenz
                ergebnis["result"]["prioritaet"] = "hoch"
        
        # Patient vollständig als Dictionary setzen für die Anzeige in der Ergebnisansicht
        patient = {
            "alter_dropdown": alter_dropdown,
            "alter_jahre": alter_jahre,
            "geschlecht": geschlecht,
            "vorerkrankungen": vorerkrankungen,
            "vorherige_operationen": vorherige_operationen,
            "untersuchung": {
                "cor": cor,
                "pulmo": pulmo,
                "abdomen": abdomen,
                "hno": hno
            },
            "labor": {
                "leukozyten": leukozyten,
                "erythrozyten": erythrozyten,
                "haemoglobin": haemoglobin,
                "crp": crp,
                "weitere": weitere_laborwerte
            },
            "vitals": vitals_dict
        }
        
        # Chat-Historie aus der Session laden oder initialisieren
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        return render_template_string(HTML_TEMPLATE, 
                                    symptome=SYMPTOME, 
                                    request=request,
                                    ergebnis=ergebnis,
                                    patient=patient,
                                    mimic_cases=mimic_cases,
                                    chat_messages=session['chat_history'])
    
    except Exception as e:
        # Umfassende Fehlerbehandlung für die gesamte Route
        app.logger.error(f"Allgemeiner Fehler in diagnose_route: {str(e)}", exc_info=True)
        return render_template_string(HTML_TEMPLATE,
                                    symptome=SYMPTOME,
                                    request=request,
                                    ergebnis={"result": {"fehler": f"Ein Fehler ist aufgetreten: {str(e)}"},
                                             "source": "error"},
                                    patient=None,
                                    mimic_cases=None,
                                    chat_messages=session.get('chat_history', []))
@app.route("/chat", methods=["POST"])
def chat():
    message = request.form.get("message", "")
    
    if not message:
        return jsonify({"error": "Keine Nachricht erhalten"})
    
    # Chat-Historie aus der Session laden oder initialisieren
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    # Benutzer-Nachricht zur Historie hinzufügen
    session['chat_history'].append({
        "sender": "user",
        "text": message,
        "time": datetime.now().strftime("%H:%M")
    })
    
    # Prompt für das LLM erstellen
    prompt = f"""
    Als medizinischer Assistent für Ärzte beantworte bitte folgende Frage:
    
    {message}
    
    Gib eine präzise, evidenzbasierte Antwort in deutscher Sprache.
    """
    
    # LLM aufrufen und Antwort verarbeiten
    try:
        llm_response = call_llm_service(prompt)
        
        if isinstance(llm_response, dict) and "error" in llm_response:
            response_text = "Entschuldigung, ich konnte Ihre Anfrage nicht verarbeiten. Bitte versuchen Sie es später erneut oder formulieren Sie Ihre Frage anders."
        else:
            response_text = llm_response
    except Exception as e:
        response_text = f"Es ist ein Fehler aufgetreten: {str(e)}"
    
    # Assistant-Antwort zur Historie hinzufügen
    session['chat_history'].append({
        "sender": "assistant",
        "text": response_text,
        "time": datetime.now().strftime("%H:%M")
    })
    
    # Historie auf maximale Länge begrenzen
    if len(session['chat_history']) > CHAT_HISTORY_LENGTH * 2:  # *2 weil Paare von Nachrichten
        session['chat_history'] = session['chat_history'][-CHAT_HISTORY_LENGTH * 2:]
    
    session.modified = True
    
    return render_template_string(HTML_TEMPLATE, 
                                 symptome=SYMPTOME, 
                                 request=request,
                                 ergebnis=None,
                                 patient=None,
                                 mimic_cases=None,
                                 chat_messages=session['chat_history'])

# API-Endpunkt für die Diagnose
@app.route("/api/diagnose", methods=["POST"])
def api_diagnose():
    data = request.json
    symptome_eingabe = data.get("symptome", "")
    vitals_eingabe = data.get("vitals", "")
    alter = data.get("alter", "Erwachsener")
    zusatz_info = data.get("zusatz_info", "")
    
    ergebnis = diagnose(symptome_eingabe, vitals_eingabe, alter, zusatz_info)
    return jsonify(ergebnis)

# API-Endpunkt für den Chat
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    message = data.get("message", "")
    
    if not message:
        return jsonify({"error": "Keine Nachricht erhalten"})
    
    # Prompt für das LLM erstellen
    prompt = f"""
    Als medizinischer Assistent für Ärzte beantworte bitte folgende Frage:
    
    {message}
    
    Gib eine präzise, evidenzbasierte Antwort in deutscher Sprache.
    """
    
    try:
        llm_response = call_llm_service(prompt)
        
        if isinstance(llm_response, dict) and "error" in llm_response:
            return jsonify({
                "error": llm_response["error"],
                "message": "Entschuldigung, ich konnte Ihre Anfrage nicht verarbeiten."
            })
        else:
            return jsonify({
                "response": llm_response,
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Es ist ein Fehler aufgetreten."
        })
# HTML Template-Variable (Beginn)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PraxisPro - KI-gestützte medizinische Diagnose</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 15px;
        }
        .logo {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .logo span {
            color: #3498db;
        }
        .developer {
            font-size: 14px;
            color: #7f8c8d;
            text-align: right;
        }
        h1 {
            color: #2c3e50;
            font-size: 28px;
            margin-bottom: 20px;
        }
        h3 {
            color: #2c3e50;
            margin-top: 25px;
            margin-bottom: 10px;
        }
        .form-group {
            margin-bottom: 15px;
            position: relative;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="text"], input[type="number"], textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
            box-sizing: border-box;
        }
        textarea {
            min-height: 80px;
            resize: vertical;
        }
        select {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
            box-sizing: border-box;
            background-color: white;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #2980b9;
        }

        #suggestions {
            position: absolute;
            width: 100%;
            max-height: 200px;
            overflow-y: auto;
            background: white;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 4px 4px;
            z-index: 10;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        #suggestions div {
            padding: 10px;
            cursor: pointer;
        }
        #suggestions div:hover {
            background-color: #f5f7fa;
        }
        .results-container {
            margin-top: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .probability-bar {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .bar-label {
            width: 200px;
            font-weight: bold;
        }
        .bar-container {
            flex-grow: 1;
            background-color: #ecf0f1;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
        }
        .bar {
            height: 100%;
            background-color: #3498db;
        }
        .bar-value {
            margin-left: 10px;
            font-weight: bold;
            width: 50px;
        }
        .result-item {
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }
        .result-item:last-child {
            border-bottom: none;
        }
        .error {
            color: #e74c3c;
            font-weight: bold;
        }
        .info-box {
            padding: 15px;
            background-color: #e8f4fc;
            border-radius: 4px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
        }
        .warning {
            background-color: #fff4e6;
            border-left: 4px solid #e67e22;
        }
        .critical {
            background-color: #ffebeb;
            border-left: 4px solid #e74c3c;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            background-color: #95a5a6;
            color: white;
            border-radius: 4px;
            font-size: 14px;
            margin-right: 5px;
            margin-bottom: 5px;
        }
        .vital-card {
            background-color: #f0f8ff;
            border-radius: 4px;
            padding: 12px;
            margin-bottom: 10px;
            display: inline-block;
            margin-right: 10px;
            min-width: 100px;
            text-align: center;
        }
        .vital-value {
            font-size: 20px;
            font-weight: bold;
            margin: 5px 0;
        }
        .vital-name {
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }
        .vital-group {
            margin-top: 10px;
        }
        .form-row {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }
        .form-col {
            flex: 1;
        }
        .vital-warning {
            color: #e67e22;
            font-weight: bold;
        }
        .section-title {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        .section-icon {
            background-color: #3498db;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        .toggle-section {
            cursor: pointer;
            padding: 10px;
            background-color: #f0f8ff;
            border-radius: 4px;
            margin-bottom: 10px;
            user-select: none;
        }
        .toggle-section:hover {
            background-color: #e1f0fa;
        }
        .top-diagnosis {
            background-color: #e1f5e1;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            border-left: 4px solid #27ae60;
        }
        .pediatric-mode {
            background-color: #e8f4fc;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
            border-left: 4px solid #f39c12;
            font-weight: bold;
        }
        .chat-container {
            margin-top: 30px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 500px;
        }
        .chat-messages {
            flex-grow: 1;
            overflow-y: auto;
            padding: 15px;
            background-color: #f9f9f9;
        }
        .chat-input {
            display: flex;
            border-top: 1px solid #ddd;
            padding: 10px;
            background-color: white;
        }
        .chat-input input {
            flex-grow: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-right: 10px;
        }
        .message {
            margin-bottom: 15px;
            max-width: 80%;
        }
        .user-message {
            margin-left: auto;
            background-color: #dcf8c6;
            padding: 10px 15px;
            border-radius: 18px 18px 0 18px;
        }
        .assistant-message {
            margin-right: auto;
            background-color: white;
            padding: 10px 15px;
            border-radius: 18px 18px 18px 0;
            box-shadow: 0 1px 1px rgba(0,0,0,0.1);
        }
        .message-time {
            font-size: 12px;
            color: #999;
            text-align: right;
            margin-top: 2px;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            opacity: 0.6;
            border-bottom: 2px solid transparent;
            transition: all 0.3s;
        }
        .tab:hover {
            opacity: 1;
            background-color: #f5f7fa;
        }
        .tab.active {
            opacity: 1;
            border-bottom: 2px solid #3498db;
            font-weight: bold;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .source-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-left: 10px;
        }
        .source-ml {
            background-color: #e1f5e1;
            color: #27ae60;
        }
        .source-llm {
            background-color: #e8f4fc;
            color: #3498db;
        }
        .source-hybrid {
            background-color: #f0f4c3;
            color: #7cb342;
        }
        .patient-summary {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f5f7fa;
            border-radius: 4px;
            border-left: 4px solid #7f8c8d;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        .info-item {
            padding: 8px;
            background-color: #fff;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .info-full {
            grid-column: 1 / span 2;
        }
        .mimic-case {
            background-color: #f8faff;
            border: 1px solid #d7e3fc;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 10px;
        }
        .mimic-case p {
            margin: 5px 0;
        }
        .similar-cases-title {
            color: #2c3e50;
            font-size: 16px;
            margin-top: 20px;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">Praxis<span>Pro</span> <small style="font-size: 14px; opacity: 0.7;">Hybrid AI</small></div>
            <div class="developer">Entwickelt von Dr. med. univ. Patrick Jorge, BA<br><small>© PraxisPro™</small></div>
        </header>
        
        <div class="tabs">
            <div class="tab active" data-tab="diagnose">Diagnose</div>
            <div class="tab" data-tab="chat">Medizinischer Chat</div>
        </div>
        
        <div id="diagnose-tab" class="tab-content active">
            <h1>KI-gestützte medizinische Diagnose</h1>
            
            <div class="info-box">
                <p>PraxisPro unterstützt Ärzte bei der Diagnose anhand von Symptomen und Vitalparametern. Das System kombiniert ein lokales ML-Modell mit einem KI-Sprachmodell für optimale Ergebnisse.</p>
                <p>Bitte geben Sie die Symptome und optional die Vitalparameter des Patienten ein.</p>
            </div>
            
            <form method="POST" action="/diagnose">
                <div class="form-row">
                    <div class="form-col">
                        <div class="form-group">
                            <label for="alter">Alter des Patienten:</label>
                            <select id="alter" name="alter">
                                <option value="Erwachsener" {{ 'selected' if request.form.get('alter') == 'Erwachsener' else '' }}>Erwachsener (>18 Jahre)</option>
                                <option value="Jugendlicher" {{ 'selected' if request.form.get('alter') == 'Jugendlicher' else '' }}>Jugendlicher (14-18 Jahre)</option>
                                <option value="Kind" {{ 'selected' if request.form.get('alter') == 'Kind' else '' }}>Kind (6-14 Jahre)</option>
                                <option value="Kleinkind" {{ 'selected' if request.form.get('alter') == 'Kleinkind' else '' }}>Kleinkind (1-5 Jahre)</option>
                                <option value="Säugling" {{ 'selected' if request.form.get('alter') == 'Säugling' else '' }}>Säugling (0-1 Jahr)</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-col">
                        <div class="form-group">
                            <label for="symptome">Symptome eingeben:</label>
                            <input type="text" id="symptome" name="symptome" placeholder="z.B. Husten, Fieber, Brustschmerzen" autocomplete="off" value="{{ request.form.get('symptome', '') }}">
                            <div id="suggestions"></div>
                        </div>
                    </div>
                </div>
                
                <div class="toggle-section" id="patientInfoToggle">
                    + Patienteninformation (optional)
                </div>
                
                <div id="patientInfoSection" style="display: none;">
                    <div class="form-row">
                        <div class="form-col">
                            <div class="form-group">
                                <label for="geschlecht">Geschlecht:</label>
                                <select id="geschlecht" name="geschlecht">
                                    <option value="männlich" {{ 'selected' if request.form.get('geschlecht') == 'männlich' else '' }}>Männlich</option>
                                    <option value="weiblich" {{ 'selected' if request.form.get('geschlecht') == 'weiblich' else '' }}>Weiblich</option>
                                    <option value="divers" {{ 'selected' if request.form.get('geschlecht') == 'divers' else '' }}>Divers</option>
                                </select>
                            </div>
                        </div>
                        <div class="form-col">
                            <div class="form-group">
                                <label for="alter_jahre">Alter (Jahre):</label>
                                <input type="number" id="alter_jahre" name="alter_jahre" placeholder="z.B. 45" value="{{ request.form.get('alter_jahre', '') }}">
                            </div>
                        </div>
                    </div>

                    <div class="form-group">
                        <label for="vorerkrankungen">Vorerkrankungen:</label>
                        <textarea id="vorerkrankungen" name="vorerkrankungen" placeholder="z.B. Diabetes mellitus Typ 2, Hypertonie, KHK...">{{ request.form.get('vorerkrankungen', '') }}</textarea>
                    </div>

                    <div class="form-group">
                        <label for="vorherige_operationen">Vorherige Operationen:</label>
                        <textarea id="vorherige_operationen" name="vorherige_operationen" placeholder="z.B. Appendektomie vor 5 Jahren, Cholezystektomie...">{{ request.form.get('vorherige_operationen', '') }}</textarea>
                    </div>
                </div>
                
                <div class="toggle-section" id="vitalsToggle">
                    + Vitalparameter (optional)
                </div>
                
                <div id="vitalsSection" style="display: none;">
                    <div class="form-row">
                        <div class="form-col">
                            <div class="form-group">
                                <label for="hr">Herzfrequenz (HR):</label>
                                <input type="text" id="hr" name="hr" placeholder="z.B. 75" value="{{ request.form.get('hr', '') }}">
                            </div>
                        </div>
                        <div class="form-col">
                            <div class="form-group">
                                <label for="bp">Blutdruck (BP):</label>
                                <input type="text" id="bp" name="bp" placeholder="z.B. 120/80" value="{{ request.form.get('bp', '') }}">
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-col">
                            <div class="form-group">
                                <label for="temp">Temperatur (T):</label>
                                <input type="text" id="temp" name="temp" placeholder="z.B. 36.8" value="{{ request.form.get('temp', '') }}">
                            </div>
                        </div>
                        <div class="form-col">
                            <div class="form-group">
                                <label for="spo2">Sauerstoffsättigung (SpO2):</label>
                                <input type="text" id="spo2" name="spo2" placeholder="z.B. 98" value="{{ request.form.get('spo2', '') }}">
                            </div>
                        </div>
                    </div>
                </div>
<div class="toggle-section" id="clinicalExamToggle">
                    + Körperliche Untersuchung (optional)
                </div>
                
                <div id="clinicalExamSection" style="display: none;">
                    <div class="form-group">
                        <label for="cor">Cor:</label>
                        <input type="text" id="cor" name="cor" placeholder="z.B. Rhythmisch, keine Herzgeräusche" value="{{ request.form.get('cor', '') }}">
                    </div>
                    
                    <div class="form-group">
                        <label for="pulmo">Pulmo:</label>
                        <input type="text" id="pulmo" name="pulmo" placeholder="z.B. Vesikuläratmung, keine Rasselgeräusche" value="{{ request.form.get('pulmo', '') }}">
                    </div>
                    
                    <div class="form-group">
                        <label for="abdomen">Abdomen:</label>
                        <input type="text" id="abdomen" name="abdomen" placeholder="z.B. Weich, kein Druckschmerz, normale Darmgeräusche" value="{{ request.form.get('abdomen', '') }}">
                    </div>
                    
                    <div class="form-group">
                        <label for="hno">HNO (Rachen, Otoskopie):</label>
                        <input type="text" id="hno" name="hno" placeholder="z.B. Rachen gerötet, Tonsillen vergrößert, Trommelfell unauffällig" value="{{ request.form.get('hno', '') }}">
                    </div>
                </div>
                
                <div class="toggle-section" id="labValuesToggle">
                    + Laborwerte (optional)
                </div>
                
                <div id="labValuesSection" style="display: none;">
                    <div class="form-row">
                        <div class="form-col">
                            <div class="form-group">
                                <label for="leukozyten">Leukozyten (/nl):</label>
                                <input type="text" id="leukozyten" name="leukozyten" placeholder="z.B. 8.5" value="{{ request.form.get('leukozyten', '') }}">
                            </div>
                        </div>
                        <div class="form-col">
                            <div class="form-group">
                                <label for="erythrozyten">Erythrozyten (/pl):</label>
                                <input type="text" id="erythrozyten" name="erythrozyten" placeholder="z.B. 4.8" value="{{ request.form.get('erythrozyten', '') }}">
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-col">
                            <div class="form-group">
                                <label for="haemoglobin">Hämoglobin (g/dl):</label>
                                <input type="text" id="haemoglobin" name="haemoglobin" placeholder="z.B. 14.5" value="{{ request.form.get('haemoglobin', '') }}">
                            </div>
                        </div>
                        <div class="form-col">
                            <div class="form-group">
                                <label for="crp">CRP (mg/l):</label>
                                <input type="text" id="crp" name="crp" placeholder="z.B. 5.2" value="{{ request.form.get('crp', '') }}">
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="weitere_laborwerte">Weitere Laborwerte:</label>
                        <textarea id="weitere_laborwerte" name="weitere_laborwerte" placeholder="z.B. Kreatinin 0.9 mg/dl, ALT 25 U/l, Kalium 4.2 mmol/l...">{{ request.form.get('weitere_laborwerte', '') }}</textarea>
                    </div>
                </div>
                
                <div class="toggle-section" id="zusatzInfoToggle">
                    + Weitere Informationen (optional)
                </div>
                
                <div id="zusatzInfoSection" style="display: none;">
                    <div class="form-group">
                        <label for="zusatz_info">Zusätzliche relevante Informationen:</label>
                        <textarea id="zusatz_info" name="zusatz_info" placeholder="Beschreiben Sie besondere Umstände, Vorerkrankungen, Allergien, etc.">{{ request.form.get('zusatz_info', '') }}</textarea>
                    </div>
                </div>
                
                <button type="submit">Analysieren</button>
            </form>
            
            {% if ergebnis %}
            <div class="results-container">
                <div class="section-title">
                    <div class="section-icon">D</div>
                    <h3>Diagnose-Ergebnisse</h3>
                    
                    {% if ergebnis.source == "ml_model" %}
                        <span class="source-badge source-ml">ML-Modell</span>
                    {% elif ergebnis.source == "ml_llm" %}
                        <span class="source-badge source-llm">ML + LLM</span>
                    {% elif ergebnis.source == "ml_mimic" %}
                        <span class="source-badge source-llm">ML + MIMIC</span>
                    {% else %}
                        <span class="source-badge source-hybrid">Hybrid</span>
                    {% endif %}
                </div>
                
                {% if patient %}
                <div class="patient-summary">
                    <h4>Patienteninformation</h4>
                    <div class="info-grid">
                        <div class="info-item">
                            <strong>Alter:</strong> 
                            {% if patient.alter_jahre %}{{ patient.alter_jahre }} Jahre{% else %}{{ patient.alter_dropdown }}{% endif %}
                        </div>
                        {% if patient.geschlecht %}
                        <div class="info-item">
                            <strong>Geschlecht:</strong> {{ patient.geschlecht }}
                        </div>
                        {% endif %}
                        
                        {% if patient.vorerkrankungen %}
                        <div class="info-item info-full">
                            <strong>Vorerkrankungen:</strong> {{ patient.vorerkrankungen }}
                        </div>
                        {% endif %}
                        
                        {% if patient.vorherige_operationen %}
                        <div class="info-item info-full">
                            <strong>Vorherige Operationen:</strong> {{ patient.vorherige_operationen }}
                        </div>
                        {% endif %}
                    </div>
                    
                    {% if patient.untersuchung.cor or patient.untersuchung.pulmo or patient.untersuchung.abdomen or patient.untersuchung.hno %}
                    <h4>Untersuchungsbefund</h4>
                    <div class="info-grid">
                        {% if patient.untersuchung.cor %}
                        <div class="info-item">
                            <strong>Cor:</strong> {{ patient.untersuchung.cor }}
                        </div>
                        {% endif %}
                        
                        {% if patient.untersuchung.pulmo %}
                        <div class="info-item">
                            <strong>Pulmo:</strong> {{ patient.untersuchung.pulmo }}
                        </div>
                        {% endif %}
                        
                        {% if patient.untersuchung.abdomen %}
                        <div class="info-item">
                            <strong>Abdomen:</strong> {{ patient.untersuchung.abdomen }}
                        </div>
                        {% endif %}
                        
                        {% if patient.untersuchung.hno %}
                        <div class="info-item">
                            <strong>HNO:</strong> {{ patient.untersuchung.hno }}
                        </div>
                        {% endif %}
                    </div>
                    {% endif %}
                    
                    {% if patient.labor.leukozyten or patient.labor.erythrozyten or patient.labor.haemoglobin or patient.labor.crp or patient.labor.weitere %}
                    <h4>Laborwerte</h4>
                    <div class="info-grid">
                        {% if patient.labor.leukozyten %}
                        <div class="info-item">
                            <strong>Leukozyten:</strong> {{ patient.labor.leukozyten }}/nl
                        </div>
                        {% endif %}
                        
                        {% if patient.labor.erythrozyten %}
                        <div class="info-item">
                            <strong>Erythrozyten:</strong> {{ patient.labor.erythrozyten }}/pl
                        </div>
                        {% endif %}
                        
                        {% if patient.labor.haemoglobin %}
                        <div class="info-item">
                            <strong>Hämoglobin:</strong> {{ patient.labor.haemoglobin }} g/dl
                        </div>
                        {% endif %}
                        
                        {% if patient.labor.crp %}
                        <div class="info-item">
                            <strong>CRP:</strong> {{ patient.labor.crp }} mg/l
                        </div>
                        {% endif %}
                        
                        {% if patient.labor.weitere %}
                        <div class="info-item info-full">
                            <strong>Weitere Laborwerte:</strong> {{ patient.labor.weitere }}
                        </div>
                        {% endif %}
                    </div>
                    {% endif %}
                </div>
                {% endif %}
{% if ergebnis.result.fehler %}
                    <div class="info-box warning">
                        <p class="error">{{ ergebnis.result.fehler }}</p>
                        {% if ergebnis.result.nicht_erkannt %}
                            <p><strong>Nicht erkannte Symptome:</strong> 
                                {% for symptom in ergebnis.result.nicht_erkannt %}
                                    <span class="badge">{{ symptom }}</span>
                                {% endfor %}
                            </p>
                        {% endif %}
                    </div>
                {% else %}
                    {% if ergebnis.result.alter in ["Säugling", "Kleinkind", "Kind"] %}
                        <div class="pediatric-mode">
                            Pädiatrischer Modus: {{ ergebnis.result.alter }}
                        </div>
                    {% endif %}
                    
                    {% if ergebnis.result.top_diagnose %}
                        <div class="top-diagnosis">
                            <h4>Hauptdiagnose: {{ ergebnis.result.top_diagnose }}</h4>
                        </div>
                    {% endif %}
                    
                    {% if ergebnis.result.vitals %}
                    <div class="result-item">
                        <h4>Vitalparameter:</h4>
                        <div>
                            {% if 'HR' in ergebnis.result.vitals %}
                            <div class="vital-card">
                                <div class="vital-name">Puls</div>
                                <div class="vital-value">{{ ergebnis.result.vitals.HR }}</div>
                                <div class="vital-name">bpm</div>
                            </div>
                            {% endif %}
                            
                            {% if 'BP' in ergebnis.result.vitals %}
                            <div class="vital-card">
                                <div class="vital-name">Blutdruck</div>
                                <div class="vital-value">{{ ergebnis.result.vitals.BP }}</div>
                                <div class="vital-name">mmHg</div>
                            </div>
                            {% endif %}
                            
                            {% if 'T' in ergebnis.result.vitals %}
                            <div class="vital-card">
                                <div class="vital-name">Temperatur</div>
                                <div class="vital-value">{{ ergebnis.result.vitals.T }}</div>
                                <div class="vital-name">°C</div>
                            </div>
                            {% endif %}
                            
                            {% if 'SpO2' in ergebnis.result.vitals %}
                            <div class="vital-card">
                                <div class="vital-name">SpO2</div>
                                <div class="vital-value">{{ ergebnis.result.vitals.SpO2 }}</div>
                                <div class="vital-name">%</div>
                            </div>
                            {% endif %}
                        </div>
                        
                        {% if ergebnis.result.vital_warnings and ergebnis.result.vital_warnings|length > 0 %}
                        <div class="info-box warning" style="margin-top: 15px;">
                            <h4>Warnungen zu Vitalparametern:</h4>
                            <ul>
                                {% for warning in ergebnis.result.vital_warnings %}
                                <li class="vital-warning">{{ warning }}</li>
                                {% endfor %}
                            </ul>
                        </div>
                        {% endif %}
                    </div>
                    {% endif %}
                    
                    <div class="result-item">
                        <h4>Diagnosewahrscheinlichkeiten:</h4>
                        {% for diagnose, prozent in ergebnis.result.diagnosen.items() %}
                            <div class="probability-bar">
                                <div class="bar-label">{{ diagnose }}</div>
                                <div class="bar-container">
                                    <div class="bar" style="width: {{ prozent }}%;"></div>
                                </div>
                                <div class="bar-value">{{ prozent }}%</div>
                            </div>
                        {% endfor %}
                    </div>
                    
                    <!-- MIMIC ähnliche Fälle -->
                    {% if mimic_cases and mimic_cases|length > 0 %}
                    <div class="result-item">
                        <h4>Ähnliche Fälle aus der klinischen Datenbank:</h4>
                        <div class="info-box">
                            <p>Die folgenden Fälle aus der klinischen Datenbank haben ähnliche Charakteristika und können als zusätzliche Entscheidungshilfe dienen:</p>
                            
                            {% for case in mimic_cases %}
                            <div class="mimic-case">
                                <p><strong>Fall #{{ case.patient_id }}</strong> (Ähnlichkeit: {{ "%.0f"|format(case.similarity_score*100) }}%) - 
                                {{ case.age }} Jahre, {{ case.gender }}</p>
                                
                                {% if case.diagnoses %}
                                <p>Diagnosen: 
                                    {% for diagnosis in case.diagnoses %}
                                    <span class="badge">{{ diagnosis }}</span>
                                    {% endfor %}
                                </p>
                                {% endif %}
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    {% endif %}
                    
                    <div class="result-item">
                        <h4>Empfohlene Behandlung:</h4>
                        <p>{{ ergebnis.result.behandlung }}</p>
                    </div>
                    
                    <div class="result-item">
                        <h4>Abrechnungscode:</h4>
                        <p>{{ ergebnis.result.abrechnungscode }}</p>
                    </div>
                    
                    <div class="result-item">
                        <h4>DeSci-Status:</h4>
                        <p>{{ ergebnis.result.desci }}</p>
                    </div>
                    
                    {% if ergebnis.result.nicht_erkannt %}
                        <div class="info-box warning">
                            <h4>Nicht erkannte Symptome:</h4>
                            {% for symptom in ergebnis.result.nicht_erkannt %}
                                <span class="badge">{{ symptom }}</span>
                            {% endfor %}
                        </div>
                    {% endif %}
                {% endif %}
            </div>
            {% endif %}
        </div>
        
        <div id="chat-tab" class="tab-content">
            <h1>Medizinischer Chat-Assistent</h1>
            
            <div class="info-box">
                <p>Hier können Sie medizinische Fragen stellen und Informationen zu Krankheitsbildern, Symptomen, Behandlungen oder Medikamenten erhalten.</p>
                <p>Der Chat-Assistent kann auch bei der medizinischen Entscheidungsfindung unterstützen.</p>
            </div>
            
            <div class="chat-container">
                <div class="chat-messages" id="chat-messages">
                    {% if chat_messages %}
                        {% for message in chat_messages %}
                            <div class="message {{ 'user-message' if message.sender == 'user' else 'assistant-message' }}">
                                {{ message.text }}
                                <div class="message-time">{{ message.time }}</div>
                            </div>
                        {% endfor %}
                    {% else %}
                        <div class="message assistant-message">
                            Hallo! Ich bin Ihr medizinischer Assistent. Wie kann ich Ihnen heute helfen?
                            <div class="message-time">Jetzt</div>
                        </div>
                    {% endif %}
                </div>
                <div class="chat-input">
                    <form method="POST" action="/chat" id="chat-form">
                        <input type="text" id="chat-message" name="message" placeholder="Stellen Sie eine medizinische Frage..." required>
                        <button type="submit">Senden</button>
                    </form>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Autovervollständigung für Symptome
        function autocomplete(input, suggestions) {
            const suggestionsDiv = document.getElementById('suggestions');
            
            // Schließen der Vorschlagsliste, wenn woanders geklickt wird
            document.addEventListener('click', function(e) {
                if (e.target !== input) {
                    suggestionsDiv.innerHTML = '';
                }
            });
            
            input.addEventListener('input', function() {
                const value = this.value.toLowerCase();
                const currentWord = getCurrentWord(this.value);
                suggestionsDiv.innerHTML = '';
                
                if (currentWord.length < 2) return;
                
                const matches = suggestions.filter(s => 
                    s.toLowerCase().includes(currentWord.toLowerCase())
                ).slice(0, 7); // Top 7 Vorschläge
                
                matches.forEach(suggestion => {
                    const div = document.createElement('div');
                    div.textContent = suggestion;
                    div.addEventListener('click', () => {
                        replaceCurrentWord(input, suggestion);
                        suggestionsDiv.innerHTML = '';
                        input.focus();
                    });
                    suggestionsDiv.appendChild(div);
                });
            });
            
            // Funktion, um das aktuelle Wort zu extrahieren
            function getCurrentWord(text) {
                const lastCommaIndex = text.lastIndexOf(',');
                if (lastCommaIndex === -1) return text.trim();
                return text.substring(lastCommaIndex + 1).trim();
            }
            
            // Funktion, um das aktuelle Wort zu ersetzen
            function replaceCurrentWord(input, replacement) {
                const text = input.value;
                const lastCommaIndex = text.lastIndexOf(',');
                if (lastCommaIndex === -1) {
                    input.value = replacement + ', ';
                } else {
                    input.value = text.substring(0, lastCommaIndex + 1) + ' ' + replacement + ', ';
                }
            }
        }
        
        // Tab-Steuerung
        function setupTabs() {
            const tabs = document.querySelectorAll('.tab');
            const tabContents = document.querySelectorAll('.tab-content');
            
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    const tabId = tab.getAttribute('data-tab');
                    
                    // Aktive Klasse von allen Tabs entfernen
                    tabs.forEach(t => t.classList.remove('active'));
                    tabContents.forEach(c => c.classList.remove('active'));
                    
                    // Aktive Klasse zum ausgewählten Tab hinzufügen
                    tab.classList.add('active');
                    document.getElementById(tabId + '-tab').classList.add('active');
                });
            });
        }
        
        // Toggle-Sektionen anzeigen/verbergen
        function setupToggleSections() {
            const vitalsToggle = document.getElementById('vitalsToggle');
            const vitalsSection = document.getElementById('vitalsSection');
            const zusatzInfoToggle = document.getElementById('zusatzInfoToggle');
            const zusatzInfoSection = document.getElementById('zusatzInfoSection');
            const patientInfoToggle = document.getElementById('patientInfoToggle');
            const patientInfoSection = document.getElementById('patientInfoSection');
            const clinicalExamToggle = document.getElementById('clinicalExamToggle');
            const clinicalExamSection = document.getElementById('clinicalExamSection');
            const labValuesToggle = document.getElementById('labValuesToggle');
            const labValuesSection = document.getElementById('labValuesSection');
            
            vitalsToggle.addEventListener('click', function() {
                if (vitalsSection.style.display === 'none') {
                    vitalsSection.style.display = 'block';
                    vitalsToggle.textContent = '- Vitalparameter (optional)';
                } else {
                    vitalsSection.style.display = 'none';
                    vitalsToggle.textContent = '+ Vitalparameter (optional)';
                }
            });
            
            zusatzInfoToggle.addEventListener('click', function() {
                if (zusatzInfoSection.style.display === 'none') {
                    zusatzInfoSection.style.display = 'block';
                    zusatzInfoToggle.textContent = '- Weitere Informationen (optional)';
                } else {
                    zusatzInfoSection.style.display = 'none';
                    zusatzInfoToggle.textContent = '+ Weitere Informationen (optional)';
                }
            });
            
            patientInfoToggle.addEventListener('click', function() {
                if (patientInfoSection.style.display === 'none') {
                    patientInfoSection.style.display = 'block';
                    patientInfoToggle.textContent = '- Patienteninformation (optional)';
                } else {
                    patientInfoSection.style.display = 'none';
                    patientInfoToggle.textContent = '+ Patienteninformation (optional)';
                }
            });
            
            clinicalExamToggle.addEventListener('click', function() {
                if (clinicalExamSection.style.display === 'none') {
                    clinicalExamSection.style.display = 'block';
                    clinicalExamToggle.textContent = '- Körperliche Untersuchung (optional)';
                } else {
                    clinicalExamSection.style.display = 'none';
                    clinicalExamToggle.textContent = '+ Körperliche Untersuchung (optional)';
                }
            });
            
            labValuesToggle.addEventListener('click', function() {
                if (labValuesSection.style.display === 'none') {
                    labValuesSection.style.display = 'block';
                    labValuesToggle.textContent = '- Laborwerte (optional)';
                } else {
                    labValuesSection.style.display = 'none';
                    labValuesToggle.textContent = '+ Laborwerte (optional)';
                }
            });
            
            // Zeige Bereiche, wenn Werte bereits eingegeben wurden
            if ({{ 'true' if request.form.get('hr') or request.form.get('bp') or request.form.get('temp') or request.form.get('spo2') else 'false' }}) {
                vitalsSection.style.display = 'block';
                vitalsToggle.textContent = '- Vitalparameter (optional)';
            }
            
            if ({{ 'true' if request.form.get('zusatz_info') else 'false' }}) {
                zusatzInfoSection.style.display = 'block';
                zusatzInfoToggle.textContent = '- Weitere Informationen (optional)';
            }
            
            if ({{ 'true' if request.form.get('geschlecht') or request.form.get('alter_jahre') or request.form.get('vorerkrankungen') or request.form.get('vorherige_operationen') else 'false' }}) {
                patientInfoSection.style.display = 'block';
                patientInfoToggle.textContent = '- Patienteninformation (optional)';
            }
            
            if ({{ 'true' if request.form.get('cor') or request.form.get('pulmo') or request.form.get('abdomen') or request.form.get('hno') else 'false' }}) {
                clinicalExamSection.style.display = 'block';
                clinicalExamToggle.textContent = '- Körperliche Untersuchung (optional)';
            }
            
            if ({{ 'true' if request.form.get('leukozyten') or request.form.get('erythrozyten') or request.form.get('haemoglobin') or request.form.get('crp') or request.form.get('weitere_laborwerte') else 'false' }}) {
                labValuesSection.style.display = 'block';
                labValuesToggle.textContent = '- Laborwerte (optional)';
            }
        }
        
        // Automatische Vorschläge basierend auf Altersauswahl
        function setupAgeRelatedSuggestions() {
            const alterSelect = document.getElementById('alter');
            alterSelect.addEventListener('change', function() {
                const selectedAlter = this.value;
                const symptomInput = document.getElementById('symptome');
                
                if (selectedAlter === 'Säugling' && symptomInput.value === '') {
                    symptomInput.value = 'Säuglingshusten, ';
                } else if (selectedAlter === 'Kleinkind' && symptomInput.value === '') {
                    symptomInput.value = 'Kindliches Fieber, ';
                }
            });
        }
        
        // Dokument geladen
        document.addEventListener('DOMContentLoaded', function() {
            const input = document.getElementById('symptome');
            autocomplete(input, {{ symptome|tojson }});
            setupTabs();
            setupToggleSections();
            setupAgeRelatedSuggestions();
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)