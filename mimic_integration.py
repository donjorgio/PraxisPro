# mimic_integration.py - Integration mit PhysioNet-MIMIC für erweiterte Diagnoseanalyse

import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mimic_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mimic_integration")

class MIMICIntegration:
    """
    Klasse zur Integration mit PhysioNet-MIMIC Datenbank für erweitertes Reasoning bei Diagnosen.
    """
    
    def __init__(self, mimic_dir=None):
        """
        Initialisiert die MIMIC-Integration.
        
        Args:
            mimic_dir (str): Pfad zum Verzeichnis mit den MIMIC-CSV-Dateien.
                             Wenn None, wird versucht einen Standard-Pfad zu nutzen.
        """
        self.mimic_dir = mimic_dir if mimic_dir else os.path.join(os.path.dirname(__file__), 'mimic_data')
        self.patients_df = None
        self.diagnoses_df = None
        self.vitals_df = None
        self.labs_df = None
        self.symptom_mapper = self._create_symptom_mapper()
        self.is_initialized = False
        
    def _create_symptom_mapper(self):
        """Erstellt eine Mapping-Tabelle von deutschen Symptomen zu englischen ICD-Codes"""
        # Basiskonvertierung von häufigen Symptomen zu ICD-Codes
        return {
            "fieber": ["R50.9", "R50.0"],
            "brustschmerzen": ["R07.1", "R07.2", "R07.9", "I20.9"],
            "atemnot": ["R06.0", "J80", "R06.89"],
            "husten": ["R05"],
            "erbrechen": ["R11.1", "R11.2"],
            "durchfall": ["A09", "K52.9"],
            "kopfschmerzen": ["R51", "G43.9", "G44.1"],
            "bauchschmerzen": ["R10.0", "R10.1", "R10.2", "R10.9"],
            "rückenschmerzen": ["M54.5", "M54.9"],
            "schwindel": ["R42", "H81.1", "H81.3"],
            "gelenkschmerzen": ["M25.5", "M79.1"],
            "hautausschlag": ["R21", "L50.9"],
            "halsschmerzen": ["J02.9", "J03.9"],
            "ohrenschmerzen": ["H92.0", "H66.9"],
            "müdigkeit": ["R53.83", "F48.0"],
            "dysurie": ["R30.0", "N39.0"],
            "pollakisurie": ["R35.0", "N39.0"]
        }
    
    def initialize_database(self, force_reload=False):
        """
        Lädt die MIMIC-Datenbank aus CSV-Dateien, wenn verfügbar.
        
        Args:
            force_reload (bool): Wenn True, werden Daten neu geladen, auch wenn sie bereits geladen sind.
        
        Returns:
            bool: True, wenn die Initialisierung erfolgreich war, sonst False.
        """
        if self.is_initialized and not force_reload:
            return True
            
        try:
            # Versuche, MIMIC-Dateien zu laden, wenn vorhanden
            # Hier verwenden wir stark vereinfachte Versionen der MIMIC-Tabellen
            
            patient_file = os.path.join(self.mimic_dir, 'patients.csv')
            diagnoses_file = os.path.join(self.mimic_dir, 'diagnoses.csv')
            vitals_file = os.path.join(self.mimic_dir, 'vitals.csv')
            labs_file = os.path.join(self.mimic_dir, 'labs.csv')
            
            # Prüfen, ob Dateien existieren
            if not all(os.path.exists(f) for f in [patient_file, diagnoses_file, vitals_file, labs_file]):
                logger.warning("Nicht alle benötigten MIMIC-Datendateien gefunden.")
                self._setup_mock_data()  # Erstellt Mock-Daten für Testzwecke
            else:
                # Lade echte MIMIC-Daten
                self.patients_df = pd.read_csv(patient_file)
                self.diagnoses_df = pd.read_csv(diagnoses_file)
                self.vitals_df = pd.read_csv(vitals_file)
                self.labs_df = pd.read_csv(labs_file)
            
            self.is_initialized = True
            logger.info("MIMIC-Integration erfolgreich initialisiert.")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung der MIMIC-Integration: {str(e)}")
            return False
    
    def _setup_mock_data(self):
        """Erstellt Mock-Daten für Testzwecke, wenn keine MIMIC-Dateien gefunden werden."""
        # Mock-Patientendaten
        self.patients_df = pd.DataFrame({
            'patient_id': range(1, 101),
            'age': np.random.randint(1, 90, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            'weight': np.random.normal(70, 15, 100),
            'height': np.random.normal(170, 10, 100)
        })
        
        # Mock-Diagnosedaten
        diagnoses = []
        for i in range(1, 101):
            num_diagnoses = np.random.randint(1, 5)
            for j in range(num_diagnoses):
                diagnoses.append({
                    'patient_id': i,
                    'icd_code': np.random.choice(['I21.9', 'J18.9', 'K35.80', 'N39.0', 'R07.9', 'J44.9', 'E11.9']),
                    'diagnosis': np.random.choice(['Myokardinfarkt', 'Pneumonie', 'Appendizitis', 'Harnwegsinfektion', 
                                                  'Brustschmerzen', 'COPD', 'Diabetes mellitus Typ 2'])
                })
        self.diagnoses_df = pd.DataFrame(diagnoses)
        
        # Mock-Vitalparameter
        vitals = []
        for i in range(1, 101):
            vitals.append({
                'patient_id': i,
                'heart_rate': np.random.normal(80, 15),
                'blood_pressure_systolic': np.random.normal(120, 20),
                'blood_pressure_diastolic': np.random.normal(80, 10),
                'temperature': np.random.normal(37, 1),
                'respiratory_rate': np.random.normal(16, 2),
                'oxygen_saturation': np.random.normal(97, 2)
            })
        self.vitals_df = pd.DataFrame(vitals)
        
        # Mock-Laborwerte
        labs = []
        for i in range(1, 101):
            labs.append({
                'patient_id': i,
                'wbc': np.random.normal(8, 2),  # Leukozyten
                'hgb': np.random.normal(14, 2),  # Hämoglobin
                'plt': np.random.normal(250, 50),  # Thrombozyten
                'crp': np.random.exponential(5),  # CRP
                'crea': np.random.normal(1, 0.2),  # Kreatinin
                'glu': np.random.normal(100, 20)  # Glukose
            })
        self.labs_df = pd.DataFrame(labs)
        
        logger.info("Mock-Daten für MIMIC-Integration erstellt.")
    
    def get_similar_cases(self, symptome=None, vitals=None, alter=None, geschlecht=None, laborwerte=None, max_cases=5):
        """
        Findet ähnliche Fälle aus der MIMIC-Datenbank basierend auf den klinischen Parametern.
        
        Args:
            symptome (str): Kommaseparierte Liste von Symptomen
            vitals (str): String mit Vitalparametern im Format "HR:80,BP:120/80,T:36.8,SpO2:98"
            alter (str): Altersgruppe oder spezifisches Alter
            geschlecht (str): Geschlecht des Patienten
            laborwerte (dict): Dictionary mit Laborwerten
            max_cases (int): Maximale Anzahl zurückzugebender Fälle
            
        Returns:
            list: Liste ähnlicher Fälle mit ihren Diagnosen
        """
        if not self.is_initialized:
            self.initialize_database()
            
        if not self.is_initialized:
            logger.error("MIMIC-Datenbank konnte nicht initialisiert werden.")
            return []
        
        try:
            # Feature-Extraktion aus den Patientendaten
            current_features = self._extract_features(symptome, vitals, alter, geschlecht, laborwerte)
            
            # Feature-Extraktion für alle MIMIC-Patienten
            all_patients_features = []
            for idx, patient in self.patients_df.iterrows():
                patient_id = patient['patient_id']
                
                # Vitalparameter für diesen Patienten abrufen
                patient_vitals = self.vitals_df[self.vitals_df['patient_id'] == patient_id].iloc[0] if len(self.vitals_df[self.vitals_df['patient_id'] == patient_id]) > 0 else None
                
                # Laborwerte für diesen Patienten abrufen
                patient_labs = self.labs_df[self.labs_df['patient_id'] == patient_id].iloc[0] if len(self.labs_df[self.labs_df['patient_id'] == patient_id]) > 0 else None
                
                # Features für diesen Patienten extrahieren
                patient_features = []
                
                # Alter
                patient_features.append(patient['age'])
                
                # Geschlecht (binär kodiert)
                patient_features.append(1 if patient['gender'] == 'M' else 0)
                
                # Vitalparameter
                if patient_vitals is not None:
                    patient_features.extend([
                        patient_vitals['heart_rate'],
                        patient_vitals['blood_pressure_systolic'],
                        patient_vitals['blood_pressure_diastolic'],
                        patient_vitals['temperature'],
                        patient_vitals['respiratory_rate'],
                        patient_vitals['oxygen_saturation']
                    ])
                else:
                    patient_features.extend([np.nan] * 6)
                
                # Laborwerte
                if patient_labs is not None:
                    patient_features.extend([
                        patient_labs['wbc'],
                        patient_labs['hgb'],
                        patient_labs['plt'],
                        patient_labs['crp'],
                        patient_labs['crea'],
                        patient_labs['glu']
                    ])
                else:
                    patient_features.extend([np.nan] * 6)
                
                all_patients_features.append(patient_features)
            
            # DataFrame mit allen Patientenfeatures erstellen
            all_features_df = pd.DataFrame(all_patients_features)
            
            # Fehlende Werte durch Mittelwerte ersetzen
            all_features_df.fillna(all_features_df.mean(), inplace=True)
            current_features_array = np.array(current_features).reshape(1, -1)
            current_features_df = pd.DataFrame(current_features_array)
            current_features_df.fillna(all_features_df.mean(), inplace=True)
            
            # Standardisierung der Features
            scaler = StandardScaler()
            all_features_scaled = scaler.fit_transform(all_features_df)
            current_features_scaled = scaler.transform(current_features_df)
            
            # Nächste Nachbarn finden
            knn = NearestNeighbors(n_neighbors=min(max_cases, len(all_features_scaled)))
            knn.fit(all_features_scaled)
            distances, indices = knn.kneighbors(current_features_scaled)
            
            # Ähnliche Fälle und ihre Diagnosen extrahieren
            similar_cases = []
            for i, idx in enumerate(indices[0]):
                patient_id = self.patients_df.iloc[idx]['patient_id']
                diagnoses = self.diagnoses_df[self.diagnoses_df['patient_id'] == patient_id]
                
                case = {
                    'patient_id': int(patient_id),
                    'similarity_score': 1.0 / (1.0 + distances[0][i]),  # Normalisierte Ähnlichkeit
                    'age': int(self.patients_df.iloc[idx]['age']),
                    'gender': self.patients_df.iloc[idx]['gender'],
                    'diagnoses': diagnoses['diagnosis'].tolist() if not diagnoses.empty else []
                }
                similar_cases.append(case)
            
            return similar_cases
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen ähnlicher Fälle: {str(e)}")
            return []
    
    def _extract_features(self, symptome, vitals, alter, geschlecht, laborwerte):
        """
        Extrahiert numerische Features aus den klinischen Parametern.
        
        Returns:
            list: Liste numerischer Features
        """
        features = []
        
        # Alter (numerisch)
        try:
            if alter in ["Säugling"]:
                alter_numeric = 0.5
            elif alter in ["Kleinkind"]:
                alter_numeric = 3
            elif alter in ["Kind"]:
                alter_numeric = 10
            elif alter in ["Jugendlicher"]:
                alter_numeric = 16
            elif alter in ["Erwachsener"]:
                alter_numeric = 40
            else:
                # Versuche, numerisches Alter zu extrahieren
                alter_numeric = float(alter.split()[0]) if alter and any(c.isdigit() for c in alter) else 40
        except Exception:
            alter_numeric = 40  # Standardwert
        features.append(alter_numeric)
        
        # Geschlecht (binär)
        geschlecht_numeric = 1 if geschlecht and geschlecht.lower() in ['m', 'männlich', 'male'] else 0
        features.append(geschlecht_numeric)
        
        # Vitalparameter
        vitals_dict = {}
        if vitals:
            for pair in vitals.split(','):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    vitals_dict[key.strip()] = value.strip()
        
        # Standardwerte für fehlende Vitalparameter
        hr = float(vitals_dict.get('HR', 80))
        
        # Blutdruck extrahieren (systolisch und diastolisch)
        bp_sys, bp_dia = 120, 80
        if 'BP' in vitals_dict and '/' in vitals_dict['BP']:
            try:
                bp_sys, bp_dia = map(float, vitals_dict['BP'].split('/'))
            except ValueError:
                pass
        
        temp = float(vitals_dict.get('T', 37.0))
        spo2 = float(vitals_dict.get('SpO2', 97))
        resp_rate = 16  # Standardwert für Atemfrequenz
        
        features.extend([hr, bp_sys, bp_dia, temp, resp_rate, spo2])
        
        # Laborwerte 
        lab_values = [np.nan] * 6  # Standardmäßig NaN für Laborwerte
        if laborwerte:
            lab_values = [
                float(laborwerte.get('leukozyten', np.nan)),
                float(laborwerte.get('haemoglobin', np.nan)),
                float(laborwerte.get('thrombozyten', 250)),
                float(laborwerte.get('crp', np.nan)),
                float(laborwerte.get('kreatinin', np.nan)),
                float(laborwerte.get('glucose', np.nan))
            ]
        features.extend(lab_values)
        
        return features
    
    def adjust_probabilities(self, diagnosen, similar_cases):
        """
        Passt die Diagnosewahrscheinlichkeiten basierend auf ähnlichen MIMIC-Fällen an.
        Mit verstärkter Gewichtung für übereinstimmende Diagnosen und höherer Konfidenz
        bei mehrfacher Bestätigung durch die Datenbank.
        
        Args:
            diagnosen (dict): Dictionary mit Diagnosen und Wahrscheinlichkeiten
            similar_cases (list): Liste ähnlicher Fälle aus der MIMIC-Datenbank
            
        Returns:
            dict: Angepasstes Dictionary mit Diagnosen und Wahrscheinlichkeiten
        """
        if not similar_cases:
            return diagnosen
        
        adjusted_diagnosen = diagnosen.copy()
        total_similarity = sum(case['similarity_score'] for case in similar_cases)
        
        # Zählen, wie oft jede Diagnose in ähnlichen Fällen vorkommt (gewichtet nach Ähnlichkeit)
        diagnosis_weights = {}
        diagnosis_counts = {}  # Zählt, in wie vielen Fällen eine Diagnose auftaucht
        
        for case in similar_cases:
            similarity = case['similarity_score']
            for diagnosis in case['diagnoses']:
                diagnosis = self._normalize_diagnosis_name(diagnosis)
                
                # Gewichtssumme aktualisieren
                if diagnosis not in diagnosis_weights:
                    diagnosis_weights[diagnosis] = 0
                    diagnosis_counts[diagnosis] = 0
                
                diagnosis_weights[diagnosis] += similarity / total_similarity
                diagnosis_counts[diagnosis] += 1
        
        # Diagnosewahrscheinlichkeiten anpassen mit höherer Konfidenz
        for diagnosis, weight in diagnosis_weights.items():
            best_match = self._find_best_match(diagnosis, adjusted_diagnosen.keys())
            if best_match:
                # Höhere Verstärkung für mehrfach auftretende Diagnosen
                if diagnosis_counts[diagnosis] > 1:
                    # Stärkere Verstärkung bei mehrfacher Bestätigung (2x oder mehr)
                    boost_factor = 1.0 + (weight * 2.5) + (diagnosis_counts[diagnosis] * 0.5)
                else:
                    # Moderate Verstärkung bei einfacher Bestätigung
                    boost_factor = 1.0 + (weight * 2.0)
                
                # Zusätzlicher Konfidenzboost für exakte Übereinstimmungen
                if diagnosis.lower() == best_match.lower():
                    boost_factor += 0.5
                    
                adjusted_diagnosen[best_match] *= boost_factor
        
        # Besondere Behandlung für Top-Diagnosen
        # Wenn eine Diagnose in mehreren MIMIC-Fällen die Hauptdiagnose ist, verstärken wir sie zusätzlich
        top_diagnoses = {}
        for case in similar_cases:
            if case['diagnoses'] and len(case['diagnoses']) > 0:
                top_diagnosis = self._normalize_diagnosis_name(case['diagnoses'][0])
                if top_diagnosis not in top_diagnoses:
                    top_diagnoses[top_diagnosis] = 0
                top_diagnoses[top_diagnosis] += case['similarity_score']
        
        # Starker Boost für gemeinsame Top-Diagnosen
        for diagnosis, score in top_diagnoses.items():
            if score > 0.5 * total_similarity:  # Wenn die Diagnose in mehr als der Hälfte der wichtigen Fälle vorkommt
                best_match = self._find_best_match(diagnosis, adjusted_diagnosen.keys())
                if best_match:
                    adjusted_diagnosen[best_match] *= 1.8  # Zusätzlicher Faktor für häufige Top-Diagnosen
        
        # Normalisieren, damit die Summe wieder 100% ergibt
        total = sum(adjusted_diagnosen.values())
        if total > 0:
            for diagnosis in adjusted_diagnosen:
                adjusted_diagnosen[diagnosis] = round((adjusted_diagnosen[diagnosis] / total) * 100, 1)
        
        return adjusted_diagnosen
    
    def _normalize_diagnosis_name(self, diagnosis):
        """Normalisiert Diagnosenamen für besseren Vergleich"""
        diagnosis = diagnosis.lower()
        # Entferne häufige Präfixe und Suffixe
        for prefix in ["akut", "chronisch", "akuter", "chronischer"]:
            if diagnosis.startswith(prefix):
                diagnosis = diagnosis[len(prefix):].strip()
                
        # Entferne ICD-Codes in Klammern, falls vorhanden
        if "(" in diagnosis and ")" in diagnosis:
            start = diagnosis.find("(")
            end = diagnosis.find(")")
            if end > start:
                diagnosis = diagnosis[:start] + diagnosis[end+1:]
                
        return diagnosis.strip()
    
    def _find_best_match(self, mimic_diagnosis, current_diagnoses):
        """Findet die beste Übereinstimmung zwischen MIMIC-Diagnose und aktuellen Diagnosen"""
        mimic_diagnosis = mimic_diagnosis.lower()
        
        # Exakte Übereinstimmung
        for diagnosis in current_diagnoses:
            if mimic_diagnosis == diagnosis.lower():
                return diagnosis
        
        # Teilweise Übereinstimmung
        best_match = None
        best_score = 0
        
        for diagnosis in current_diagnoses:
            # Berechne Jaccard-Ähnlichkeit der Wörter
            dx_words = set(diagnosis.lower().split())
            mimic_words = set(mimic_diagnosis.split())
            
            if not dx_words or not mimic_words:
                continue
                
            intersection = len(dx_words.intersection(mimic_words))
            union = len(dx_words.union(mimic_words))
            
            score = intersection / union if union > 0 else 0
            
            # Prüfe auf Teilstring-Übereinstimmung
            if mimic_diagnosis in diagnosis.lower() or diagnosis.lower() in mimic_diagnosis:
                score += 0.3  # Bonus für Teilstring-Übereinstimmung
            
            if score > best_score:
                best_score = score
                best_match = diagnosis
        
        # Nur zurückgeben, wenn es eine gewisse Ähnlichkeit gibt
        return best_match if best_score > 0.3 else None