# ============================================================
# MODULE 1: Intelligent Agent — Healthcare Diagnostic Agent
# Covers: Week 2 (Intelligent Agents) + PEAS Framework
# ============================================================

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import datetime
import csv
import os

def load_valid_symptoms(csv_path="data/symptoms.csv"):
    """Load the master symptom list so perceive() can catch typos early."""
    valid = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            valid = {row["symptom_name"] for row in reader}
    return valid

VALID_SYMPTOMS = load_valid_symptoms()
class AgentState(Enum):
    IDLE         = "idle"
    COLLECTING   = "collecting_symptoms"
    DIAGNOSING   = "diagnosing"
    RECOMMENDING = "recommending"
    PLANNING     = "planning_treatment"
    DONE         = "done"

@dataclass
class PatientPercept:
    """What the agent perceives from the environment"""
    patient_id:   str
    symptoms:     List[str]
    age:          int
    temperature:  float
    heart_rate:   int
    blood_pressure: str
    timestamp:    str = field(
        default_factory=lambda: datetime.datetime.now().isoformat())

@dataclass
class AgentMemory:
    """Internal model — makes this a model-based agent"""
    patient_history:  List[Dict]  = field(default_factory=list)
    current_patient:  Optional[PatientPercept] = None
    diagnosis_history: List[str]  = field(default_factory=list)
    action_log:       List[str]   = field(default_factory=list)

class HealthcareDiagnosticAgent:
    """
    PEAS Definition:
    ─────────────────────────────────────────────────
    Performance : Diagnostic accuracy, patient safety,
                  recommendation quality, response time
    Environment : Hospital/clinic, patient data, EMR
    Actuators   : Diagnosis report, treatment plan,
                  referral recommendation, alerts
    Sensors     : Symptom input, vitals, lab results,
                  patient history
    ─────────────────────────────────────────────────
    Agent Type  : Model-Based + Goal-Based + Learning
    """

    def __init__(self):
        self.state   = AgentState.IDLE
        self.memory  = AgentMemory()
        self.performance_score = 0
        self._modules = {}  # Will hold sub-modules
        self._excluded_symptoms = []  # populated by perceive() when any
                                       # symptom isn't recognized

    def register_module(self, name: str, module):
        """Plug in AI sub-modules (KB, Bayes, ML, etc.)"""
        if not hasattr(module, 'analyze'):
            raise ValueError(
                f"Module '{name}' must implement an analyze(patient) method "
                f"before it can be registered."
            )
        self._modules[name] = module
        print(f"  🔌 Module registered: [{name}]")

    def perceive(self, percept: PatientPercept):
        """Step 1: Perceive the environment"""
        self._excluded_symptoms = []

        if VALID_SYMPTOMS:
            unknown = [s for s in percept.symptoms if s not in VALID_SYMPTOMS]
            if unknown:
                # CHANGED: this used to raise a ValueError and reject the
                # whole patient outright. That's too harsh for a live
                # system -- one unrecognized symptom (a typo, a symptom
                # genuinely outside our 26-symptom vocabulary, etc.)
                # shouldn't stop diagnosis entirely when the patient may
                # have described several OTHER perfectly valid symptoms.
                #
                # Now: warn, strip out only the unrecognized ones, and
                # keep going with whatever WAS recognized. The excluded
                # list is stored so act() can surface it in the final
                # report -- the patient/clinician should know some of
                # what was described got dropped, not have it silently
                # vanish.
                print(f"  ⚠️  Warning: unrecognized symptom(s) ignored: "
                      f"{unknown}. Check data/symptoms.csv for the master "
                      f"list. Diagnosis will proceed using only the "
                      f"recognized symptoms.")
                self._excluded_symptoms = unknown
                percept.symptoms = [s for s in percept.symptoms if s in VALID_SYMPTOMS]

        self.memory.current_patient = percept
        self.memory.patient_history.append({
            'id': percept.patient_id,
            'symptoms': percept.symptoms,
            'time': percept.timestamp
        })
        self.state = AgentState.COLLECTING
        self._log(f"Perceived patient {percept.patient_id} "
                  f"with {len(percept.symptoms)} symptoms")
        return self

    def think(self):
        """Step 2: Process and reason"""
        self.state = AgentState.DIAGNOSING
        self._log("Agent thinking: running diagnostic modules...")

        results = {}

        # Run each registered module
        for module_name, module in self._modules.items():
            if hasattr(module, 'analyze'):
                try:
                    result = module.analyze(self.memory.current_patient)
                except Exception as e:
                    # One module failing must not crash the whole diagnostic run --
                    # the other modules' opinions are still valuable.
                    result = {'diagnosis': None, 'confidence': 0.0, 'error': str(e)}
                    self._log(f"  [{module_name}] → ERROR: {e}")
                else:
                    self._log(f"  [{module_name}] → {result.get('summary','done')}")
                results[module_name] = result

        self.memory.diagnosis_history.append(results)
        self.state = AgentState.RECOMMENDING
        return results

    def act(self, diagnosis_results: Dict) -> Dict:
        """Step 3: Generate action/recommendation"""
        self.state = AgentState.PLANNING
        patient = self.memory.current_patient

        # Aggregate confidence from multiple modules
        # Same exclusion as _aggregate_diagnosis(): a module that
        # abstained (diagnosis == "Unknown", filler confidence 0.5)
        # shouldn't drag down the average confidence of the modules
        # that actually found something.
        confidences = [
            v.get('confidence', 0)
            for v in diagnosis_results.values()
            if isinstance(v, dict) and 'confidence' in v
            and v.get('diagnosis') != 'Unknown'
        ]
        avg_confidence = sum(confidences)/len(confidences) if confidences else 0.5

        # Determine urgency
        urgency = self._assess_urgency(patient, avg_confidence)

        action_report = {
            'patient_id':   patient.patient_id,
            'timestamp':    patient.timestamp,
            'symptoms':     patient.symptoms,
            'excluded_symptoms': self._excluded_symptoms,
            'diagnosis':    self._aggregate_diagnosis(diagnosis_results),
            'confidence':   round(avg_confidence, 3),
            'urgency':      urgency,
            'recommendations': self._generate_recommendations(
                urgency, diagnosis_results),
            'next_action':  self._decide_next_action(urgency)
        }

        self.performance_score += (10 if avg_confidence > 0.7 else 5)
        self.state = AgentState.DONE
        self._log(f"Action generated: {urgency} urgency")
        return action_report

    def run(self, percept: PatientPercept) -> Dict:
        """Full agent cycle: Perceive → Think → Act"""
        self.perceive(percept)
        results = self.think()
        return self.act(results)

    def _assess_urgency(self, patient, confidence):
        if patient.temperature > 39.5 or patient.heart_rate > 120:
            return "CRITICAL"
        elif patient.temperature > 38.5 or confidence > 0.8:
            return "HIGH"
        elif patient.temperature > 37.5:
            return "MEDIUM"
        return "LOW"

    def _aggregate_diagnosis(self, results):
        # NOTE: Module 2 (Knowledge Base) returns diagnosis names like
        # "covid19_suspected" or "covid19_confirmed", while Modules 3/4/5
        # (Bayesian, ML, Neural Network) return bare disease names like
        # "covid19". Without normalizing these, the Counter-based majority
        # vote below would NEVER see these as the same diagnosis, even
        # when every module is correctly pointing at the same disease --
        # defeating the whole point of aggregating multiple modules'
        # opinions. Stripping the suffix here fixes that.
        #
        # ALSO: a module that found no matching rules/pattern (e.g.
        # Module 2 when no rule's conditions are fully met) returns
        # diagnosis="Unknown" with a filler confidence of 0.5. This is
        # NOT a real vote for a disease called "Unknown" -- it's the
        # module abstaining. Including it in the majority vote would
        # let "not sure" compete against actual disease predictions,
        # and including its filler 0.5 confidence in the average (see
        # act() below) would understate the real diagnostic confidence
        # of the modules that DID find something. Both are excluded here.
        diagnoses = [
            v.get('diagnosis', 'Unknown')
                .replace('_suspected', '')
                .replace('_confirmed', '')
            for v in results.values()
            if isinstance(v, dict) and v.get('diagnosis')
            and v.get('diagnosis') != 'Unknown'
        ]
        if not diagnoses:
            return "Insufficient data"
        from collections import Counter
        return Counter(diagnoses).most_common(1)[0][0]

    def _generate_recommendations(self, urgency, results):
        base = {
            "CRITICAL": [
                "🚨 Immediate emergency consultation required",
                "📞 Alert attending physician now",
                "🏥 Transfer to emergency ward",
                "💊 Administer first-line medications"
            ],
            "HIGH": [
                "⚠️  Schedule urgent appointment within 24 hours",
                "🧪 Order blood panel and cultures",
                "💊 Prescribe symptomatic relief",
                "📋 Monitor vitals every 2 hours"
            ],
            "MEDIUM": [
                "📅 Schedule appointment within 3 days",
                "💊 Over-the-counter treatment advised",
                "🌡️  Monitor temperature twice daily",
                "💧 Increase fluid intake"
            ],
            "LOW": [
                "🏠 Home rest recommended",
                "💧 Stay hydrated",
                "📱 Follow up if symptoms worsen",
                "📋 General wellness monitoring"
            ]
        }
        return base.get(urgency, base["LOW"])

    def _decide_next_action(self, urgency):
        actions = {
            "CRITICAL": "EMERGENCY_REFERRAL",
            "HIGH":     "URGENT_APPOINTMENT",
            "MEDIUM":   "SCHEDULE_FOLLOWUP",
            "LOW":      "MONITOR_AT_HOME"
        }
        return actions.get(urgency, "MONITOR_AT_HOME")

    def _log(self, message):
        entry = f"[{self.state.value}] {message}"
        self.memory.action_log.append(entry)
        print(entry)

    def print_log(self):
        print("\n📋 Agent Action Log:")
        print("─" * 50)
        for entry in self.memory.action_log:
            print(f"  {entry}")

    def get_performance(self):
        return {
            'total_patients':    len(self.memory.patient_history),
            'performance_score': self.performance_score,
            'diagnoses_made':    len(self.memory.diagnosis_history)
        }


# ============================================================
# STANDALONE TEST
# Per the manual's "Pro Tip": run each module independently
# first using its own test code, before wiring into app.py.
#
# This matches the manual's own test setup and expected output
# exactly:
#     [collecting_symptoms] Perceived patient P001 with 2 symptoms
#     Agent test passed!
# ============================================================

if __name__ == "__main__":
    # This represents one patient's data (per the manual's own example)
    patient = PatientPercept(
        patient_id="P001",
        symptoms=["fever", "cough"],
        age=34,
        temperature=38.9,     # Celsius
        heart_rate=98,        # BPM
        blood_pressure="120/80"
    )

    agent = HealthcareDiagnosticAgent()
    agent.perceive(patient)
    print("Agent test passed!")

    # ------------------------------------------------------------------
    # OPTIONAL: uncomment to test the full perceive -> think -> act cycle
    # using a fake module, before Modules 2-7 are wired in for real.
    # ------------------------------------------------------------------
    # class FakeModule:
    #     """Stands in for a real specialist module during early testing."""
    #     def analyze(self, patient):
    #         return {
    #             'diagnosis': 'test_condition',
    #             'confidence': 0.5,
    #             'summary': 'test result'
    #         }
    #
    # agent.register_module('FakeModule', FakeModule())
    # report = agent.run(patient)
    # print(report)