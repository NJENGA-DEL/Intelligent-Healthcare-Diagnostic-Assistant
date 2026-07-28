# ============================================================
# CAPSTONE MAIN APPLICATION
# Intelligent Healthcare Diagnostic Assistant
# Introduction to AI — 13-Week Capstone
# ============================================================
#
# INTEGRATION NOTES (read before modifying):
#
# Only 4 modules are registered with the Agent via
# agent.register_module(): KnowledgeBase, BayesianNet, MLClassifier,
# NeuralNetwork. These 4 all share the same contract -- given a
# patient, predict a DISEASE with a confidence score -- which is what
# the Agent's think()/act() majority-vote aggregation is built for.
#
# FuzzySeverityAssessor and TreatmentPlanner are NOT registered the
# same way, for reasons specific to each (documented inline in
# process_patient() below):
#   - Fuzzy answers a different question ("how severe?" not "what
#     disease?") -- mixing its output into the disease majority vote
#     would corrupt it.
#   - Planner needs the AGGREGATED diagnosis + urgency as input, which
#     doesn't exist yet at the point think() runs all modules in
#     parallel -- it has to run strictly AFTER agent.act().
# ============================================================

import sys
import csv
import os
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from modules.agent            import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base   import MedicalKnowledgeBase
from modules.bayesian_net     import SimpleBayesianDiagnostics
from modules.ml_classifier    import MLDiagnosticClassifier
from modules.neural_network   import NeuralDiagnosticModel
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner          import TreatmentPlanner
from modules.nlp_processor    import extract_symptoms_from_text
from modules.search           import find_similar_patients
from modules.rl_agent         import train as train_rl_agent, SEVERITY_STATES, TRIAGE_ACTIONS


# ── ANSI Colors ────────────────────────────────────────────
class C:
    HEADER = '\033[95m'; BLUE   = '\033[94m'
    GREEN  = '\033[92m'; YELLOW = '\033[93m'
    RED    = '\033[91m'; BOLD   = '\033[1m'
    END    = '\033[0m'


def banner():
    print(f"""
{C.BOLD}{C.BLUE}
==============================================================
        INTELLIGENT HEALTHCARE DIAGNOSTIC AI
         Introduction to AI -- Capstone Project
  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy | Plan
==============================================================
{C.END}""")


def section(title: str):
    print(f"\n{C.BOLD}{C.YELLOW}{'='*60}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}{'='*60}{C.END}")


# ------------------------------------------------------------
def build_system():
    """
    Instantiate and wire all AI modules.

    Returns (agent, fuzzy_assessor, planner, rl_agent) -- the agent has
    the 4 disease-diagnosing modules registered; fuzzy_assessor, planner,
    and rl_agent are returned separately since they're called explicitly
    in process_patient() rather than through agent.think().
    """
    section("Building AI System — Registering Modules")

    agent = HealthcareDiagnosticAgent()

    print("\n  Initializing modules...")
    kb  = MedicalKnowledgeBase()
    bn  = SimpleBayesianDiagnostics()
    ml  = MLDiagnosticClassifier()
    nn  = NeuralDiagnosticModel()

    # Train the modules that need training. NN epochs reduced to 30
    # (per the manual's own suggestion, "reduce epochs to 30 for
    # testing") to keep startup time reasonable during development/demo.
    print("\n  Training ML Classifier...")
    ml.train(verbose=False)
    print("  Training Neural Network...")
    nn.train(epochs=30, verbose=0)

    agent.register_module('KnowledgeBase',   kb)
    agent.register_module('BayesianNet',     bn)
    agent.register_module('MLClassifier',    ml)
    agent.register_module('NeuralNetwork',   nn)

    fuzzy_assessor = FuzzySeverityAssessor()
    planner = TreatmentPlanner()

    print("  Training RL triage agent...")
    rl_agent, _ = train_rl_agent(episodes=2000, verbose=False)

    print(f"\n{C.GREEN}  System ready: 4 diagnostic modules + "
          f"Fuzzy Severity + Planner + RL Agent{C.END}")

    return agent, fuzzy_assessor, planner, rl_agent


# ------------------------------------------------------------
def process_patient(agent: HealthcareDiagnosticAgent,
                     fuzzy_assessor: FuzzySeverityAssessor,
                     planner: TreatmentPlanner,
                     rl_agent,
                     patient: PatientPercept,
                     patient_history: list) -> dict:
    """
    Run one patient through the FULL pipeline:

        perceive -> think -> act  (4 registered diagnostic modules,
                                    aggregated into one diagnosis)
              |
              +--> Fuzzy Severity  (runs independently, on vitals only)
              |
              +--> RL Agent        (learned severity->action policy,
                                     shown alongside the Agent's own
                                     hardcoded _decide_next_action())
              |
              +--> Case-Based Search (similarity check against
                                       previously-processed patients
                                       in this run)
              |
              +--> Planner         (runs AFTER, using the aggregated
                                     diagnosis + urgency as input)

    Each module call is wrapped so that ONE module failing doesn't
    prevent the rest of the report from being generated -- consistent
    with agent.py's own per-module error isolation in think().

    `patient_history` is a running list this function APPENDS to after
    each patient, so later patients in the same run can be matched
    against earlier ones via search.find_similar_patients().
    """
    # --- Step 1-3: perceive -> think -> act (the 4 registered modules) ---
    agent.perceive(patient)
    diagnosis_results = agent.think()
    action_report = agent.act(diagnosis_results)

    # --- Fuzzy severity assessment (vitals-only, independent of diagnosis) ---
    try:
        severity = fuzzy_assessor.assess(
            patient.temperature, patient.heart_rate, len(patient.symptoms)
        )
    except Exception as e:
        severity = {'severity_score': None, 'severity_label': 'UNKNOWN',
                    'error': str(e)}

    # --- RL Agent: learned severity -> action policy, for comparison
    #     against agent.py's own hardcoded _decide_next_action() mapping ---
    try:
        urgency = action_report['urgency']
        state_idx = SEVERITY_STATES.index(urgency) if urgency in SEVERITY_STATES else None
        rl_action = TRIAGE_ACTIONS[rl_agent.choose_action(state_idx, greedy=True)] \
            if state_idx is not None else None
    except Exception as e:
        rl_action = None

    # --- Case-based search: how similar is this patient to earlier
    #     ones processed in this same run? ---
    try:
        similar = find_similar_patients(patient.symptoms, patient_history, top_k=2)
        similar_cases = [
            {'patient_id': rec['patient_id'], 'diagnosis': rec['diagnosis'],
             'similarity': round(score, 2)}
            for rec, score in similar if score > 0
        ]
    except Exception:
        similar_cases = []

    # --- Treatment plan (needs the AGGREGATED diagnosis + urgency,
    #     which only exist now, after act() has run) ---
    try:
        treatment_plan = planner.create_treatment_plan(
            action_report['diagnosis'], action_report['urgency']
        )
    except Exception as e:
        treatment_plan = {'error': str(e), 'plan': []}

    report = {
        'patient_id': patient.patient_id,
        'symptoms': patient.symptoms,
        'excluded_symptoms': action_report.get('excluded_symptoms', []),
        'diagnosis': action_report['diagnosis'],
        'confidence': action_report['confidence'],
        'urgency': action_report['urgency'],
        'recommendations': action_report['recommendations'],
        'next_action': action_report['next_action'],
        'rl_recommended_action': rl_action,
        'severity_score': severity.get('severity_score'),
        'severity_label': severity.get('severity_label'),
        'similar_cases': similar_cases,
        'treatment_plan': treatment_plan.get('plan', []),
        'module_results': diagnosis_results,
    }

    # Add this patient to the running history so FUTURE patients in
    # this run can be matched against them.
    patient_history.append({
        'patient_id': patient.patient_id,
        'symptoms': patient.symptoms,
        'diagnosis': action_report['diagnosis'],
    })

    return report


# ------------------------------------------------------------
def load_patient_records(csv_path: str = "data/patient_records.csv") -> list:
    """
    Load existing patient records (both 'seed' and prior 'runtime' rows)
    at startup, so search.find_similar_patients() has the FULL
    accumulated history to match against -- not just patients processed
    in the current run.
    """
    if not os.path.exists(csv_path):
        return []
    records = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            records.append({
                "patient_id": row["patient_id"],
                "symptoms": row["symptoms"].split(";"),
                "diagnosis": row["diagnosis"],
            })
    return records


# ------------------------------------------------------------
def append_patient_record(patient: PatientPercept, report: dict,
                           csv_path: str = "data/patient_records.csv"):
    """
    Auto-append this patient + the AI's own prediction to
    data/patient_records.csv, tagged source='runtime'.

    IMPORTANT: these rows are the AI's own (unverified) predictions,
    NOT ground truth -- unlike the hand-curated 'seed' rows already in
    this file. evaluation/metrics.py should filter to source=='seed'
    only when computing accuracy/precision/recall against known-correct
    labels; mixing in unverified runtime predictions as if they were
    ground truth would make the model look artificially accurate
    (it would just be agreeing with itself).

    Runtime rows ARE useful for search.py's case-based similarity
    search though, since more accumulated cases (even unverified ones)
    make future similarity matches richer over time.
    """
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["patient_id", "symptoms", "age", "temperature",
                              "heart_rate", "blood_pressure", "diagnosis",
                              "confidence", "source"])
        writer.writerow([
            patient.patient_id,
            ";".join(patient.symptoms),
            patient.age,
            patient.temperature,
            patient.heart_rate,
            patient.blood_pressure,
            report["diagnosis"],
            report["confidence"],
            "runtime",
        ])


# ------------------------------------------------------------
def print_patient_report(report: dict):
    """Pretty-print one patient's full diagnostic report to the terminal."""
    urgency_colors = {
        'CRITICAL': C.RED, 'HIGH': C.YELLOW,
        'MEDIUM': C.BLUE, 'LOW': C.GREEN
    }
    uc = urgency_colors.get(report['urgency'], C.END)

    print(f"\n{C.BOLD}Patient {report['patient_id']}{C.END}")
    print(f"  Symptoms      : {', '.join(report['symptoms'])}")
    if report.get('excluded_symptoms'):
        print(f"  {C.YELLOW}Note: unrecognized symptom(s) ignored: "
              f"{', '.join(report['excluded_symptoms'])}{C.END}")
    print(f"  Diagnosis     : {C.BOLD}{report['diagnosis']}{C.END} "
          f"(confidence: {report['confidence']:.1%})")
    print(f"  Urgency       : {uc}{C.BOLD}{report['urgency']}{C.END}")
    if report['severity_score'] is not None:
        print(f"  Severity      : {report['severity_label']} "
              f"({report['severity_score']}/100)")
    print(f"  Next Action   : {report['next_action']}")
    if report.get('rl_recommended_action'):
        match = "matches" if report['rl_recommended_action'] == report['next_action'] else "differs from"
        print(f"  RL Agent Rec. : {report['rl_recommended_action']} "
              f"({match} the agent's own rule-based action)")
    if report.get('similar_cases'):
        print(f"  Similar Cases : " + "; ".join(
            f"{c['patient_id']} ({c['diagnosis']}, sim={c['similarity']:.0%})"
            for c in report['similar_cases']
        ))
    print(f"  Recommendations:")
    for rec in report['recommendations']:
        print(f"    - {rec}")
    if report['treatment_plan']:
        print(f"  Treatment Plan ({len(report['treatment_plan'])} steps):")
        for step in report['treatment_plan']:
            print(f"    {step['step']}. {step['action']} [{step['duration']}]")


# ------------------------------------------------------------
def build_nlp_patient() -> PatientPercept:
    """
    Demonstrates the NLP pipeline: a patient described in raw free
    text (as if typed into a chat/intake form), rather than a
    pre-built symptom list -- symptoms are extracted automatically via
    nlp_processor.extract_symptoms_from_text().

    NOTE: this used to be called eagerly during all_patients list
    construction (all_patients = get_test_patients() +
    [get_nlp_test_patient()]), which meant its print statements fired
    BEFORE the loop even started -- making it look, misleadingly, like
    the NLP output belonged to whichever patient was processed first
    (P001), when it actually belonged to a different, unrelated patient
    processed last. Fixed by building this patient INSIDE the loop,
    right when it's actually processed, so the NLP print output appears
    directly next to that patient's own report.
    """
    raw_complaint = (
        "I've had a really high temperature and a bad cough for the "
        "past two days. My whole body aches and I keep getting chills. "
        "No rash though."
    )
    symptoms = extract_symptoms_from_text(raw_complaint)
    print(f"\n  {C.BLUE}[NLP] Raw complaint:{C.END} \"{raw_complaint}\"")
    print(f"  {C.BLUE}[NLP] Extracted symptoms:{C.END} {symptoms}")

    return PatientPercept(
        patient_id="P007-NLP",
        symptoms=symptoms,
        age=38, temperature=39.1, heart_rate=100,
        blood_pressure="122/80"
    )


def get_test_patients():
    """
    At least 5 test patients, per the manual's final deliverables
    checklist. Deliberately varied: a clear-cut case, an ambiguous
    case, a critical case, a non-infectious case (migraine), and a
    contagious case -- to exercise different paths through the system.
    """
    return [
        PatientPercept(
            patient_id="P001",
            symptoms=["fever", "cough", "loss_of_smell", "fatigue"],
            age=34, temperature=38.9, heart_rate=98,
            blood_pressure="120/80"
        ),
        PatientPercept(
            patient_id="P002",
            symptoms=["headache", "dizziness", "nausea"],
            age=29, temperature=37.1, heart_rate=76,
            blood_pressure="118/76"
        ),
        PatientPercept(
            patient_id="P003",
            symptoms=["fever", "rash", "joint_pain", "vomiting"],
            age=41, temperature=39.8, heart_rate=115,
            blood_pressure="100/65"
        ),
        PatientPercept(
            patient_id="P004",
            symptoms=["cough", "weight_loss", "night_sweats", "fatigue",
                      "chest_pain"],
            age=55, temperature=38.2, heart_rate=105,
            blood_pressure="110/70"
        ),
        PatientPercept(
            patient_id="P005",
            symptoms=["painful_urination", "abdominal_pain"],
            age=27, temperature=37.4, heart_rate=82,
            blood_pressure="115/75"
        ),
        PatientPercept(
            patient_id="P006",
            symptoms=["sore_throat", "swollen_tonsils", "fever"],
            age=19, temperature=38.3, heart_rate=90,
            blood_pressure="112/72"
        ),
    ]


# ------------------------------------------------------------
def get_float_input(prompt: str, min_val: float = None, max_val: float = None) -> float:
    """Prompt for a float, re-prompting on invalid or out-of-range input.
    A real person typing live input WILL make mistakes -- this must not
    crash the whole session over one bad keystroke."""
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
            if min_val is not None and value < min_val:
                print(f"  Please enter a value >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"  Please enter a value <= {max_val}")
                continue
            return value
        except ValueError:
            print(f"  '{raw}' isn't a valid number, try again.")


def get_int_input(prompt: str, min_val: int = None, max_val: int = None) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
            if min_val is not None and value < min_val:
                print(f"  Please enter a value >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"  Please enter a value <= {max_val}")
                continue
            return value
        except ValueError:
            print(f"  '{raw}' isn't a valid whole number, try again.")


# ------------------------------------------------------------
def run_interactive_session(agent, fuzzy_assessor, planner, rl_agent,
                             patient_history: list) -> list:
    """
    Real interactive mode: a live person describes their own symptoms
    and vitals, typed at the keyboard, run through the FULL pipeline
    (NLP extraction -> diagnosis -> severity -> RL comparison ->
    similarity search -> treatment plan) -- same pipeline as the
    hardcoded test patients, just with real input instead of fixed data.

    Returns the list of reports generated, so main()'s summary count
    reflects patients processed here too.
    """
    section("Interactive Patient Intake")
    patient_counter = 1
    session_reports = []

    while True:
        patient_id = f"LIVE-{patient_counter:03d}"
        print(f"\n{C.BOLD}New patient ({patient_id}){C.END}")

        raw_complaint = input(
            "Describe your symptoms (a sentence, or a comma-separated "
            "list): "
        ).strip()

        if not raw_complaint:
            print("  No symptoms entered -- skipping this patient.")
        else:
            symptoms = extract_symptoms_from_text(raw_complaint)
            print(f"  {C.BLUE}Recognized symptoms:{C.END} "
                  f"{symptoms if symptoms else '(none recognized)'}")

            if not symptoms:
                print(f"  {C.YELLOW}None of the described symptoms were "
                      f"recognized -- diagnosis would be unreliable. "
                      f"Skipping.{C.END}")
            else:
                age = get_int_input("Age: ", min_val=0, max_val=120)
                temperature = get_float_input("Temperature (Celsius): ",
                                               min_val=30.0, max_val=45.0)
                heart_rate = get_int_input("Heart rate (BPM): ",
                                            min_val=30, max_val=250)
                blood_pressure = input("Blood pressure (e.g. 120/80): ").strip() or "N/A"

                patient = PatientPercept(
                    patient_id=patient_id,
                    symptoms=symptoms,
                    age=age,
                    temperature=temperature,
                    heart_rate=heart_rate,
                    blood_pressure=blood_pressure,
                )

                try:
                    report = process_patient(agent, fuzzy_assessor, planner,
                                              rl_agent, patient, patient_history)
                    print_patient_report(report)
                    append_patient_record(patient, report)
                    session_reports.append(report)
                except Exception as e:
                    print(f"{C.RED}  ERROR processing {patient_id}: {e}{C.END}")

        patient_counter += 1
        again = input("\nDiagnose another patient? (y/n): ").strip().lower()
        if again != "y":
            break

    return session_reports


# ------------------------------------------------------------
def main():
    banner()
    agent, fuzzy_assessor, planner, rl_agent = build_system()

    patient_history = load_patient_records()   # seeded with prior runs +
                                                 # hand-curated seed data
    print(f"\n  Loaded {len(patient_history)} existing patient records "
          f"for case-based similarity search")

    print(f"\n{C.BOLD}Choose a mode:{C.END}")
    print("  1. Run the built-in test patients ")
    print("  2. Interactive mode ")
    print("  3. Both")
    choice = input("Enter 1, 2, or 3: ").strip()

    reports = []

    if choice in ("1", "3"):
        section("Running Test Patients")
        all_patients = get_test_patients()
        # A marker (None) signals "build the NLP patient here" -- deferring
        # its construction (and print output) until this exact point in
        # the loop, instead of eagerly beforehand.
        all_patients.append(None)

        for patient_or_marker in all_patients:
            patient = build_nlp_patient() if patient_or_marker is None else patient_or_marker
            try:
                report = process_patient(agent, fuzzy_assessor, planner,
                                          rl_agent, patient, patient_history)
                reports.append(report)
                print_patient_report(report)
                append_patient_record(patient, report)
            except Exception as e:
                print(f"{C.RED}  ERROR processing {patient.patient_id}: {e}{C.END}")

    if choice in ("2", "3"):
        interactive_reports = run_interactive_session(
            agent, fuzzy_assessor, planner, rl_agent, patient_history
        )
        reports.extend(interactive_reports)

    section("Run Summary")
    print(f"  Total patients processed: {len(reports)}")
    performance = agent.get_performance()
    print(f"  Agent performance score : {performance['performance_score']}")
    print(f"  Diagnoses made           : {performance['diagnoses_made']}")

    return reports


if __name__ == "__main__":
    main()