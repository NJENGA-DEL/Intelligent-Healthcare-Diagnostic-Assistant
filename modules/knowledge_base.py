# ============================================================
# MODULE 2: FOL Knowledge Base + Inference Engine
# Covers: Week 5 (First-Order Logic & Inference)
#
# NOTE ON THIS REVISION:
# The rule set below was rewritten to use ONLY the 15 diseases and
# 26 symptoms locked in data/diseases.csv and data/symptoms.csv, so
# this module stays consistent with Modules 3, 4, and 5 -- if each
# module reasons about a different disease list, the Agent's majority
# vote in _aggregate_diagnosis() would never actually get modules
# agreeing with each other. All core logic (forward chaining, backward
# chaining, certainty factor combination) is unchanged from the
# original.
# ============================================================

from typing import Set, List, Dict, Tuple, Optional

class MedicalKnowledgeBase:
    """
    First-Order Logic based medical knowledge base.
    Supports forward chaining, backward chaining,
    and confidence-weighted inference.
    """

    def __init__(self):
        self.facts:  Set[str]             = set()
        self.rules:  List[Tuple]          = []
        self.certainty_factors: Dict[str, float] = {}
        self._load_medical_knowledge()

    def _load_medical_knowledge(self):
        """Load domain medical knowledge"""
        # ── Symptom Facts (loaded dynamically per patient) ──
        # ── Disease Rules ──
        # Level 1: symptom combinations -> "<disease>_suspected"
        # One rule per disease in our locked 15-disease list, using
        # only symptom names present in data/symptoms.csv.
        disease_rules = [
            # (conditions,                              conclusion,                    certainty)
            (["fever", "cough", "fatigue"],
             "flu_suspected",                                                          0.75),
            (["fever", "cough", "loss_of_smell", "fatigue"],
             "covid19_suspected",                                                      0.85),
            (["runny_nose", "sore_throat", "cough"],
             "common_cold_suspected",                                                  0.70),
            (["fever", "rash", "joint_pain"],
             "dengue_suspected",                                                       0.80),
            (["fever", "chills", "headache"],
             "malaria_suspected",                                                      0.78),
            (["fever", "abdominal_pain", "loss_of_appetite"],
             "typhoid_suspected",                                                      0.72),
            (["cough", "fever", "chest_pain"],
             "pneumonia_suspected",                                                    0.80),
            (["headache", "dizziness", "nausea"],
             "migraine_suspected",                                                     0.70),
            (["cough", "weight_loss", "night_sweats", "fatigue"],
             "tuberculosis_suspected",                                                 0.82),
            (["jaundice", "fatigue", "nausea"],
             "hepatitis_a_suspected",                                                  0.80),
            (["painful_urination", "abdominal_pain"],
             "urinary_tract_infection_suspected",                                      0.75),
            (["facial_pain", "headache", "runny_nose"],
             "sinusitis_suspected",                                                    0.72),
            (["vomiting", "diarrhea", "abdominal_pain"],
             "gastroenteritis_suspected",                                              0.78),
            (["cough", "wheezing", "chest_pain"],
             "bronchitis_suspected",                                                   0.74),
            (["sore_throat", "swollen_tonsils", "fever"],
             "tonsillitis_suspected",                                                  0.76),

            # Level 2: "suspected" + additional corroborating evidence
            # -> "confirmed". Demonstrates multi-step forward chaining
            # (a conclusion from one rule becomes a condition for another).
            (["flu_suspected", "chills"],
             "flu_confirmed",                                                          0.85),
            (["covid19_suspected", "shortness_of_breath"],
             "covid19_confirmed",                                                      0.90),
            (["pneumonia_suspected", "shortness_of_breath"],
             "pneumonia_confirmed",                                                    0.90),
            (["dengue_suspected", "vomiting"],
             "dengue_confirmed",                                                       0.85),
            (["tuberculosis_suspected", "chest_pain"],
             "tuberculosis_confirmed",                                                 0.85),

            # Urgency rules -- map confirmed diagnoses to an action tier.
            (["pneumonia_confirmed"],
             "EMERGENCY",                                                              0.95),
            (["dengue_confirmed"],
             "EMERGENCY",                                                              0.90),
            (["covid19_confirmed"],
             "ISOLATE_AND_TREAT",                                                      0.95),
            (["tuberculosis_confirmed"],
             "ISOLATE_AND_TREAT",                                                      0.88),
            (["flu_confirmed"],
             "REST_AND_MEDICATE",                                                      0.85),
        ]
        for conditions, conclusion, cf in disease_rules:
            self.add_rule(conditions, conclusion, cf)

    def add_fact(self, fact: str, certainty: float = 1.0):
        self.facts.add(fact)
        self.certainty_factors[fact] = certainty

    def add_rule(self, conditions: List[str],
                 conclusion: str, certainty: float = 1.0):
        self.rules.append((conditions, conclusion, certainty))

    def load_patient_symptoms(self, symptoms: List[str]):
        """Load patient symptoms as facts"""
        for symptom in symptoms:
            self.add_fact(symptom.lower().replace(' ', '_'))

    def forward_chain(self, verbose: bool = False) -> Dict[str, float]:
        """Forward chaining with certainty factors"""
        inferred = {}
        changed  = True
        iteration = 0

        while changed:
            changed   = False
            iteration += 1
            for conditions, conclusion, rule_cf in self.rules:
                all_known = all(
                    c in self.facts or c in inferred for c in conditions
                )
                if all_known and conclusion not in inferred:
                    # Combine certainty factors
                    cond_cfs = [
                        self.certainty_factors.get(c,
                            inferred.get(c, 1.0))
                        for c in conditions
                    ]
                    combined_cf = rule_cf * min(cond_cfs)
                    inferred[conclusion] = round(combined_cf, 4)

                    if verbose:
                        cond_str = " ∧ ".join(conditions)
                        print(f"  Iter {iteration}: "
                              f"{cond_str} → {conclusion} "
                              f"(CF={combined_cf:.3f})")
                    changed = True
        return inferred

    def backward_chain(self, goal: str,
                       visited: Optional[Set] = None,
                       depth: int = 0) -> Tuple[bool, float]:
        """Backward chaining — prove a goal"""
        indent  = "  " * depth
        visited = visited or set()

        if goal in self.facts:
            return True, self.certainty_factors.get(goal, 1.0)
        if goal in visited:
            return False, 0.0
        visited.add(goal)

        for conditions, conclusion, rule_cf in self.rules:
            if conclusion == goal:
                results = [
                    self.backward_chain(c, visited.copy(), depth+1)
                    for c in conditions
                ]
                if all(proved for proved, _ in results):
                    cf = rule_cf * min(cf for _, cf in results)
                    return True, round(cf, 4)
        return False, 0.0

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        self.facts = set()
        self.certainty_factors = {}
        self.load_patient_symptoms(percept.symptoms)

        # Add vitals as facts. These aren't required by any rule above
        # (the rules only use symptom-derived facts, kept consistent with
        # symptoms.csv), but are kept here as extra contextual evidence
        # a future rule could use.
        if percept.temperature > 38.0:
            self.add_fact("fever",
                min(1.0, (percept.temperature - 37.0) / 3.0))
        if percept.temperature > 39.5:
            self.add_fact("high_fever", 1.0)
        if percept.heart_rate > 100:
            self.add_fact("tachycardia", 1.0)

        inferred = self.forward_chain()
        diseases  = {k: v for k, v in inferred.items()
                     if 'suspected' in k or 'confirmed' in k}

        top = max(diseases, key=diseases.get) if diseases else "Unknown"
        return {
            'summary':    f"Inferred {len(inferred)} conclusions",
            'diagnosis':  top,
            'confidence': diseases.get(top, 0.5),
            'all_inferred': inferred
        }

    def get_explanation(self, diagnosis: str) -> str:
        """Explain how a diagnosis was reached"""
        for conditions, conclusion, cf in self.rules:
            if conclusion == diagnosis:
                return (f"'{diagnosis}' derived from: "
                        f"{' + '.join(conditions)} (CF={cf})")
        return f"'{diagnosis}' is a base fact"


# ============================================================
# STANDALONE TEST
# Matches the manual's own test example exactly.
# ============================================================
if __name__ == "__main__":
    kb = MedicalKnowledgeBase()
    kb.add_fact("fever")
    kb.add_fact("cough")
    kb.add_fact("loss_of_smell")
    kb.add_fact("fatigue")

    results = kb.forward_chain(verbose=True)
    print("Inferred:", results)

    # Test backward chaining
    proved, cf = kb.backward_chain("covid19_suspected")
    print(f"COVID-19 suspected: {proved}, Confidence: {cf}")