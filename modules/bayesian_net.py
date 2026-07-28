# ============================================================
# MODULE 3: Bayesian Network — Probabilistic Diagnosis
# Covers: Week 7 (Bayesian Networks)
#
# NOTE ON THIS REVISION:
# The original version of this file used a 7-disease list (flu,
# covid19, dengue, cardiac, diabetes, common_cold, healthy) that
# doesn't match the 15-disease/26-symptom list locked in
# data/diseases.csv and data/symptoms.csv (which Module 2's rules
# were also aligned to). This revision keeps the original structure,
# math, and extra methods (explain(), ranked_diagnoses) completely
# intact -- only the priors/likelihoods were replaced so this module
# stays consistent with the rest of the team's work.
#
# Priors are now loaded directly from data/diseases.csv rather than
# hardcoded, so they can never drift out of sync with the disease
# list again.
#
# LIKELIHOOD VALUES ARE ILLUSTRATIVE/SYNTHETIC, chosen for reasonable
# medical plausibility, not sourced from real clinical data --
# consistent with how Module 4/5's synthetic training data works too.
# ============================================================

import csv
import os
import numpy as np
from typing import Dict, List


class SimpleBayesianDiagnostics:
    """
    Simplified Bayesian diagnostic model using
    pre-computed conditional probabilities.
    """

    def __init__(self, diseases_csv: str = "data/diseases.csv"):
        # Prior probabilities P(Disease) -- loaded from data/diseases.csv
        # so this always matches the same 15-disease list every other
        # module (Agent, Knowledge Base, ML, NN) is built around.
        self.priors = self._load_priors(diseases_csv)

        # Likelihoods: P(Symptom | Disease)
        # Format: disease -> {symptom -> P(symptom|disease)}
        self.likelihoods = {
            'flu': {
                'fever': 0.90, 'cough': 0.85, 'fatigue': 0.88,
                'body_aches': 0.80, 'chills': 0.60,
                'sore_throat': 0.50, 'headache': 0.55,
            },
            'covid19': {
                'fever': 0.85, 'cough': 0.80, 'fatigue': 0.75,
                'loss_of_smell': 0.65, 'sore_throat': 0.45,
                'shortness_of_breath': 0.40, 'headache': 0.50,
            },
            'common_cold': {
                'runny_nose': 0.90, 'sore_throat': 0.75, 'cough': 0.65,
                'fatigue': 0.40, 'fever': 0.25, 'headache': 0.30,
            },
            'dengue': {
                'fever': 0.98, 'rash': 0.75, 'joint_pain': 0.85,
                'headache': 0.70, 'nausea': 0.55, 'vomiting': 0.40,
            },
            'malaria': {
                'fever': 0.95, 'chills': 0.85, 'headache': 0.70,
                'fatigue': 0.65, 'nausea': 0.50, 'vomiting': 0.35,
            },
            'typhoid': {
                'fever': 0.90, 'abdominal_pain': 0.70,
                'loss_of_appetite': 0.75, 'headache': 0.55,
                'fatigue': 0.60, 'diarrhea': 0.40,
            },
            'pneumonia': {
                'cough': 0.90, 'fever': 0.80, 'chest_pain': 0.65,
                'shortness_of_breath': 0.75, 'fatigue': 0.55,
                'wheezing': 0.40,
            },
            'migraine': {
                'headache': 0.95, 'dizziness': 0.55, 'nausea': 0.60,
                'fatigue': 0.30,
            },
            'tuberculosis': {
                'cough': 0.85, 'weight_loss': 0.75, 'night_sweats': 0.70,
                'fatigue': 0.65, 'fever': 0.50, 'chest_pain': 0.40,
            },
            'hepatitis_a': {
                'jaundice': 0.80, 'fatigue': 0.70, 'nausea': 0.65,
                'loss_of_appetite': 0.60, 'abdominal_pain': 0.45,
                'vomiting': 0.35,
            },
            'urinary_tract_infection': {
                'painful_urination': 0.90, 'abdominal_pain': 0.55,
                'fever': 0.30, 'fatigue': 0.25,
            },
            'sinusitis': {
                'facial_pain': 0.85, 'headache': 0.70,
                'runny_nose': 0.65, 'fever': 0.20,
            },
            'gastroenteritis': {
                'vomiting': 0.80, 'diarrhea': 0.85, 'abdominal_pain': 0.75,
                'nausea': 0.70, 'fever': 0.30,
            },
            'bronchitis': {
                'cough': 0.90, 'wheezing': 0.60, 'chest_pain': 0.45,
                'fatigue': 0.40, 'fever': 0.25,
            },
            'tonsillitis': {
                'sore_throat': 0.90, 'swollen_tonsils': 0.85,
                'fever': 0.55, 'fatigue': 0.30,
            },
        }

    # -----------------------------------------------------------------
    def _load_priors(self, diseases_csv: str) -> Dict[str, float]:
        """Load priors from data/diseases.csv, with a hardcoded fallback
        (matching the CSV exactly) so this file can still run standalone
        if the CSV path isn't found."""
        fallback = {
            'flu': 0.14, 'covid19': 0.10, 'common_cold': 0.16,
            'dengue': 0.07, 'malaria': 0.09, 'typhoid': 0.06,
            'pneumonia': 0.07, 'migraine': 0.05, 'tuberculosis': 0.04,
            'hepatitis_a': 0.03, 'urinary_tract_infection': 0.06,
            'sinusitis': 0.05, 'gastroenteritis': 0.04,
            'bronchitis': 0.03, 'tonsillitis': 0.01,
        }
        if not os.path.exists(diseases_csv):
            return fallback

        priors = {}
        with open(diseases_csv) as f:
            for row in csv.DictReader(f):
                priors[row['disease_name']] = float(row['prior_probability'])
        return priors if priors else fallback

    # -----------------------------------------------------------------
    def compute_posterior(self, symptoms: List[str]) -> Dict[str, float]:
        """
        Naive Bayes posterior:
        P(D|S1,...,Sn) proportional to P(D) x product of P(Si|D)
        """
        posteriors = {}
        symptoms_clean = [s.lower().replace(' ', '_') for s in symptoms]

        for disease, prior in self.priors.items():
            log_prob = np.log(prior)
            disease_likelihoods = self.likelihoods.get(disease, {})
            for symptom in symptoms_clean:
                p_s_given_d = disease_likelihoods.get(symptom, 0.01)
                log_prob += np.log(p_s_given_d)
            posteriors[disease] = log_prob

        # Convert log-probabilities to probabilities
        max_log = max(posteriors.values())
        exp_probs = {d: np.exp(v - max_log)
                     for d, v in posteriors.items()}
        total = sum(exp_probs.values())
        return {d: round(v / total, 4) for d, v in exp_probs.items()}

    # -----------------------------------------------------------------
    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        posteriors = self.compute_posterior(percept.symptoms)
        top_disease = max(posteriors, key=posteriors.get)
        top_prob = posteriors[top_disease]
        sorted_dx = sorted(posteriors.items(),
                            key=lambda x: x[1], reverse=True)

        return {
            'summary': f"Top: {top_disease} ({top_prob:.2%})",
            'diagnosis': top_disease,
            'confidence': top_prob,
            'all_posteriors': posteriors,
            'ranked_diagnoses': sorted_dx[:5]
        }

    # -----------------------------------------------------------------
    def explain(self, disease: str, symptoms: List[str]) -> str:
        symptoms_clean = [s.lower().replace(' ', '_') for s in symptoms]
        likelihoods = self.likelihoods.get(disease, {})
        evidence = [
            f"P({s}|{disease})={likelihoods.get(s, 0.01):.2f}"
            for s in symptoms_clean
        ]
        return f"P({disease}) = {self.priors[disease]} x " + \
               " x ".join(evidence)


# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == "__main__":
    bn = SimpleBayesianDiagnostics()
    posteriors = bn.compute_posterior(
        ["fever", "cough", "loss_of_smell", "fatigue"]
    )

    print("Top 3 Diagnoses:")
    ranked = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
    for disease, prob in ranked[:3]:
        print(f"  {disease:<25}: {prob:.2%}")

    print()
    print(bn.explain("covid19", ["fever", "cough", "loss_of_smell", "fatigue"]))