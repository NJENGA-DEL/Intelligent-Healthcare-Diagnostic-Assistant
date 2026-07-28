# ============================================================
# EVALUATION MODULE: metrics.py
#
# Evaluates each diagnostic module (KnowledgeBase, BayesianNet,
# MLClassifier, NeuralNetwork) against the hand-curated GROUND TRUTH
# patients in data/patient_records.csv.
#
# CRITICAL: only rows with source=='seed' are used here. The
# 'runtime' rows (auto-logged by app.py) are the AI's own unverified
# predictions -- using them as ground truth would let the system
# grade itself against its own guesses, making accuracy artificially
# look perfect regardless of whether the underlying modules are any
# good. This is enforced in load_seed_patients() below.
#
# HONEST LIMITATION: the seed set has only 15 patients (one per
# disease), since that's what was hand-curated for this project. This
# is a small evaluation set -- metrics here are illustrative of the
# evaluation PIPELINE working correctly, not a statistically robust
# accuracy claim. Expanding data/patient_records.csv with more
# labeled examples per disease would make these numbers more
# trustworthy.
# ============================================================

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize


# ------------------------------------------------------------
def load_seed_patients(csv_path: str = "data/patient_records.csv"):
    """
    Load ONLY source=='seed' rows as (PatientPercept, true_diagnosis)
    pairs. Imported lazily to avoid a hard dependency at module import
    time if this file is used standalone.
    """
    from modules.agent import PatientPercept

    if not os.path.exists(csv_path):
        print(f"WARNING: {os.path.abspath(csv_path)} not found. "
              f"Are you running this from the repo ROOT folder? "
              f"(current directory: {os.getcwd()})")
        return []

    pairs = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if (row.get("source") or "").strip() != "seed":
                continue
            patient = PatientPercept(
                patient_id=row["patient_id"],
                symptoms=row["symptoms"].split(";"),
                age=int(row["age"]),
                temperature=float(row["temperature"]),
                heart_rate=int(row["heart_rate"]),
                blood_pressure=row["blood_pressure"],
            )
            pairs.append((patient, row["diagnosis"]))
    return pairs


# ------------------------------------------------------------
def _normalize_diagnosis(diagnosis: str) -> str:
    """Strip Module 2's _suspected/_confirmed suffixes so its output
    can be compared fairly against Modules 3/4/5's bare disease names --
    same normalization already applied in agent.py's _aggregate_diagnosis()."""
    if not diagnosis:
        return diagnosis
    return diagnosis.replace("_suspected", "").replace("_confirmed", "")


# ------------------------------------------------------------
def _get_probability_vector(module_name: str, module, patient,
                             disease_list: List[str]) -> Optional[np.ndarray]:
    """
    Try to get a full probability distribution over all diseases from
    a module, needed for ROC-AUC. Returns None if the module doesn't
    naturally expose one.

    - BayesianNet: compute_posterior() already returns a full distribution.
    - MLClassifier: predict_proba() on the underlying sklearn model.
    - NeuralNetwork: analyze()'s 'all_probs' already gives a full distribution.
    - KnowledgeBase: forward chaining does NOT produce a normalized
      probability over every disease (only over diseases whose rules
      actually fired) -- ROC-AUC is not computed for this module, and
      that's noted honestly in the report rather than faked.
    """
    try:
        if module_name == "BayesianNet":
            posteriors = module.compute_posterior(patient.symptoms)
            return np.array([posteriors.get(d, 0.0) for d in disease_list])

        elif module_name == "MLClassifier":
            vector = np.array([[1 if s in patient.symptoms else 0
                                 for s in module.SYMPTOM_FEATURES]])
            proba = module.best_model.predict_proba(vector)[0]
            classes = module.label_encoder.inverse_transform(range(len(proba)))
            proba_map = dict(zip(classes, proba))
            return np.array([proba_map.get(d, 0.0) for d in disease_list])

        elif module_name == "NeuralNetwork":
            result = module.predict(patient.symptoms)
            all_probs = result.get("all_probs", {})
            return np.array([all_probs.get(d, 0.0) for d in disease_list])

    except Exception:
        return None

    return None  # KnowledgeBase, or any unrecognized module


# ------------------------------------------------------------
def evaluate_module(module_name: str, module,
                     seed_patients: List[Tuple], disease_list: List[str]) -> Dict:
    """
    Run one module against every seed patient, compute accuracy,
    precision/recall/F1 (macro-averaged, appropriate for multi-class
    with small per-class counts), a confusion matrix, and ROC-AUC
    (where available).
    """
    y_true, y_pred = [], []
    proba_rows = []
    proba_available = True

    for patient, true_diagnosis in seed_patients:
        try:
            result = module.analyze(patient)
            predicted = _normalize_diagnosis(result.get("diagnosis", "Unknown"))
        except Exception:
            predicted = "Unknown"

        y_true.append(true_diagnosis)
        y_pred.append(predicted)

        proba = _get_probability_vector(module_name, module, patient, disease_list)
        if proba is None:
            proba_available = False
        else:
            proba_rows.append(proba)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=disease_list
    )
    cm = confusion_matrix(y_true, y_pred, labels=disease_list)

    roc_auc = None
    if proba_available and len(proba_rows) == len(seed_patients):
        try:
            y_true_bin = label_binarize(y_true, classes=disease_list)
            proba_matrix = np.array(proba_rows)
            # Only compute if every class actually appears at least
            # once in y_true (roc_auc_score requires this) and
            # probabilities are properly shaped.
            if y_true_bin.shape[1] == proba_matrix.shape[1]:
                roc_auc = roc_auc_score(y_true_bin, proba_matrix,
                                         average="macro", multi_class="ovr")
        except Exception:
            roc_auc = None

    return {
        "module_name": module_name,
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "roc_auc_macro": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "labels": disease_list,
    }


# ------------------------------------------------------------
def evaluate_all_modules(modules: Dict, seed_patients: List[Tuple],
                          disease_list: List[str]) -> Dict[str, Dict]:
    """Evaluate every module in `modules` (name -> instance) and
    return a dict of results keyed by module name."""
    results = {}
    for name, module in modules.items():
        results[name] = evaluate_module(name, module, seed_patients, disease_list)
    return results


# ------------------------------------------------------------
def print_metrics_report(results: Dict[str, Dict]):
    """Human-readable summary table."""
    print(f"\n{'Module':<15} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 70)
    for name, r in results.items():
        roc = f"{r['roc_auc_macro']:.4f}" if r['roc_auc_macro'] is not None else "N/A"
        print(f"{name:<15} {r['accuracy']:>10.4f} {r['precision_macro']:>10.4f} "
              f"{r['recall_macro']:>10.4f} {r['f1_macro']:>10.4f} {roc:>10}")
    print()
    print("Note: ROC-AUC is N/A for KnowledgeBase -- forward chaining does not")
    print("produce a normalized probability distribution over every disease,")
    print("only over diseases whose rules actually fired for a given patient.")


# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from modules.knowledge_base import MedicalKnowledgeBase
    from modules.bayesian_net import SimpleBayesianDiagnostics
    from modules.ml_classifier import MLDiagnosticClassifier
    from modules.neural_network import NeuralDiagnosticModel

    print("Loading seed patients...")
    seed_patients = load_seed_patients()

    if not seed_patients:
        print("\n" + "=" * 60)
        print("ERROR: No seed patients loaded from data/patient_records.csv")
        print("=" * 60)
        print("This usually means one of:")
        print("  1. data/patient_records.csv doesn't exist at this path")
        print("     (make sure you're running this from the repo ROOT,")
        print("     e.g. `python3 -m evaluation.metrics`, not from inside")
        print("     the evaluation/ folder)")
        print("  2. The CSV exists but doesn't have a 'source' column with")
        print("     'seed' values -- check the header row matches:")
        print("     patient_id,symptoms,age,temperature,heart_rate,"
              "blood_pressure,diagnosis,confidence,source")
        print("  3. The CSV has a 'source' column, but no rows are marked")
        print("     'seed' (they might all say 'runtime' instead)")
        print("\nRun this to check your file's actual header:")
        print("  head -1 data/patient_records.csv")
        sys.exit(1)

    print(f"Loaded {len(seed_patients)} ground-truth patients")

    disease_list = sorted(set(d for _, d in seed_patients))
    print(f"Diseases: {len(disease_list)}")

    print("\nInitializing and training modules...")
    kb = MedicalKnowledgeBase()
    bn = SimpleBayesianDiagnostics()
    ml = MLDiagnosticClassifier()
    ml.train(verbose=False)
    nn = NeuralDiagnosticModel()
    nn.train(epochs=30, verbose=0)

    modules = {
        "KnowledgeBase": kb,
        "BayesianNet": bn,
        "MLClassifier": ml,
        "NeuralNetwork": nn,
    }

    print("\nEvaluating all modules against seed ground truth...")
    results = evaluate_all_modules(modules, seed_patients, disease_list)
    print_metrics_report(results)