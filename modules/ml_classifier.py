# ============================================================
# MODULE 4: ML Classifier — Supervised Diagnosis
# Covers: Week 9 (Supervised Learning & Decision Trees)
#
# NOTE ON THIS REVISION:
# Two fixes applied to the original version of this file:
#
# 1. BUG FIX: `Dict` and `List` were used in type hints (e.g.
#    `-> Dict`, `symptoms: List[str]`) but never imported from
#    `typing`. Since Python evaluates function annotations at
#    definition time, this would raise `NameError: name 'Dict' is
#    not defined` the moment the module is imported -- before any
#    code even runs. Fixed by adding the missing import.
#
# 2. DATA ALIGNMENT: the original SYMPTOM_FEATURES (18 symptoms,
#    including stiff_neck, light_sensitivity, sweating,
#    frequent_urination, excessive_thirst, blurred_vision) and
#    DISEASE_LABELS (flu, covid19, dengue, cardiac_event, diabetes,
#    common_cold, tuberculosis, meningitis) didn't match the locked
#    15-disease/26-symptom list in data/diseases.csv and
#    data/symptoms.csv, which Modules 2 and 3 are already aligned to.
#    Replaced with the locked list, loaded from the CSVs directly.
#
# Everything else -- the LabelEncoder approach, top5 predictions,
# symptom_vector in predict() output, and the seaborn evaluation
# plots -- is unchanged from the original design.
# ============================================================

import csv
import os
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')


class MLDiagnosticClassifier:
    """
    Ensemble ML-based diagnostic classifier.
    Uses Decision Trees, Random Forest, and
    Gradient Boosting for robust diagnosis.
    """

    # Fallback symptom list, used only if data/symptoms.csv can't be
    # found -- normally overridden by _load_symptom_list() in __init__
    # so this always matches Modules 2/3's symptom list exactly.
    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'headache', 'sore_throat',
        'runny_nose', 'body_aches', 'chills', 'nausea', 'vomiting',
        'diarrhea', 'rash', 'joint_pain', 'chest_pain',
        'shortness_of_breath', 'loss_of_smell', 'abdominal_pain',
        'dizziness', 'night_sweats', 'weight_loss',
        'loss_of_appetite', 'jaundice', 'painful_urination',
        'facial_pain', 'swollen_tonsils', 'wheezing',
    ]

    # Fallback disease list -- normally overridden from
    # data/diseases.csv, same reasoning as above.
    DISEASE_LABELS = [
        'flu', 'covid19', 'common_cold', 'dengue', 'malaria',
        'typhoid', 'pneumonia', 'migraine', 'tuberculosis',
        'hepatitis_a', 'urinary_tract_infection', 'sinusitis',
        'gastroenteritis', 'bronchitis', 'tonsillitis',
    ]

    def __init__(self,
                 symptoms_csv: str = "data/symptoms.csv",
                 diseases_csv: str = "data/diseases.csv"):
        self.SYMPTOM_FEATURES = self._load_symptom_list(symptoms_csv)
        self.DISEASE_LABELS = self._load_disease_list(diseases_csv)

        self.models = {
            'Decision Tree':     DecisionTreeClassifier(
                max_depth=8, criterion='entropy', random_state=42),
            'Random Forest':     RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42),
        }
        self.best_model    = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        self.is_trained    = False

    # -----------------------------------------------------------------
    def _load_symptom_list(self, symptoms_csv: str) -> List[str]:
        if not os.path.exists(symptoms_csv):
            return self.SYMPTOM_FEATURES
        with open(symptoms_csv) as f:
            symptoms = [row["symptom_name"] for row in csv.DictReader(f)]
        return symptoms if symptoms else self.SYMPTOM_FEATURES

    def _load_disease_list(self, diseases_csv: str) -> List[str]:
        if not os.path.exists(diseases_csv):
            return self.DISEASE_LABELS
        with open(diseases_csv) as f:
            diseases = [row["disease_name"] for row in csv.DictReader(f)]
        return diseases if diseases else self.DISEASE_LABELS

    # -----------------------------------------------------------------
    def _generate_synthetic_data(self, n_samples: int = 4500) -> pd.DataFrame:
        """Generate realistic synthetic medical dataset.

        n_samples default raised from the original 2000 to 4500 (300 per
        disease across 15 diseases) -- 15 classes need more per-class
        samples than 8 classes did to avoid the models under-fitting
        rarer diseases.
        """
        np.random.seed(42)
        records = []

        # Disease profiles: P(symptom | disease). Same illustrative
        # probabilities used in Module 3's likelihood table, kept
        # consistent across modules.
        profiles = {
            'flu': {'fever': 0.90, 'cough': 0.85, 'fatigue': 0.88,
                    'body_aches': 0.80, 'chills': 0.60, 'sore_throat': 0.50,
                    'headache': 0.55},
            'covid19': {'fever': 0.85, 'cough': 0.80, 'fatigue': 0.75,
                        'loss_of_smell': 0.65, 'sore_throat': 0.45,
                        'shortness_of_breath': 0.40, 'headache': 0.50},
            'common_cold': {'runny_nose': 0.90, 'sore_throat': 0.75,
                             'cough': 0.65, 'fatigue': 0.40, 'fever': 0.25,
                             'headache': 0.30},
            'dengue': {'fever': 0.98, 'rash': 0.75, 'joint_pain': 0.85,
                       'headache': 0.70, 'nausea': 0.55, 'vomiting': 0.40},
            'malaria': {'fever': 0.95, 'chills': 0.85, 'headache': 0.70,
                        'fatigue': 0.65, 'nausea': 0.50, 'vomiting': 0.35},
            'typhoid': {'fever': 0.90, 'abdominal_pain': 0.70,
                        'loss_of_appetite': 0.75, 'headache': 0.55,
                        'fatigue': 0.60, 'diarrhea': 0.40},
            'pneumonia': {'cough': 0.90, 'fever': 0.80, 'chest_pain': 0.65,
                          'shortness_of_breath': 0.75, 'fatigue': 0.55,
                          'wheezing': 0.40},
            'migraine': {'headache': 0.95, 'dizziness': 0.55,
                         'nausea': 0.60, 'fatigue': 0.30},
            'tuberculosis': {'cough': 0.85, 'weight_loss': 0.75,
                              'night_sweats': 0.70, 'fatigue': 0.65,
                              'fever': 0.50, 'chest_pain': 0.40},
            'hepatitis_a': {'jaundice': 0.80, 'fatigue': 0.70,
                             'nausea': 0.65, 'loss_of_appetite': 0.60,
                             'abdominal_pain': 0.45, 'vomiting': 0.35},
            'urinary_tract_infection': {'painful_urination': 0.90,
                                         'abdominal_pain': 0.55,
                                         'fever': 0.30, 'fatigue': 0.25},
            'sinusitis': {'facial_pain': 0.85, 'headache': 0.70,
                          'runny_nose': 0.65, 'fever': 0.20},
            'gastroenteritis': {'vomiting': 0.80, 'diarrhea': 0.85,
                                 'abdominal_pain': 0.75, 'nausea': 0.70,
                                 'fever': 0.30},
            'bronchitis': {'cough': 0.90, 'wheezing': 0.60,
                            'chest_pain': 0.45, 'fatigue': 0.40,
                            'fever': 0.25},
            'tonsillitis': {'sore_throat': 0.90, 'swollen_tonsils': 0.85,
                             'fever': 0.55, 'fatigue': 0.30},
        }

        n_per_class = n_samples // len(profiles)
        for disease, symptom_probs in profiles.items():
            for _ in range(n_per_class):
                record = {f: 0 for f in self.SYMPTOM_FEATURES}
                for symptom, prob in symptom_probs.items():
                    if symptom in record:
                        record[symptom] = int(np.random.random() < prob)
                # Add background noise so unrelated diseases aren't
                # trivially separable by symptom absence alone.
                for feat in self.SYMPTOM_FEATURES:
                    if record[feat] == 0 and np.random.random() < 0.05:
                        record[feat] = 1
                record['disease'] = disease
                records.append(record)

        df = pd.DataFrame(records).sample(frac=1, random_state=42)
        return df

    # -----------------------------------------------------------------
    def train(self, verbose: bool = True) -> Dict:
        """Train all models and select the best one via cross-validation
        (per the manual's explicit guidance: use cross_val_score, not
        just a single train/test split, so the chosen model is the one
        that generalizes best)."""
        df = self._generate_synthetic_data(4500)
        X  = df[self.SYMPTOM_FEATURES].values
        y  = self.label_encoder.fit_transform(df['disease'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        results = {}
        best_cv_mean = 0.0

        if verbose:
            print("=" * 55)
            print("  ML Diagnostic Classifier — Training")
            print("=" * 55)

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            test_acc  = model.score(X_test, y_test)
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std':  cv_scores.std(),
                'test_acc': test_acc
            }
            if verbose:
                print(f"\n  {name}")
                print(f"     CV Accuracy  : {cv_scores.mean():.4f} "
                      f"+/- {cv_scores.std():.4f}")
                print(f"     Test Accuracy: {test_acc:.4f}")

            # Selection is based on cross-val mean (not single test
            # accuracy) per the manual's guidance.
            if cv_scores.mean() > best_cv_mean:
                best_cv_mean       = cv_scores.mean()
                self.best_model    = model
                self.best_model_name = name

        self.is_trained = True
        self._X_test = X_test
        self._y_test = y_test

        if verbose:
            print(f"\n  Best Model: {self.best_model_name} "
                  f"(CV accuracy: {best_cv_mean:.4f})")
        return results

    # -----------------------------------------------------------------
    def predict(self, symptoms: List[str]) -> Dict:
        """Predict disease from symptom list"""
        if not self.is_trained:
            self.train(verbose=False)

        features = np.array([
            [1 if s in symptoms else 0
             for s in self.SYMPTOM_FEATURES]
        ])
        pred_encoded = self.best_model.predict(features)[0]
        pred_proba   = self.best_model.predict_proba(features)[0]

        disease  = str(self.label_encoder.inverse_transform([pred_encoded])[0])
        classes  = self.label_encoder.inverse_transform(
            range(len(pred_proba)))
        prob_map = dict(zip(classes, pred_proba))
        top5     = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'diagnosis':      disease,
            'confidence':     round(float(pred_proba[pred_encoded]), 4),
            'top5':           top5,
            'model_used':     self.best_model_name,
            'symptom_vector': features[0].tolist()
        }

    # -----------------------------------------------------------------
    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        result = self.predict(percept.symptoms)
        result['summary'] = (f"{result['model_used']}: "
                             f"{result['diagnosis']} "
                             f"({result['confidence']:.2%})")
        return result

    # -----------------------------------------------------------------
    def plot_evaluation(self, save_path: str = "evaluation/ml_evaluation.png"):
        """Visualize model performance. Saved to disk (headless-safe --
        no plt.show(), since this needs to run in CI/terminal
        environments without a display)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not self.is_trained:
            self.train(verbose=False)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        y_pred = self.best_model.predict(self._X_test)
        cm     = confusion_matrix(self._y_test, y_pred)
        labels = self.label_encoder.classes_

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title(f"Confusion Matrix\n({self.best_model_name})",
                          fontweight='bold')
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Feature Importance
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            sorted_idx  = np.argsort(importances)[::-1][:12]
            top_features = [self.SYMPTOM_FEATURES[i] for i in sorted_idx]
            top_values   = importances[sorted_idx]
            colors = plt.cm.RdYlGn(top_values / top_values.max())
            axes[1].barh(range(len(top_features)), top_values[::-1],
                         color=colors[::-1])
            axes[1].set_yticks(range(len(top_features)))
            axes[1].set_yticklabels(top_features[::-1])
            axes[1].set_title("Feature Importances (Top 12)",
                              fontweight='bold')
            axes[1].set_xlabel("Importance Score")

        plt.suptitle(f"ML Diagnostic Model Evaluation — {self.best_model_name}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")


# ============================================================
# STANDALONE TEST
# Matches the manual's own test pattern.
# ============================================================
if __name__ == "__main__":
    clf = MLDiagnosticClassifier()
    clf.train(verbose=True)

    result = clf.predict(["fever", "cough", "fatigue", "loss_of_smell"])
    print(f"\nDiagnosis : {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Model Used: {result['model_used']}")
    print(f"Top 5     : {result['top5']}")

    clf.plot_evaluation()