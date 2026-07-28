# ============================================================
# MODULE 5: Deep Neural Network Diagnostic Model
# Covers: Week 10 (Neural Networks)
#
# NOTE ON THIS REVISION:
# Two fixes applied to the original version of this file:
#
# 1. BUG FIX: `Dict` and `List` were used in type hints (`-> Dict`,
#    `symptoms: List[str]`) but never imported from `typing`. Since
#    Python evaluates function annotations at definition time, this
#    raises `NameError` the moment the module is imported. Fixed by
#    adding the missing import.
#
# 2. DATA ALIGNMENT: SYMPTOM_FEATURES (18 symptoms) and
#    DISEASE_LABELS (8 diseases: flu, covid19, dengue, cardiac_event,
#    diabetes, common_cold, tuberculosis, meningitis) didn't match
#    the locked 15-disease/26-symptom list in data/diseases.csv and
#    data/symptoms.csv, which Modules 2, 3, and 4 are already aligned
#    to. Replaced with the locked list, loaded from the CSVs.
#    The output layer is now Dense(15, softmax) instead of Dense(8, ...).
#
# The architecture itself (128 -> 64 -> 32 -> output, BatchNorm,
# Dropout, L2 regularization, EarlyStopping, ReduceLROnPlateau) is
# completely unchanged -- it already matched the manual's spec exactly.
# ============================================================

import csv
import os
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


class NeuralDiagnosticModel:
    """
    Deep Neural Network for medical diagnosis.
    Architecture: Input -> Dense -> BN -> Dropout -> Output
    """

    # Fallback lists, used only if the CSVs can't be found -- normally
    # overridden in __init__ so this always matches Modules 2/3/4.
    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'headache', 'sore_throat',
        'runny_nose', 'body_aches', 'chills', 'nausea', 'vomiting',
        'diarrhea', 'rash', 'joint_pain', 'chest_pain',
        'shortness_of_breath', 'loss_of_smell', 'abdominal_pain',
        'dizziness', 'night_sweats', 'weight_loss',
        'loss_of_appetite', 'jaundice', 'painful_urination',
        'facial_pain', 'swollen_tonsils', 'wheezing',
    ]

    DISEASE_LABELS = [
        'flu', 'covid19', 'common_cold', 'dengue', 'malaria',
        'typhoid', 'pneumonia', 'migraine', 'tuberculosis',
        'hepatitis_a', 'urinary_tract_infection', 'sinusitis',
        'gastroenteritis', 'bronchitis', 'tonsillitis',
    ]

    def __init__(self,
                 symptoms_csv: str = "data/symptoms.csv",
                 diseases_csv: str = "data/diseases.csv"):
        self.SYMPTOM_FEATURES = self._load_list(symptoms_csv, "symptom_name",
                                                  self.SYMPTOM_FEATURES)
        self.DISEASE_LABELS = self._load_list(diseases_csv, "disease_name",
                                                self.DISEASE_LABELS)

        self.model      = None
        self.history    = None
        self.is_trained = False
        self._build_model()

    # -----------------------------------------------------------------
    @staticmethod
    def _load_list(csv_path: str, column: str, fallback: List[str]) -> List[str]:
        if not os.path.exists(csv_path):
            return fallback
        with open(csv_path) as f:
            values = [row[column] for row in csv.DictReader(f)]
        return values if values else fallback

    # -----------------------------------------------------------------
    def _build_model(self):
        """Build deep MLP architecture"""
        n_inputs  = len(self.SYMPTOM_FEATURES)
        n_outputs = len(self.DISEASE_LABELS)

        self.model = models.Sequential([
            layers.Input(shape=(n_inputs,)),

            # Block 1
            layers.Dense(128, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Block 2
            layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            # Block 3
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),

            # Output -- n_outputs now driven by the locked 15-disease list
            layers.Dense(n_outputs, activation='softmax')
        ], name='MedicalDNN')

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    # -----------------------------------------------------------------
    def _generate_data(self, n: int = 4500):
        """Generate synthetic training data.

        n default raised from 3000 to 4500 (300 per disease across 15
        diseases instead of 8) -- more classes need more per-class
        samples to train reliably.
        """
        np.random.seed(42)

        # Same illustrative disease profiles used in Modules 3 and 4,
        # kept consistent across modules.
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

        X_list, y_list = [], []
        n_per = n // len(profiles)

        for label_idx, disease in enumerate(self.DISEASE_LABELS):
            probs = profiles.get(disease, {})
            for _ in range(n_per):
                row = np.array([
                    1 if (np.random.random() <
                          probs.get(feat, 0.03)) else 0
                    for feat in self.SYMPTOM_FEATURES
                ], dtype=np.float32)
                X_list.append(row)
                y_list.append(label_idx)

        X = np.array(X_list)
        y = np.array(y_list)
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

    # -----------------------------------------------------------------
    def train(self, epochs: int = 50, verbose: int = 1) -> Dict:
        """Train the neural network"""
        X, y = self._generate_data(4500)
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        cb_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10,
                restore_best_weights=True),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=5, min_lr=1e-6)
        ]

        print("=" * 55)
        print("  Neural Network — Medical Diagnosis Training")
        print(f"  Architecture: {len(self.SYMPTOM_FEATURES)} -> "
              f"128 -> 64 -> 32 -> {len(self.DISEASE_LABELS)}")
        print("=" * 55)
        self.model.summary()

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=64,
            callbacks=cb_list, verbose=verbose
        )

        val_acc = max(self.history.history['val_accuracy'])
        self.is_trained = True
        print(f"\nBest Validation Accuracy: {val_acc:.4f}")
        return {'val_accuracy': val_acc}

    # -----------------------------------------------------------------
    def predict(self, symptoms: List[str]) -> Dict:
        """Predict from symptom list"""
        if not self.is_trained:
            self.train(verbose=0)

        cleaned = [s.lower().replace(' ', '_') for s in symptoms]
        features = np.array([
            [1.0 if feat in cleaned else 0.0
             for feat in self.SYMPTOM_FEATURES]
        ], dtype=np.float32)

        proba     = self.model.predict(features, verbose=0)[0]
        pred_idx  = int(np.argmax(proba))
        diagnosis = self.DISEASE_LABELS[pred_idx]

        return {
            'diagnosis':  diagnosis,
            'confidence': round(float(proba[pred_idx]), 4),
            'all_probs':  dict(zip(self.DISEASE_LABELS,
                                   proba.round(4).tolist()))
        }

    # -----------------------------------------------------------------
    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        result = self.predict(percept.symptoms)
        result['summary'] = (f"DNN: {result['diagnosis']} "
                             f"({result['confidence']:.2%})")
        return result

    # -----------------------------------------------------------------
    def plot_training(self, save_path: str = "evaluation/nn_training.png"):
        """Plot training history. Saved to disk (headless-safe -- no
        plt.show(), since this needs to run without a display)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.history:
            print("Train model first!")
            return

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        metrics = [('accuracy', 'val_accuracy', 'Accuracy'),
                   ('loss',     'val_loss',     'Loss')]
        colors  = [('#3498db', '#e74c3c'), ('#2ecc71', '#e67e22')]

        for ax, (train_m, val_m, title), (tc, vc) in zip(
                axes, metrics, colors):
            ax.plot(self.history.history[train_m],
                    color=tc, linewidth=2, label='Train')
            ax.plot(self.history.history[val_m],
                    color=vc, linewidth=2,
                    linestyle='--', label='Validation')
            ax.set_title(f"Model {title}",
                         fontsize=13, fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle("Neural Network Training Curves",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {save_path}")


# ============================================================
# STANDALONE TEST
# Matches the manual's own test pattern (epochs reduced to 30
# for faster testing, per the manual's own suggestion).
# ============================================================
if __name__ == "__main__":
    nn = NeuralDiagnosticModel()
    nn.train(epochs=30)

    result = nn.predict(["fever", "rash", "joint_pain", "headache"])
    print(f"\nDiagnosis : {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2%}")

    nn.plot_training()