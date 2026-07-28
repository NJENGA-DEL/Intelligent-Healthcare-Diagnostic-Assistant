# ============================================================
# EVALUATION MODULE: visualizations.py
#
# Generates the visual deliverables the manual's checklist requires:
#   [ ] Confusion matrices generated for each classifier
#   [ ] Module comparison bar chart generated
#
# Takes the `results` dict produced by metrics.evaluate_all_modules().
# ============================================================

import os
from typing import Dict

import numpy as np


def plot_confusion_matrices(results: Dict[str, Dict],
                             save_path: str = "evaluation/confusion_matrices.png"):
    """
    One confusion matrix heatmap per module, in a grid, so all 4
    diagnostic modules can be visually compared at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    n_modules = len(results)
    fig, axes = plt.subplots(1, n_modules, figsize=(7 * n_modules, 6))
    if n_modules == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=r["confusion_matrix"], display_labels=r["labels"]
        )
        disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
        ax.set_title(f"{name}\nAccuracy: {r['accuracy']:.1%}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_module_comparison(results: Dict[str, Dict],
                            save_path: str = "evaluation/module_comparison.png"):
    """
    Single bar chart comparing accuracy, precision, recall, and F1
    across all evaluated modules -- the manual's required "module
    comparison bar chart" deliverable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    modules = list(results.keys())
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    x = np.arange(len(modules))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[m][metric] for m in modules]
        ax.bar(x + i * width, values, width, label=label)

    ax.set_xlabel("Module")
    ax.set_ylabel("Score")
    ax.set_title("Diagnostic Module Comparison (vs. seed ground truth)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(modules)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_all_evaluation_plots(results: Dict[str, Dict],
                                   output_dir: str = "evaluation"):
    """Convenience wrapper: generate both required plots at once."""
    plot_confusion_matrices(results, os.path.join(output_dir, "confusion_matrices.png"))
    plot_module_comparison(results, os.path.join(output_dir, "module_comparison.png"))


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
    from evaluation.metrics import load_seed_patients, evaluate_all_modules, print_metrics_report

    seed_patients = load_seed_patients()

    if not seed_patients:
        print("\nERROR: No seed patients loaded from data/patient_records.csv")
        print("Run this to check your file's actual header:")
        print("  head -1 data/patient_records.csv")
        print("It should read:")
        print("  patient_id,symptoms,age,temperature,heart_rate,"
              "blood_pressure,diagnosis,confidence,source")
        sys.exit(1)

    disease_list = sorted(set(d for _, d in seed_patients))

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

    results = evaluate_all_modules(modules, seed_patients, disease_list)
    print_metrics_report(results)
    generate_all_evaluation_plots(results)