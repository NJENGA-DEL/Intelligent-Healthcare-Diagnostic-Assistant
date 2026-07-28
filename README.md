# Intelligent Healthcare Diagnostic Assistant

**AI Capstone Project — CCS 3101 Introduction to Artificial Intelligence**
Dedan Kimathi University of Science and Technology

An end-to-end AI system that integrates intelligent agents, search, probabilistic reasoning, machine learning, deep learning, NLP, fuzzy logic, reinforcement learning, and automated planning into a unified diagnostic and treatment-recommendation platform.

---

## Overview

This system simulates a hospital triage pipeline. A patient's symptoms and vitals are evaluated by four independent diagnostic modules — a rule-based expert system, a Bayesian probabilistic reasoner, a classical machine learning classifier, and a deep neural network — whose opinions are aggregated by a central coordinating agent into a single diagnosis and urgency level. A separate fuzzy logic module independently assesses severity, and a STRIPS-style planner generates a concrete, step-by-step treatment plan based on the final diagnosis.

No single module has the final say. The design deliberately mirrors how a real medical team works: several specialists weigh in, and a coordinator synthesizes their opinions into one recommendation.

## Key Design Principle

The Agent (`modules/agent.py`) never diagnoses anything itself. It only knows how to call `.analyze(patient)` on whatever modules are registered with it and combine the results — meaning any diagnostic module can be added, removed, or replaced without changing the Agent's code, as long as it honors that interface.

---

## System Architecture

```
                        PATIENT INPUT
                (symptoms, vitals, age, history)
                              │
                              ▼
                    ┌───────────────────┐
                    │   AGENT (Module 1) │
                    │  Perceive→Think→Act │
                    └─────────┬──────────┘
                              │
        ┌──────────┬──────────┼──────────┬──────────┐
        ▼          ▼          ▼          ▼
  ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
  │Knowledge ││ Bayesian ││    ML    ││  Neural  │
  │  Base    ││ Network  ││Classifier││ Network  │
  │(Module 2)││(Module 3)││(Module 4)││(Module 5)│
  └────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
       └────────────┴───────────┴────────────┘
                     │
          Majority-vote aggregation
                     │
                     ▼
          ┌─────────────────────┐
          │  Aggregated Diagnosis │
          │     + Urgency Level    │
          └──────────┬────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
 ┌─────────────┐┌──────────┐┌──────────────┐
 │Fuzzy Severity││ RL Agent ││   Planner    │
 │  (Module 6)  ││(compare) ││ (Module 7)   │
 └─────────────┘└──────────┘└──────┬───────┘
                                     ▼
                          FINAL PATIENT REPORT
              (Diagnosis, Confidence, Urgency, Severity,
                   Similar Cases, Treatment Plan)
```

---

## Modules

| #   | Module                     | File                          | AI Concept                                                          |
| --- | -------------------------- | ----------------------------- | ------------------------------------------------------------------- |
| 1   | Intelligent Agent          | `modules/agent.py`            | Model-Based, Goal-Based Agent · PEAS · Perceive-Think-Act cycle     |
| 2   | Knowledge Base & Inference | `modules/knowledge_base.py`   | First-Order Logic · Forward & backward chaining · Certainty factors |
| 3   | Bayesian Network           | `modules/bayesian_net.py`     | Naive Bayes · Probabilistic reasoning · Log-space computation       |
| 4   | ML Diagnostic Classifier   | `modules/ml_classifier.py`    | Decision Tree, Random Forest, Gradient Boosting · Cross-validation  |
| 5   | Deep Neural Network        | `modules/neural_network.py`   | MLP · BatchNorm · Dropout · Early stopping · L2 regularization      |
| 6   | Fuzzy Severity Assessor    | `modules/fuzzy_controller.py` | Fuzzification · Rule evaluation · Centroid defuzzification          |
| 7   | Treatment Planner          | `modules/planner.py`          | STRIPS planning · BFS state-space search                            |
| —   | Search Algorithms          | `modules/search.py`           | BFS, DFS, UCS, A\* · Case-based similarity search                   |
| —   | NLP Processor              | `modules/nlp_processor.py`    | Tokenization · Synonym matching · Negation handling                 |
| —   | RL Triage Agent            | `modules/rl_agent.py`         | Q-learning · Epsilon-greedy exploration · Reward shaping            |

Modules 1–7 form the manual's core pipeline. `search.py`, `nlp_processor.py`, and `rl_agent.py` extend coverage into Search, NLP, and Reinforcement Learning respectively.

### Why the Planner and Fuzzy Assessor aren't registered like the other four

Modules 2–5 all share a common contract — given a patient, predict a **disease** with a confidence score — which is what the Agent's majority-vote aggregation is built for. Fuzzy Severity answers a different question ("how severe?" not "what disease?"), and the Planner needs the Agent's _already-aggregated_ diagnosis and urgency as input, which doesn't exist yet while the other four are still being consulted in parallel. Both are called explicitly by `app.py`, downstream of the Agent's decision — not registered through `agent.register_module()`.

---

## Project Structure

```
├── data/
│   ├── symptoms.csv           # 26 symptoms — shared vocabulary across all modules
│   ├── diseases.csv           # 15 diseases + prior probabilities
│   └── patient_records.csv    # Ground-truth seed patients + auto-logged runtime predictions
│
├── modules/
│   ├── agent.py
│   ├── knowledge_base.py
│   ├── bayesian_net.py
│   ├── ml_classifier.py
│   ├── neural_network.py
│   ├── fuzzy_controller.py
│   ├── planner.py
│   ├── search.py
│   ├── nlp_processor.py
│   └── rl_agent.py
│
├── evaluation/
│   ├── metrics.py              # Accuracy, precision, recall, F1, ROC-AUC per module
│   └── visualizations.py       # Confusion matrices + module comparison chart
│
├── reports/
│   └── final_report.pdf
│
├── app.py                      # Main entry point
├── requirements.txt
└── README.md
```

---

## Diseases & Symptoms Covered

**15 diseases:** flu, covid19, common_cold, dengue, malaria, typhoid, pneumonia, migraine, tuberculosis, hepatitis_a, urinary_tract_infection, sinusitis, gastroenteritis, bronchitis, tonsillitis

**26 symptoms:** see `data/symptoms.csv` for the full list with descriptions.

> **Note:** No real hospital dataset exists for this project. All disease-symptom probability relationships used to train Modules 3, 4, and 5 are illustrative, synthetic values chosen for reasonable medical plausibility — not sourced from real clinical or epidemiological data. This is consistent with the manual's own guidance to generate synthetic training data.

---

## Setup

### 1. Clone and enter the repository

```bash
git clone https://github.com/NJENGA-DEL/Intelligent-Healthcare-Diagnostic-Assistant.git
cd Intelligent-Healthcare-Diagnostic-Assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK tokenizer data (one-time)

```bash
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Usage

Run the application from the **repository root**:

```bash
python3 app.py
```

You'll be prompted to choose a mode:

```
1. Run the built-in test patients
2. Interactive mode (describe your own symptoms live)
3. Both
```

- **Option 1** runs 7 hardcoded test patients (6 structured + 1 via free-text NLP extraction) through the full pipeline — this matches the "at least 5 test patients diagnosed" deliverable.
- **Option 2** lets you type your own symptoms as a sentence or comma-separated list, plus vitals, and get a live diagnosis.
- **Option 3** runs both in sequence.

Each patient report includes: diagnosis, confidence, urgency level, fuzzy severity score, the RL agent's independently-learned action recommendation (for comparison against the Agent's own rule-based recommendation), similar historical cases, and a full step-by-step treatment plan.

### Running Evaluation

```bash
python3 -m evaluation.metrics          # Prints accuracy/precision/recall/F1/ROC-AUC per module
python3 -m evaluation.visualizations   # Generates confusion_matrices.png and module_comparison.png
```

Both scripts must be run from the repository root and evaluate every module against the 15 hand-curated **ground-truth** patients in `data/patient_records.csv` (rows marked `source=seed`). Rows marked `source=runtime` are the AI's own unverified predictions, automatically logged each time `app.py` runs, and are excluded from evaluation to avoid the system grading itself against its own guesses.

### Testing Individual Modules

Every module can be run standalone for isolated testing:

```bash
python3 -m modules.agent
python3 -m modules.knowledge_base
python3 -m modules.bayesian_net
python3 -m modules.ml_classifier
python3 -m modules.neural_network
python3 -m modules.fuzzy_controller
python3 -m modules.planner
python3 -m modules.search
python3 -m modules.nlp_processor
python3 -m modules.rl_agent
```

---

## Data Files

| File                       | Purpose                                                                                                                                                                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/symptoms.csv`        | Master list of 26 symptoms. Every module loads from this file so all diagnostic modules share the exact same symptom vocabulary and ordering.                                                                                               |
| `data/diseases.csv`        | 15 diseases with prior probabilities (summing to 1.0), loaded by the Bayesian, ML, and Neural Network modules.                                                                                                                              |
| `data/patient_records.csv` | Ground-truth seed patients (`source=seed`, hand-curated, one per disease) used for evaluation, plus auto-logged runtime predictions (`source=runtime`) that accumulate across every `app.py` run and feed the case-based similarity search. |

---

## Evaluation Results

Metrics are computed against the 15-patient seed ground-truth set using scikit-learn (`accuracy_score`, `precision_recall_fscore_support`, `confusion_matrix`, `roc_auc_score`). ROC-AUC is reported as N/A for the Knowledge Base module, since forward chaining does not produce a normalized probability distribution over every disease — only over diseases whose rules actually fired for a given patient.

> The seed evaluation set is intentionally small (15 patients, one per disease). Results demonstrate that the evaluation pipeline itself works correctly, not a statistically robust accuracy claim — expanding `patient_records.csv` with more labeled examples per disease would make these numbers more meaningful.

---

## Known Limitations

- **Synthetic data only.** No real patient data was used anywhere in this project; all training data and symptom-disease relationships are illustrative approximations, not clinical fact.
- **Small evaluation set.** 15 ground-truth patients (one per disease) is sufficient to demonstrate the evaluation pipeline, not to make statistically rigorous accuracy claims.
- **Fuzzy severity scores** are internally consistent but don't exactly reproduce the manual's own illustrative example numbers, since the manual describes membership function _shapes_ in words without specifying exact formulas.
- **This is not a medical device.** Nothing in this system should be used for real diagnostic or treatment decisions.

---

## Team

Built as part of the CCS 3101 AI Capstone, Weeks 8–13.
