"""
modules/nlp_processor.py

NLP + TEXT PROCESSING MODULE
=============================

WHAT THIS MODULE DOES
----------------------
Every example patient elsewhere in this system already arrives as a clean
symptom list, e.g. ["fever", "cough", "loss_of_smell"]. But the Agent's own
PEAS table (see agent.py) lists "symptom text input" as a SENSOR -- meaning
somewhere, a real patient's free-text complaint has to become that clean
list. Nothing in Modules 1-7 does this. This module is that missing bridge.

Example:
    Input : "I've had a fever and a bad cough for three days, and my
             nose won't stop running. No chest pain though."
    Output: ["fever", "cough", "runny_nose"]
            (chest_pain correctly EXCLUDED because it was negated)

THREE STEPS, MIRRORING CLASSIC NLP PIPELINES
----------------------------------------------
1. TOKENIZATION + NORMALIZATION
   Lowercase the text, split into tokens/n-grams using NLTK.

2. SYMPTOM MATCHING (with synonym handling)
   Real patients don't say "loss_of_smell" -- they say "can't smell
   anything". A synonym map translates common phrasings to the exact
   symptom_name values in data/symptoms.csv, which is the single source
   of truth every other module (Bayesian, ML, NN) already relies on.

3. NEGATION HANDLING
   "No fever", "denies chest pain", "without nausea" must NOT be extracted
   as present symptoms. A simple negation-window check looks a few tokens
   before each matched symptom phrase for a negation cue.

WHY THIS MATTERS FOR THE REST OF THE SYSTEM
---------------------------------------------
Modules 2-5 all match symptom strings EXACTLY against their internal
rule/likelihood/feature tables. If this module leaks a raw phrase like
"running nose" instead of the canonical "runny_nose", every downstream
module silently fails to match it -- exactly the "Common Student Mistake"
called out repeatedly in the manual for Modules 2 and 3. This module's
entire job is to prevent that class of bug at the source.
"""

import csv
import os
import re
from typing import Dict, List, Set, Tuple

try:
    import nltk
    from nltk.tokenize import word_tokenize
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


# ---------------------------------------------------------------------------
# SYNONYM MAP
# ---------------------------------------------------------------------------
# Maps common everyday phrasings -> canonical symptom_name from symptoms.csv.
# This list is NOT exhaustive -- extend it as your team encounters more
# real patient phrasings during testing (the manual's outstanding task
# "real query collection" for the other coursework project is directly
# analogous to what would grow this map over time).
# ---------------------------------------------------------------------------

SYNONYM_MAP: Dict[str, str] = {
    # fever
    "high temperature": "fever", "hot": "fever", "burning up": "fever",
    # cough
    "coughing": "cough",
    # fatigue
    "tired": "fatigue", "exhausted": "fatigue", "no energy": "fatigue",
    "low energy": "fatigue",
    # headache
    "head pain": "headache", "head hurts": "headache",
    # sore_throat
    "throat pain": "sore_throat", "scratchy throat": "sore_throat",
    "throat hurts": "sore_throat",
    # runny_nose
    "running nose": "runny_nose", "nose running": "runny_nose",
    "stuffy nose": "runny_nose", "nasal congestion": "runny_nose",
    "nose won't stop running": "runny_nose", "nose wont stop running": "runny_nose",
    "nose keeps running": "runny_nose", "nose is running": "runny_nose",
    # body_aches
    "body pain": "body_aches", "muscle pain": "body_aches",
    "aching all over": "body_aches",
    # chills
    "shivering": "chills", "cold shivers": "chills",
    # nausea
    "feel sick": "nausea", "queasy": "nausea",
    # vomiting
    "throwing up": "vomiting", "been sick": "vomiting",
    # diarrhea
    "loose stools": "diarrhea", "watery stools": "diarrhea",
    # rash
    "skin rash": "rash", "spots on skin": "rash",
    # joint_pain
    "joints hurt": "joint_pain", "painful joints": "joint_pain",
    # chest_pain
    "chest hurts": "chest_pain", "pain in chest": "chest_pain",
    # shortness_of_breath
    "hard to breathe": "shortness_of_breath", "cant breathe": "shortness_of_breath",
    "can't breathe": "shortness_of_breath", "breathless": "shortness_of_breath",
    # loss_of_smell
    "cant smell": "loss_of_smell", "can't smell": "loss_of_smell",
    "lost my sense of smell": "loss_of_smell",
    # abdominal_pain
    "stomach pain": "abdominal_pain", "stomach ache": "abdominal_pain",
    "belly pain": "abdominal_pain",
    # dizziness
    "dizzy": "dizziness", "lightheaded": "dizziness", "feel faint": "dizziness",
    # night_sweats
    "sweating at night": "night_sweats", "night sweating": "night_sweats",
    # weight_loss
    "losing weight": "weight_loss", "lost weight": "weight_loss",
    # loss_of_appetite
    "no appetite": "loss_of_appetite", "not hungry": "loss_of_appetite",
    "dont want to eat": "loss_of_appetite",
    # jaundice
    "yellow skin": "jaundice", "yellow eyes": "jaundice",
    # painful_urination
    "burning when i pee": "painful_urination", "hurts to pee": "painful_urination",
    "painful peeing": "painful_urination",
    # facial_pain
    "face pain": "facial_pain", "sinus pain": "facial_pain",
    "pressure in face": "facial_pain",
    # swollen_tonsils
    "swollen throat": "swollen_tonsils", "tonsils swollen": "swollen_tonsils",
    # wheezing
    "wheezy": "wheezing", "whistling breath": "wheezing",
}

# Words that flip a following symptom mention from "present" to "absent".
NEGATION_CUES = {"no", "not", "denies", "denied", "without", "never", "none"}


# ---------------------------------------------------------------------------
# LOAD CANONICAL SYMPTOM LIST FROM symptoms.csv
# ---------------------------------------------------------------------------

def load_canonical_symptoms(csv_path: str = "data/symptoms.csv") -> List[str]:
    """
    Load the master symptom list so this module always stays in sync with
    the same source of truth Modules 2-5 use -- never a separately
    maintained/duplicated list that can drift out of agreement.
    """
    if not os.path.exists(csv_path):
        return []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return [row["symptom_name"] for row in reader]


class SymptomExtractor:
    """
    Extracts a clean, canonical symptom list from a patient's free-text
    complaint.
    """

    def __init__(self, symptoms_csv: str = "data/symptoms.csv"):
        self.canonical_symptoms: List[str] = load_canonical_symptoms(symptoms_csv)
        # A symptom can always be referred to by its own canonical name too
        # (e.g. a patient literally says "fever") -- so canonical names are
        # themselves valid "phrases" to match, in addition to synonyms.
        self.phrase_to_symptom: Dict[str, str] = dict(SYNONYM_MAP)
        for symptom in self.canonical_symptoms:
            readable = symptom.replace("_", " ")
            self.phrase_to_symptom.setdefault(readable, symptom)
            self.phrase_to_symptom.setdefault(symptom, symptom)

        # Sort phrases longest-first so multi-word phrases (e.g.
        # "shortness of breath") are matched before shorter substrings
        # that might otherwise match part of them incorrectly.
        self._sorted_phrases = sorted(self.phrase_to_symptom.keys(),
                                       key=len, reverse=True)

    # -----------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        """Lowercase + tokenize using NLTK if available, else a regex fallback."""
        text = text.lower()
        if _NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except LookupError:
                pass  # punkt data not downloaded -- fall through to regex
        # Fallback: simple word-boundary regex tokenizer.
        return re.findall(r"[a-z']+", text)

    # -----------------------------------------------------------------
    def extract(self, text: str) -> List[str]:
        """
        Main entry point: free text -> list of canonical symptom names.

        Returns symptoms in the exact lower_snake_case format Modules 2-5
        expect, with negated mentions excluded.
        """
        lowered = " " + text.lower().strip() + " "
        found: Set[str] = set()

        for phrase in self._sorted_phrases:
            symptom = self.phrase_to_symptom[phrase]
            if symptom in found:
                continue  # already matched via a different phrase

            # Use word-boundary regex so "hot" doesn't match inside "shot"
            pattern = r"\b" + re.escape(phrase) + r"\b"
            for match in re.finditer(pattern, lowered):
                if not self._is_negated(lowered, match.start()):
                    found.add(symptom)
                break  # one confirmed match is enough for this phrase

        return sorted(found)

    # -----------------------------------------------------------------
    def _is_negated(self, text: str, match_start: int, window: int = 4) -> bool:
        """
        Check whether a negation cue appears in the few words immediately
        before a matched symptom phrase.

        Example: "no fever" -> negated. "fever for three days" -> not negated.

        `window` = how many preceding words to check. A small window keeps
        this from over-triggering on unrelated negations earlier in a long
        sentence (e.g. "I didn't sleep well, and I have a fever" should
        still count fever as present).
        """
        preceding_text = text[:match_start]
        preceding_words = re.findall(r"[a-z']+", preceding_text)
        nearby_words = preceding_words[-window:]
        return any(word in NEGATION_CUES for word in nearby_words)


# ---------------------------------------------------------------------------
# CONVENIENCE FUNCTION for direct use elsewhere (e.g. app.py, agent.py)
# ---------------------------------------------------------------------------

def extract_symptoms_from_text(text: str,
                                symptoms_csv: str = "data/symptoms.csv"
                                ) -> List[str]:
    """One-line helper: raw complaint text -> canonical symptom list."""
    extractor = SymptomExtractor(symptoms_csv)
    return extractor.extract(text)


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    extractor = SymptomExtractor()

    test_cases = [
        "I've had a fever and a bad cough for three days, and my nose "
        "won't stop running. No chest pain though.",

        "Feeling really tired and dizzy, plus I can't smell anything "
        "anymore. Denies vomiting.",

        "My joints hurt and I have a skin rash, but no fever at all.",

        "Hurts to pee and I've been having stomach pain since yesterday.",
    ]

    for i, text in enumerate(test_cases, 1):
        symptoms = extractor.extract(text)
        print(f"Test {i}: \"{text}\"")
        print(f"  -> Extracted symptoms: {symptoms}\n")