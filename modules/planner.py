# ============================================================
# MODULE 7: AI Planning — Treatment Plan Generator
# Covers: Week 12 (AI Planning Techniques)
#
# NOTE ON THIS REVISION -- two real bugs were found and fixed, plus
# disease-list alignment:
#
# BUG 1 (unreachable goal for ALL viral-infection diagnoses,
# including the manual's own covid19 example): `VIRAL_INFECTION` is
# only ever REMOVED by PrescribeAntiviral, never ADDED by any action.
# So it must already be present in the initial state, or
# PrescribeAntiviral's precondition can never be satisfied, and
# nothing else produces TREATMENT_STARTED. The original
# diagnosis_states mapping never included it for covid19 -- meaning
# the manual's own flagship 6-step COVID plan was NEVER actually
# reachable. Verified by running generate_plan() directly: it
# returned None. Fixed by including the infection-type fact
# (VIRAL_INFECTION / BACTERIAL_INFECTION / PARASITIC_INFECTION /
# NEUROLOGICAL_CONDITION) directly in each diagnosis's initial state.
#
# BUG 2 (unreachable goal for ANY critical-urgency patient):
# PATIENT_IN_ICU is only reachable via
# CallEmergencyServices -> TransferToICU, which requires
# EMERGENCY_CASE and ICU_AVAILABLE already in the initial state. The
# original code added PATIENT_IN_ICU to the GOAL for CRITICAL
# urgency, but never added the prerequisite facts to the INITIAL
# STATE -- so no critical-urgency plan could ever be found. Verified
# the same way. Fixed in create_treatment_plan().
#
# DATA ALIGNMENT: diagnosis_states originally covered
# flu/covid19/cardiac_event/dengue/meningitis/tuberculosis/diabetes/
# common_cold -- replaced with the locked 15-disease list. Two
# diseases needed action-library additions the original 8-disease
# version had no path for:
#   - malaria (parasitic, not viral/bacterial) -> new
#     PrescribeAntimalarial action
#   - migraine (not an infection at all) -> new PrescribePainRelief
#     action, skipping the infection-diagnosis chain entirely
# ============================================================

from collections import deque
from typing import Dict, List, Set, Tuple, Optional


class TreatmentPlanner:
    """
    STRIPS-based treatment planner.
    Generates step-by-step treatment plans
    from patient diagnosis to recovery.
    """

    def __init__(self):
        self.action_library = self._build_action_library()

    def _build_action_library(self) -> List[Dict]:
        """Define medical treatment actions"""
        return [
            # Emergency Actions
            {
                'name': 'CallEmergencyServices',
                'precond': {'EMERGENCY_CASE', 'PATIENT_PRESENT'},
                'delete':  {'EMERGENCY_CASE'},
                'add':     {'EMERGENCY_SERVICES_CALLED'},
                'cost': 0, 'duration': '5 minutes'
            },
            {
                'name': 'TransferToICU',
                'precond': {'EMERGENCY_SERVICES_CALLED', 'ICU_AVAILABLE'},
                'delete':  {'EMERGENCY_SERVICES_CALLED'},
                'add':     {'PATIENT_IN_ICU', 'MONITORING_ACTIVE'},
                'cost': 0, 'duration': '15 minutes'
            },
            # Diagnostics
            {
                'name': 'OrderBloodPanel',
                'precond': {'PATIENT_PRESENT', 'DIAGNOSIS_NEEDED'},
                'delete':  {'DIAGNOSIS_NEEDED'},
                'add':     {'BLOOD_RESULTS_PENDING'},
                'cost': 1, 'duration': '30 minutes'
            },
            {
                'name': 'ReceiveBloodResults',
                'precond': {'BLOOD_RESULTS_PENDING'},
                'delete':  {'BLOOD_RESULTS_PENDING'},
                'add':     {'BLOOD_RESULTS_AVAILABLE', 'DIAGNOSIS_REFINED'},
                'cost': 0, 'duration': '2 hours'
            },
            {
                # NEW: BUG 3 FIX. The original action library had only
                # ONE path to DIAGNOSIS_CONFIRMED -- the COVID-specific
                # PCR pathway. Every Prescribe* action requires
                # DIAGNOSIS_CONFIRMED, but the generic diagnostic path
                # (OrderBloodPanel -> ReceiveBloodResults) only ever
                # produced DIAGNOSIS_REFINED, with nothing bridging the
                # two. This meant every disease OTHER than covid19 had
                # NO possible path to treatment at all -- confirmed by
                # running the full 15-disease sweep, where only
                # covid19 (PCR path) and migraine (a new action that
                # skips diagnostics entirely) produced a plan; all 13
                # others failed with "No plan found".
                'name': 'ConfirmDiagnosis',
                'precond': {'DIAGNOSIS_REFINED'},
                'delete':  set(),
                'add':     {'DIAGNOSIS_CONFIRMED'},
                'cost': 0, 'duration': '1 hour'
            },
            {
                'name': 'OrderPCRTest',
                'precond': {'COVID_SUSPECTED', 'PATIENT_PRESENT'},
                'delete':  {'COVID_SUSPECTED'},
                'add':     {'PCR_PENDING'},
                'cost': 1, 'duration': '24 hours'
            },
            {
                'name': 'ReceivePCRResult',
                'precond': {'PCR_PENDING'},
                'delete':  {'PCR_PENDING'},
                'add':     {'PCR_RESULT_AVAILABLE', 'DIAGNOSIS_CONFIRMED'},
                'cost': 0, 'duration': '24 hours'
            },
            # Treatment
            {
                'name': 'PrescribeAntiviral',
                'precond': {'DIAGNOSIS_CONFIRMED', 'VIRAL_INFECTION'},
                'delete':  {'VIRAL_INFECTION'},
                'add':     {'ANTIVIRAL_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '10 minutes'
            },
            {
                'name': 'PrescribeAntibiotics',
                'precond': {'DIAGNOSIS_CONFIRMED', 'BACTERIAL_INFECTION'},
                'delete':  {'BACTERIAL_INFECTION'},
                'add':     {'ANTIBIOTICS_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '10 minutes'
            },
            {
                # NEW: malaria is parasitic, not viral or bacterial --
                # the original 2-action treatment set (antiviral/
                # antibiotic only) had no valid path for it at all.
                'name': 'PrescribeAntimalarial',
                'precond': {'DIAGNOSIS_CONFIRMED', 'PARASITIC_INFECTION'},
                'delete':  {'PARASITIC_INFECTION'},
                'add':     {'ANTIMALARIAL_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '10 minutes'
            },
            {
                # NEW: migraine is not an infection -- deliberately
                # skips the whole diagnosis-confirmation chain, since
                # migraines are typically diagnosed clinically without
                # lab tests.
                'name': 'PrescribePainRelief',
                'precond': {'PATIENT_PRESENT', 'NEUROLOGICAL_CONDITION'},
                'delete':  {'NEUROLOGICAL_CONDITION'},
                'add':     {'PAIN_RELIEF_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '10 minutes'
            },
            {
                'name': 'AdministerFluids',
                'precond': {'PATIENT_IN_ICU', 'DEHYDRATION_RISK'},
                'delete':  {'DEHYDRATION_RISK'},
                'add':     {'FLUIDS_ADMINISTERED'},
                'cost': 1, 'duration': '1 hour'
            },
            {
                'name': 'MonitorVitals',
                'precond': {'TREATMENT_STARTED', 'PATIENT_PRESENT'},
                'delete':  set(),
                'add':     {'VITALS_MONITORED'},
                'cost': 0, 'duration': 'Continuous'
            },
            {
                'name': 'IsolatePatient',
                'precond': {'CONTAGIOUS_DISEASE', 'PATIENT_PRESENT'},
                'delete':  {'CONTAGIOUS_DISEASE'},
                'add':     {'PATIENT_ISOLATED'},
                'cost': 0, 'duration': '14 days'
            },
            {
                'name': 'ScheduleFollowUp',
                'precond': {'TREATMENT_STARTED', 'VITALS_MONITORED'},
                'delete':  set(),
                'add':     {'FOLLOWUP_SCHEDULED', 'PLAN_COMPLETE'},
                'cost': 0, 'duration': '5 minutes'
            },
            {
                'name': 'DischargePatient',
                'precond': {'PLAN_COMPLETE', 'SYMPTOMS_RESOLVED'},
                'delete':  {'PLAN_COMPLETE'},
                'add':     {'PATIENT_DISCHARGED'},
                'cost': 0, 'duration': '30 minutes'
            },
        ]

    def _apply_action(self, state: frozenset,
                      action: Dict) -> Optional[frozenset]:
        if not action['precond'].issubset(state):
            return None
        return frozenset((state - action['delete']) | action['add'])

    def generate_plan(self,
                      initial_state: Set[str],
                      goal_state:    Set[str]) -> Optional[List[Dict]]:
        """BFS-based plan generation"""
        initial = frozenset(initial_state)
        goal    = frozenset(goal_state)

        queue   = deque([(initial, [])])
        visited = {initial}

        while queue:
            state, plan = queue.popleft()
            if goal.issubset(state):
                return plan

            for action in self.action_library:
                new_state = self._apply_action(state, action)
                if new_state and new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, plan + [action]))

        return None

    def create_treatment_plan(self, diagnosis: str,
                              urgency: str) -> Dict:
        """Generate a treatment plan for a given diagnosis"""

        # Map diagnosis to initial state predicates. Every entry
        # includes the correct INFECTION-TYPE fact (VIRAL_INFECTION /
        # BACTERIAL_INFECTION / PARASITIC_INFECTION /
        # NEUROLOGICAL_CONDITION) up front -- these facts are only
        # ever consumed by a Prescribe* action, never produced by one,
        # so they MUST start in the initial state or no plan reaching
        # TREATMENT_STARTED can ever be found (this was Bug 1).
        #
        # Simplified infection-type classification for this academic
        # project -- e.g. some of these (bronchitis, gastroenteritis)
        # are commonly viral OR bacterial in reality; one is chosen
        # here for a clean STRIPS action path, not as clinical guidance.
        diagnosis_states = {
            'flu':          {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
            'covid19':      {'COVID_SUSPECTED', 'CONTAGIOUS_DISEASE',
                             'DIAGNOSIS_NEEDED', 'VIRAL_INFECTION'},
            'common_cold':  {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
            'dengue':       {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED',
                             'DEHYDRATION_RISK'},
            'malaria':      {'PARASITIC_INFECTION', 'DIAGNOSIS_NEEDED'},
            'typhoid':      {'BACTERIAL_INFECTION', 'DIAGNOSIS_NEEDED'},
            'pneumonia':    {'BACTERIAL_INFECTION', 'DIAGNOSIS_NEEDED'},
            'migraine':     {'NEUROLOGICAL_CONDITION'},
            'tuberculosis': {'BACTERIAL_INFECTION', 'CONTAGIOUS_DISEASE',
                             'DIAGNOSIS_NEEDED'},
            'hepatitis_a':  {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
            'urinary_tract_infection': {'BACTERIAL_INFECTION',
                                         'DIAGNOSIS_NEEDED'},
            'sinusitis':    {'BACTERIAL_INFECTION', 'DIAGNOSIS_NEEDED'},
            'gastroenteritis': {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED',
                                'DEHYDRATION_RISK'},
            'bronchitis':   {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
            'tonsillitis':  {'BACTERIAL_INFECTION', 'DIAGNOSIS_NEEDED'},
        }

        base_state = {'PATIENT_PRESENT'}
        dx_state   = diagnosis_states.get(
            diagnosis.lower().replace(' ', '_'),
            {'DIAGNOSIS_NEEDED'}
        )
        initial_state = base_state | dx_state

        # Goal state: always end with treatment and monitoring.
        goal_state = {'TREATMENT_STARTED', 'VITALS_MONITORED',
                      'FOLLOWUP_SCHEDULED'}

        # If the disease is contagious, isolation is REQUIRED, not
        # optional -- add it to the goal so BFS won't skip it (it
        # would otherwise, since IsolatePatient doesn't feed into any
        # other goal fact).
        if 'CONTAGIOUS_DISEASE' in initial_state:
            goal_state.add('PATIENT_ISOLATED')

        # BUG 2 FIX: for CRITICAL urgency, PATIENT_IN_ICU must be
        # reachable, which means EMERGENCY_CASE and ICU_AVAILABLE need
        # to be in the INITIAL state (not just PATIENT_IN_ICU added to
        # the goal, which was the original bug -- that left the goal
        # permanently unreachable for every critical-urgency patient).
        if urgency == 'CRITICAL':
            initial_state = initial_state | {'EMERGENCY_CASE', 'ICU_AVAILABLE'}
            goal_state.add('PATIENT_IN_ICU')

        plan = self.generate_plan(initial_state, goal_state)

        if plan is None:
            return {'error': 'No plan found', 'plan': []}

        return {
            'diagnosis':     diagnosis,
            'urgency':       urgency,
            'initial_state': sorted(initial_state),
            'goal_state':    sorted(goal_state),
            'steps':         len(plan),
            'total_duration': self._estimate_duration(plan),
            'plan': [
                {
                    'step':     i+1,
                    'action':   a['name'],
                    'duration': a['duration'],
                    'cost':     a['cost']
                }
                for i, a in enumerate(plan)
            ]
        }

    def _estimate_duration(self, plan: List[Dict]) -> str:
        durations = [a['duration'] for a in plan]
        return f"{len(plan)} actions | see individual durations"

    def analyze(self, percept) -> Dict:
        """
        Module interface for the agent.

        IMPORTANT ARCHITECTURAL NOTE: unlike Modules 2-6, this module
        cannot meaningfully analyze a raw PatientPercept alone -- per
        the manual's own "Big Picture" diagram, the Planner sits
        DOWNSTREAM of the other 5 diagnostic modules, consuming their
        AGGREGATED diagnosis + urgency, not the raw patient symptoms.
        A PatientPercept has no 'diagnosis' or 'urgency' field, so
        there is no real diagnosis available at the point agent.py's
        think() loop calls .analyze() on every registered module in
        parallel.

        The original version of this file worked around this by
        hardcoding a fake 'flu'/'MEDIUM' plan regardless of the actual
        patient -- which produces a plan that has nothing to do with
        the real patient being processed.

        RECOMMENDATION: don't register this module the same way as
        Modules 2-6 via agent.register_module(). Instead, call
        planner.create_treatment_plan(diagnosis, urgency) directly
        from app.py (or a new step in agent.act()) AFTER the other
        modules' results have been aggregated into a final diagnosis
        and urgency. This analyze() method is kept only so the module
        can still be registered/tested like the others if needed, but
        it should be treated as a placeholder, not the real
        integration path.
        """
        result = self.create_treatment_plan('flu', 'MEDIUM')
        result['summary']    = (f"[PLACEHOLDER] Plan: {result['steps']} steps "
                                f"generated for 'flu' -- see analyze() "
                                f"docstring, this should not be used as the "
                                f"real integration path")
        result['diagnosis']  = 'flu'
        result['confidence'] = 0.7
        return result


# ============================================================
# STANDALONE TEST
# Matches the manual's own test example, plus a full sweep of all
# 15 diseases (including the CRITICAL urgency path) to confirm every
# one actually finds a plan.
# ============================================================
if __name__ == "__main__":
    planner = TreatmentPlanner()

    # Test for COVID-19 case (manual's own example)
    plan = planner.create_treatment_plan('covid19', 'HIGH')
    print(f"Diagnosis : {plan['diagnosis']}")
    print(f"Plan Steps: {plan['steps']}")
    print()
    for step in plan['plan']:
        print(f"  Step {step['step']:2d}: {step['action']:<30} [{step['duration']}]")

    # Full sweep: confirm every one of the 15 diseases finds a valid
    # plan, at both a normal urgency and CRITICAL urgency.
    print("\n" + "=" * 60)
    print("Full 15-disease sweep (HIGH and CRITICAL urgency)")
    print("=" * 60)
    diseases = ['flu', 'covid19', 'common_cold', 'dengue', 'malaria',
                'typhoid', 'pneumonia', 'migraine', 'tuberculosis',
                'hepatitis_a', 'urinary_tract_infection', 'sinusitis',
                'gastroenteritis', 'bronchitis', 'tonsillitis']

    for disease in diseases:
        for urgency in ['HIGH', 'CRITICAL']:
            result = planner.create_treatment_plan(disease, urgency)
            status = "OK" if 'error' not in result else "FAILED"
            steps = result.get('steps', 0)
            print(f"  {disease:<25} {urgency:<10} -> {status} ({steps} steps)")