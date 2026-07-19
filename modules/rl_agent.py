"""
modules/rl_agent.py

REINFORCEMENT LEARNING MODULE
===============================

WHAT THIS MODULE DOES
----------------------
agent.py's own docstring describes the system as a:
    "Model-Based + Goal-Based + Learning" agent

...but nothing in Modules 1-7 actually LEARNS from outcomes. The
`performance_score` in HealthcareDiagnosticAgent just accumulates -- it
never feeds back into future decisions. This module is that missing
piece: a Q-LEARNING agent that learns, through repeated trial and reward
feedback, which triage ACTION is correct for a given patient SEVERITY
level.

This deliberately reuses concepts already in your system rather than
inventing new ones:
    - The 4 actions are exactly agent.py's _decide_next_action() outputs:
          MONITOR_AT_HOME, SCHEDULE_FOLLOWUP,
          URGENT_APPOINTMENT, EMERGENCY_REFERRAL
    - The 4 severity levels are exactly Module 6's fuzzy severity labels:
          LOW, MILD/MODERATE (combined), HIGH, CRITICAL

WHY REINFORCEMENT LEARNING (not just a lookup table)?
-------------------------------------------------------
You could hardcode "CRITICAL severity -> EMERGENCY_REFERRAL" directly,
and in fact agent.py already does something similar. The point of doing
this via RL instead is to demonstrate the Week [RL] concept: the agent
ISN'T told the correct mapping up front -- it starts by guessing actions
essentially at random, and only learns the correct severity->action
mapping through a REWARD SIGNAL, the same way a real learning agent
would need to if the "correct" action per severity level weren't
obvious in advance (e.g. if outcomes data was noisy or costs varied).

REWARD DESIGN (this is the part worth explaining in your report):
    - Correct action for the severity level  -> +10
    - UNDER-triage a serious case (e.g. treating CRITICAL as if it were
      LOW)                                    -> -20  (heavily penalized:
                                                          this is the
                                                          dangerous
                                                          mistake)
    - OVER-triage (e.g. treating LOW as CRITICAL) -> -5 (wasteful, but
                                                          far safer than
                                                          under-triage)
    - Adjacent-but-wrong (off by one severity level) -> -2

This asymmetric penalty is intentional and clinically motivated: in
real triage, missing a critical patient is much worse than being overly
cautious with a mild one.

ALGORITHM: Tabular Q-Learning
    Q(state, action) <- Q(state, action) + alpha * [reward +
                          gamma * max(Q(next_state, a')) - Q(state, action)]

Since this is a stateless, single-step triage decision (severity in,
action out, episode ends), gamma's future-reward term contributes
nothing meaningful here (there IS no next_state to speak of) -- but it's
kept in the update rule since it's the standard Q-learning formula
taught in the course, and matters if this module is later extended to
multi-step scenarios (e.g. planning a monitoring sequence over several
follow-up visits, not just one decision).
"""

import random
from typing import Dict, List, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False


# ---------------------------------------------------------------------------
# STATE / ACTION DEFINITIONS
# ---------------------------------------------------------------------------
# States: the 4 urgency levels -- matches agent.py's own _assess_urgency()
# scale EXACTLY (CRITICAL / HIGH / MEDIUM / LOW, per the manual's Module 1
# spec), NOT Module 6's separate 5-level fuzzy severity scale (which uses
# MILD/MODERATE and serves a different, finer-grained purpose). Keeping
# this aligned with agent.py is what lets _decide_next_action() eventually
# call into this learned policy without a naming mismatch.
SEVERITY_STATES: List[str] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Actions: exactly agent.py's _decide_next_action() outputs.
TRIAGE_ACTIONS: List[str] = [
    "MONITOR_AT_HOME",
    "SCHEDULE_FOLLOWUP",
    "URGENT_APPOINTMENT",
    "EMERGENCY_REFERRAL",
]

# The "correct" action index for each severity state -- used only to
# COMPUTE REWARDS during training, never given directly to the agent as
# an answer. This mirrors how, in a real system, you'd have historical
# outcome labels (did the patient actually need emergency care?) without
# ever hardcoding the policy itself.
CORRECT_ACTION_INDEX: Dict[str, int] = {
    "LOW": 0,        # MONITOR_AT_HOME
    "MEDIUM": 1,     # SCHEDULE_FOLLOWUP
    "HIGH": 2,       # URGENT_APPOINTMENT
    "CRITICAL": 3,   # EMERGENCY_REFERRAL
}


# ---------------------------------------------------------------------------
# GYMNASIUM ENVIRONMENT
# ---------------------------------------------------------------------------

class TriageEnv(gym.Env if _GYM_AVAILABLE else object):
    """
    A minimal Gymnasium environment simulating one triage decision.

    Each episode:
        1. A random patient severity is presented (the "observation").
        2. The agent picks one of 4 actions.
        3. The environment returns a reward based on how appropriate
           that action was for the true severity, using the asymmetric
           penalty scheme described in this file's module docstring.
        4. The episode ends immediately (single-step decision).
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(len(SEVERITY_STATES))
        self.action_space = spaces.Discrete(len(TRIAGE_ACTIONS))
        self._current_state_idx: int = 0

    def reset(self, seed=None, options=None):
        if _GYM_AVAILABLE:
            super().reset(seed=seed)
        self._current_state_idx = random.randrange(len(SEVERITY_STATES))
        return self._current_state_idx, {}

    def step(self, action: int):
        severity = SEVERITY_STATES[self._current_state_idx]
        correct_action = CORRECT_ACTION_INDEX[severity]

        distance = abs(action - correct_action)

        if distance == 0:
            reward = 10.0
        elif action < correct_action:
            # Under-triaged: treated a more severe case too lightly.
            # Penalty scales with how much more severe the true case was.
            reward = -20.0 * distance
        else:
            # Over-triaged: treated a milder case too seriously.
            # Costly (wasted resources) but far safer than under-triage.
            reward = -5.0 * distance

        terminated = True   # single-step episode
        truncated = False
        info = {"severity": severity, "correct_action": TRIAGE_ACTIONS[correct_action]}

        return self._current_state_idx, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# TABULAR Q-LEARNING AGENT
# ---------------------------------------------------------------------------

class QLearningTriageAgent:
    """
    Learns a severity -> action policy purely from reward feedback,
    using the standard Q-learning update rule with epsilon-greedy
    exploration.
    """

    def __init__(self,
                 n_states: int = len(SEVERITY_STATES),
                 n_actions: int = len(TRIAGE_ACTIONS),
                 alpha: float = 0.1,      # learning rate
                 gamma: float = 0.9,      # discount factor
                 epsilon: float = 1.0,    # exploration rate (starts high)
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.05):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions

    def choose_action(self, state: int, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection:
            - With probability epsilon, pick a RANDOM action (exploration).
            - Otherwise, pick the action with the highest known Q-value
              (exploitation).

        `greedy=True` forces pure exploitation -- used at evaluation time,
        once training is done and we want the agent's best learned policy,
        not further exploration.
        """
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Standard Q-learning (Bellman) update:
            Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]

        Since this environment's episodes are single-step (done=True
        always), the (1 - done) term zeroes out the future-reward
        component -- there IS no meaningful next action to bootstrap
        from in this version of the problem.
        """
        best_next_q = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Gradually shift from exploration to exploitation as training progresses."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> Dict[str, str]:
        """Return the learned best action for every severity state, human-readable."""
        return {
            SEVERITY_STATES[s]: TRIAGE_ACTIONS[int(np.argmax(self.q_table[s]))]
            for s in range(len(SEVERITY_STATES))
        }


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

def train(episodes: int = 2000, verbose: bool = True) -> Tuple[QLearningTriageAgent, List[float]]:
    """
    Train the Q-learning agent over many simulated triage episodes.

    Returns the trained agent and a list of per-episode rewards (useful
    for plotting a learning curve in evaluation/visualizations.py).
    """
    env = TriageEnv()
    agent = QLearningTriageAgent()
    reward_history: List[float] = []

    for episode in range(episodes):
        state, _ = env.reset()
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        agent.update(state, action, reward, next_state, done=terminated)
        agent.decay_epsilon()
        reward_history.append(reward)

        if verbose and (episode + 1) % 500 == 0:
            recent_avg = np.mean(reward_history[-500:])
            print(f"Episode {episode + 1:5d} | "
                  f"epsilon={agent.epsilon:.3f} | "
                  f"avg reward (last 500)={recent_avg:.2f}")

    return agent, reward_history


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not _GYM_AVAILABLE:
        print("gymnasium is not installed. Run: pip install gymnasium")
    else:
        print("Training Q-learning triage agent...\n")
        trained_agent, rewards = train(episodes=2000, verbose=True)

        print("\nLearned policy (severity -> action):")
        policy = trained_agent.get_policy()
        for severity in SEVERITY_STATES:
            correct = TRIAGE_ACTIONS[CORRECT_ACTION_INDEX[severity]]
            learned = policy[severity]
            status = "correct" if learned == correct else "INCORRECT"
            print(f"  {severity:<10} -> {learned:<20} ({status})")

        print("\nFinal Q-table (rows=severity, cols=action):")
        print("             ", "  ".join(f"{a:<20}" for a in TRIAGE_ACTIONS))
        for i, severity in enumerate(SEVERITY_STATES):
            row = "  ".join(f"{q:>18.2f}  " for q in trained_agent.q_table[i])
            print(f"{severity:<12} {row}")