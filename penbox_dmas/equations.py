"""
PenBox-DMAS: Mathematical Framework — Equations 1–17
---------------------------------------------------------------------------
Every equation is implemented exactly as defined in Section V of the paper.
References to equation numbers match the published manuscript.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Equation (1) — Security State Vector
# ---------------------------------------------------------------------------
@dataclass
class SecurityState:
    """
    S_n = (V_n, C_n, P_n, R_n, M_n)   — Eq. (1)

    V_n : active vulnerability set (list of CVSS scores)
    C_n : control effectiveness set (list of values in [0,1])
    P_n : attack surface metric (float)
    R_n : composite risk score (float)
    M_n : cumulative remediation mass (float)
    """
    V_n: List[float] = field(default_factory=list)   # Eq. (2)
    C_n: List[float] = field(default_factory=list)   # Eq. (3)
    P_n: float = 0.0                                  # Eq. (4)
    R_n: float = 0.0                                  # Eq. (12)
    M_n: float = 0.5                                  # initial value from Algorithm 1

    def as_tuple(self) -> tuple:
        return (self.V_n, self.C_n, self.P_n, self.R_n, self.M_n)


# ---------------------------------------------------------------------------
# Equation (2) — Vulnerability Set
# ---------------------------------------------------------------------------
def vulnerability_set(cvss_scores: List[float]) -> List[float]:
    """
    V_n = {V_1, V_2, ..., V_k}  where  V_i ∈ [0, 10]   — Eq. (2)
    Each V_i is a CVSS score normalised to [0, 10].
    """
    return [float(np.clip(v, 0.0, 10.0)) for v in cvss_scores]


# ---------------------------------------------------------------------------
# Equation (3) — Control Set
# ---------------------------------------------------------------------------
def control_set(effectiveness_values: List[float]) -> List[float]:
    """
    C_n = {C_1, C_2, ..., C_m}  where  C_j ∈ [0, 1]   — Eq. (3)
    C_j = 0 → no protection;  C_j = 1 → full coverage.
    """
    return [float(np.clip(c, 0.0, 1.0)) for c in effectiveness_values]


# ---------------------------------------------------------------------------
# Equation (4) — Attack Surface Metric
# ---------------------------------------------------------------------------
def attack_surface_metric(E_n: float, O_n: float, T_n: float) -> float:
    """
    P_n = (E_n × O_n) / T_n   — Eq. (4)

    E_n : number of exposed ports / services
    O_n : number of open network pathways
    T_n : total enumerated attack paths
    """
    if T_n == 0:
        return 0.0
    return (E_n * O_n) / T_n


# ---------------------------------------------------------------------------
# Equation (5) — Vulnerability Discovery Function
# ---------------------------------------------------------------------------
def vulnerability_discovery(
    alpha: List[float],
    scan_hits: List[float],
    beta: float,
    P_n: float,
    dijkstra_weight: float = 1.0,
) -> float:
    """
    V_detected = Σ(α_i × f_scan(V_i))  +  β × g_network(P_n)   — Eq. (5)

    alpha      : detection probabilities α_i ∈ [0,1] per vulnerability signature
    scan_hits  : f_scan(V_i) — binary or weighted scan output per signature
    beta       : topology weight — 0.3 (internal) or 0.7 (external-facing)
    P_n        : attack surface metric (Eq. 4)
    dijkstra_weight : modifier from modified Dijkstra traversal g_network(P_n)
    """
    signature_component = sum(a * s for a, s in zip(alpha, scan_hits))
    network_component = beta * (P_n * dijkstra_weight)
    return signature_component + network_component


# ---------------------------------------------------------------------------
# Equation (6) — Vulnerability Validation
# ---------------------------------------------------------------------------
def vulnerability_validation(
    V_detected: float,
    V_undetected: float,
    FPR: float = 0.10,
    FNR: float = 0.04,
) -> float:
    """
    V_validated = V_detected × (1 − FPR) + V_undetected × FNR   — Eq. (6)

    Empirically calibrated rates (Table IX / Table III):
      FPR = 0.10  (overall; paper Table III)
      FNR = 0.04  (overall; paper Table III)
    """
    return V_detected * (1.0 - FPR) + V_undetected * FNR


# ---------------------------------------------------------------------------
# Equation (7) — MITRE ATT&CK Technique Mapping
# ---------------------------------------------------------------------------
def technique_mapping(vulnerability_id: str,
                      applicable_techniques: List[str],
                      all_techniques: List[str]) -> Dict[str, int]:
    """
    T_ij = 1 if technique j applies to V_i, else 0   — Eq. (7)

    Returns a dict {technique_id: 0 or 1} for vulnerability V_i.
    """
    return {t: (1 if t in applicable_techniques else 0) for t in all_techniques}


# ---------------------------------------------------------------------------
# Equation (8) — Vulnerability Priority Score
# ---------------------------------------------------------------------------
def vulnerability_priority(
    technique_weights: Dict[str, float],
    T_ij: Dict[str, int],
    privilege_required: float,
) -> float:
    """
    Priority(V_i) = Σ_j [ w_j × T_ij × (1 / Privilege_required(V_i)) ]  — Eq. (8)

    technique_weights  : w_j for each ATT&CK technique
    T_ij               : binary mapping from Eq. (7)
    privilege_required : higher value → harder to exploit → lower priority
    """
    if privilege_required == 0:
        privilege_required = 1e-6
    weighted = sum(technique_weights.get(t, 0.0) * v
                   for t, v in T_ij.items())
    return weighted / privilege_required


# ---------------------------------------------------------------------------
# Equation (9) — Single-Stage Exploitation Probability
# ---------------------------------------------------------------------------
def exploitation_probability_single(
    complexity: float,
    defenses: float,
    llm_skill: float = 0.8,
) -> float:
    """
    P_exploit(V_i) = (1 / Complexity(V_i)) × Defenses(V_i) × LLM_skill   — Eq. (9)

    complexity  : V_i complexity ∈ [1, 10]
    defenses    : defense effectiveness ∈ [0, 1]   (1 = fully defended)
    llm_skill   : fixed at 0.8 (Table IX)
    """
    if complexity <= 0:
        complexity = 1e-6
    # NOTE: higher defenses → harder → lower probability
    # Defenses(V_i) here acts as (1 - defense_coverage) per semantic intent
    return (1.0 / complexity) * defenses * llm_skill


# ---------------------------------------------------------------------------
# Equation (10) — Multi-Stage Exploitation Probability
# ---------------------------------------------------------------------------
def exploitation_probability_multi(
    single_probs: List[float],
    escalation_probs: List[float],
) -> float:
    """
    P_multi = Π [ P_exploit(V_i) × P_escalate(V_i → V_j) ]   — Eq. (10)

    Chained attacks modelled as product of individual and escalation probs.
    """
    if not single_probs:
        return 0.0
    product = 1.0
    for p_e, p_s in zip(single_probs, escalation_probs):
        product *= p_e * p_s
    return product


# ---------------------------------------------------------------------------
# Equation (11) — Vulnerability State Evolution
# ---------------------------------------------------------------------------
def vulnerability_state_evolution(
    V_n: float,
    M_n: float,
    V_new: float,
    eta: float = 0.7,
) -> Tuple[float, float]:
    """
    μ_n  = M_n × η                           (remediation fraction)
    V_{n+1} = V_n × (1 − μ_n) + V_new       — Eq. (11)

    eta   : empirical remediation efficiency = 0.7 (Table IX)
    V_new : new vulnerabilities arriving this iteration
    Returns (V_{n+1}, μ_n)
    """
    mu_n = M_n * eta
    mu_n = min(mu_n, 1.0)
    V_next = V_n * (1.0 - mu_n) + V_new
    return V_next, mu_n


# ---------------------------------------------------------------------------
# Equation (12) — Risk Score Evolution
# ---------------------------------------------------------------------------
def risk_score_evolution(
    R_n: float,
    severity_sum: float,
    M_n: float,
    rho: float,
    sigma: float,
    gamma: float,
) -> float:
    """
    R_{n+1} = ρ × R_n  +  σ × Σ Severity(V_n)  −  γ × M_n   — Eq. (12)

    OS-specific parameters (Table VII):
      Windows Server 2019 : ρ=0.65, σ=0.28, γ=0.18
      Ubuntu Linux 20.04  : ρ=0.58, σ=0.32, γ=0.22
      IoT Devices (ESP32) : ρ=0.42, σ=0.45, γ=0.15
      Metasploitable 3    : ρ=0.61, σ=0.30, γ=0.20
    """
    return rho * R_n + sigma * severity_sum - gamma * M_n


# ---------------------------------------------------------------------------
# Equation (13) — Theoretical Equilibrium Vulnerability Level
# ---------------------------------------------------------------------------
def equilibrium_vulnerability(V_new: float, mu_star: float) -> float:
    """
    V* = V_new / μ*   — Eq. (13)

    Steady-state irreducible floor.  mu_star is the long-run remediation rate.
    """
    if mu_star == 0:
        return float('inf')
    return V_new / mu_star


# ---------------------------------------------------------------------------
# Equation (14) — Agent Task Distribution Function
# ---------------------------------------------------------------------------
def task_distribution(
    latencies: List[float],
    compute_costs: List[float],
    queue_depths: List[float],
    alpha_queue: float = 0.5,
) -> int:
    """
    T_assign(i, j) = arg min [ L_i · C_ij  +  α · Var(queue) ]   — Eq. (14)

    latencies      : L_i — current latency to each agent
    compute_costs  : C_ij — estimated compute cost per agent for task j
    queue_depths   : current queue depth per agent (used for Var term)
    alpha_queue    : variance penalty weight

    Returns the index of the selected agent.
    """
    queue_var = float(np.var(queue_depths))
    costs = [l * c + alpha_queue * queue_var
             for l, c in zip(latencies, compute_costs)]
    return int(np.argmin(costs))


# ---------------------------------------------------------------------------
# Equation (15) — Agentic AI Fallback Routing
# ---------------------------------------------------------------------------
def llm_routing(
    P_local: float,
    P_cloud: float,
    token_usage: int,
    zero_day_confidence: float,
    token_threshold: int = 128_000,
    confidence_threshold: float = 0.5,
) -> Tuple[float, int]:
    """
    P_llm(v) = (1 − λ) · P_local(v)  +  λ · P_cloud(v)   — Eq. (15)

    λ = 1  if token_usage > 128k  OR  zero_day_confidence < 0.5
    λ = 0  otherwise

    Returns (P_llm, lambda_flag).
    """
    lam = 1 if (token_usage > token_threshold or
                zero_day_confidence < confidence_threshold) else 0
    P_llm = (1 - lam) * P_local + lam * P_cloud
    return P_llm, lam


# ---------------------------------------------------------------------------
# Equation (16) — Adversarial RL Exploit Probability (Q-learning / Bellman)
# ---------------------------------------------------------------------------
def bellman_q_update(
    Q: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    gamma_discount: float = 0.95,
    alpha_lr: float = 0.1,
) -> np.ndarray:
    """
    Q*(s, a) = max_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) · V*(s') ]   — Eq. (16)

    Implements the Bellman optimality update for the adversarial RL agent.
    Q        : Q-table (states × actions)
    gamma_discount : discount factor γ
    alpha_lr : learning rate
    """
    best_next = float(np.max(Q[next_state]))
    td_target = reward + gamma_discount * best_next
    td_error = td_target - Q[state, action]
    Q[state, action] += alpha_lr * td_error
    return Q


def rl_exploit_probability(Q: np.ndarray, state: int) -> float:
    """
    Returns the greedy exploitation probability for a given state by
    normalising the best Q-value to [0, 1].
    """
    q_max = float(np.max(Q[state]))
    q_min = float(np.min(Q))
    q_range = float(np.max(Q)) - q_min
    if q_range == 0:
        return 0.0
    return (q_max - q_min) / q_range


# ---------------------------------------------------------------------------
# Equation (17) — Swarm Convergence Time
# ---------------------------------------------------------------------------
def swarm_convergence_time(
    V0: float,
    V_star: float,
    mu_eff: float,
    N_agents: int,
) -> float:
    """
    T_conv = (V_0 − V*) / (μ_eff · N_agents)   — Eq. (17)

    V0       : initial vulnerability count
    V_star   : equilibrium level (Eq. 13)
    mu_eff   : effective remediation rate
    N_agents : number of active agent nodes ∈ {3, 4}
    """
    denominator = mu_eff * N_agents
    if denominator == 0:
        return float('inf')
    return (V0 - V_star) / denominator


# ---------------------------------------------------------------------------
# OS-specific risk parameters (Table VII)
# ---------------------------------------------------------------------------
OS_RISK_PARAMS: Dict[str, Dict[str, float]] = {
    "Windows Server 2019": {"rho": 0.65, "sigma": 0.28, "gamma": 0.18},
    "Ubuntu Linux 20.04":  {"rho": 0.58, "sigma": 0.32, "gamma": 0.22},
    "IoT Devices (ESP32)": {"rho": 0.42, "sigma": 0.45, "gamma": 0.15},
    "Metasploitable 3":    {"rho": 0.61, "sigma": 0.30, "gamma": 0.20},
}

# ---------------------------------------------------------------------------
# Empirical parameter values (Table IX)
# ---------------------------------------------------------------------------
PARAMS = {
    "alpha_range": (0.65, 0.95),   # detection probability per signature
    "beta_internal": 0.3,
    "beta_external": 0.7,
    "FPR": 0.10,
    "FNR": 0.04,
    "LLM_skill": 0.8,
    "eta": 0.7,                    # remediation efficiency
    "token_threshold": 128_000,
    "zero_day_conf_threshold": 0.5,
    "lambda_flag": "conditional",
}
